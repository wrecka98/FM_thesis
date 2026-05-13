import os
import shutil
import argparse
import hashlib
import ctypes
from typing import Optional
from contextlib import contextmanager

import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd

import pydicom
import dicomsdl
import torch
import timm

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from nvidia.dali.backend import TensorGPU, TensorListGPU


# -----------------------------
# tqdm + joblib integration
# (shows progress on COMPLETED jobs, not just submitted)
# -----------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


# -----------------------------
# Constants / dtype mapping
# -----------------------------
JPEG2000_UID = "1.2.840.10008.1.2.4.90"

# UINT16 uses int16 as a transport buffer (we reinterpret bits to unsigned after copy)
to_torch_type = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.UINT16: torch.int16,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64,
}


def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """Copy contents of DALI tensor to PyTorch Tensor."""
    dali_type = to_torch_type[dali_tensor.dtype]
    assert dali_type == arr.dtype, f"DALI dtype != torch dtype: {dali_type} vs {arr.dtype}"
    assert dali_tensor.shape() == list(arr.size()), f"Shapes do not match: {dali_tensor.shape()} vs {list(arr.size())}"

    cuda_stream = types._raw_cuda_stream(cuda_stream)
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())

    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


# -----------------------------
# Path helpers (support absolute image_path)
# -----------------------------
def normalize_path(p: str) -> str:
    """Normalize separators; preserve absolute paths."""
    p = str(p).strip()
    p = p.replace("\\", os.sep).replace("/", os.sep)
    return p


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_key(s: str) -> str:
    """Collision-resistant key from a string (path)."""
    base = os.path.splitext(os.path.basename(s))[0]
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{h}"


def resolve_dcm_path(dicom_root: str, p: str) -> str:
    """
    If p is absolute -> return it.
    If p is relative -> join with dicom_root.
    """
    p = normalize_path(p)
    if os.path.isabs(p):
        return p
    return os.path.join(dicom_root, p.lstrip("/\\"))


def safe_relpath(dcm_path: str, dicom_root: str) -> str:
    """
    Compute a safe relative path for writing outputs under output_root.
    If dcm_path is not under dicom_root, fall back to a hashed filename.
    """
    dcm_path = os.path.abspath(dcm_path)
    dicom_root = os.path.abspath(dicom_root)

    try:
        rel = os.path.relpath(dcm_path, dicom_root)
    except Exception:
        rel = os.path.basename(dcm_path)

    # prevent output escaping output_root (../)
    if rel.startswith(".."):
        rel = make_key(dcm_path) + ".dcm"
    return rel


# -----------------------------
# Missing paths -> CSV (NEW)
# -----------------------------
def write_missing_paths_csv_and_filter(
    df: pd.DataFrame,
    dicom_root: str,
    path_col: str,
    output_root: str,
    missing_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    - Resolves each image path (absolute or relative) to an absolute DICOM file path
    - Writes *all* missing paths to a CSV
    - Returns df filtered to only existing paths
    """
    os.makedirs(output_root, exist_ok=True)
    if missing_csv_path is None:
        missing_csv_path = os.path.join(output_root, "missing_paths.csv")

    # Keep original values too
    orig = df[path_col].astype(str).tolist()
    norm = [normalize_path(x) for x in orig]
    resolved = [resolve_dcm_path(dicom_root, x) for x in norm]
    exists = [os.path.exists(x) for x in resolved]

    missing_df = pd.DataFrame(
        {
            path_col: orig,
            "normalized_path": norm,
            "resolved_path": resolved,
            "exists": exists,
        }
    )
    missing_only = missing_df.loc[~missing_df["exists"]].copy()
    missing_only.to_csv(missing_csv_path, index=False)

    print(f"Missing paths: {len(missing_only)} / {len(df)}")
    print(f"Missing paths CSV saved to: {missing_csv_path}")

    # Filter df
    return df.loc[exists].reset_index(drop=True)


# -----------------------------
# DICOM windowing helpers (preserve viewer-like contrast)
# -----------------------------
def _first_number(x, default=None):
    if x is None:
        return default
    try:
        if isinstance(x, (list, tuple)):
            return float(x[0])
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)) and len(x) > 0:
            return float(x[0])
        return float(x)
    except Exception:
        return default


def window_uint16_to_uint8_torch(
    img_u16_float: torch.Tensor,
    ds,
    fallback: str = "percentile",  # "percentile" or "minmax"
    p_lo: float = 0.01,
    p_hi: float = 0.99,
) -> torch.Tensor:
    """
    Apply display mapping:
      1) RescaleSlope/Intercept
      2) WindowCenter/WindowWidth if present
      3) else fallback (percentile/minmax)
    Returns float image in [0,255].
    """
    slope = _first_number(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _first_number(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    x = img_u16_float * float(slope) + float(intercept)

    wc = _first_number(getattr(ds, "WindowCenter", None), None)
    ww = _first_number(getattr(ds, "WindowWidth", None), None)

    if wc is not None and ww is not None and ww > 1e-6:
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        x = x.clamp(low, high)
        x = (x - low) / (high - low + 1e-6) * 255.0
        return x.clamp(0.0, 255.0)

    if fallback == "minmax":
        mn = x.min()
        mx = x.max()
        x = (x - mn) / (mx - mn + 1e-6) * 255.0
        return x.clamp(0.0, 255.0)

    lo = torch.quantile(x, p_lo)
    hi = torch.quantile(x, p_hi)
    x = x.clamp(lo, hi)
    x = (x - lo) / (hi - lo + 1e-6) * 255.0
    return x.clamp(0.0, 255.0)


def window_uint16_to_uint8_numpy(
    img_u16_float: np.ndarray,
    ds,
    fallback: str = "percentile",
    p_lo: float = 0.01,
    p_hi: float = 0.99,
) -> np.ndarray:
    slope = _first_number(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _first_number(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    x = img_u16_float * float(slope) + float(intercept)

    wc = _first_number(getattr(ds, "WindowCenter", None), None)
    ww = _first_number(getattr(ds, "WindowWidth", None), None)

    if wc is not None and ww is not None and ww > 1e-6:
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        x = np.clip(x, low, high)
        x = (x - low) / (high - low + 1e-6) * 255.0
        return np.clip(x, 0.0, 255.0)

    if fallback == "minmax":
        mn = float(np.min(x))
        mx = float(np.max(x))
        x = (x - mn) / (mx - mn + 1e-6) * 255.0
        return np.clip(x, 0.0, 255.0)

    lo = float(np.quantile(x, p_lo))
    hi = float(np.quantile(x, p_hi))
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6) * 255.0
    return np.clip(x, 0.0, 255.0)


def transport_int16_to_u16_float(img_i16: torch.Tensor) -> torch.Tensor:
    """Fix wraparound by reinterpreting bits as unsigned, return float32 [0..65535]."""
    return (img_i16.to(torch.int32) & 0xFFFF).to(torch.float32)


def maybe_hw(img: torch.Tensor) -> torch.Tensor:
    """Handle HxWx1."""
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    return img


# -----------------------------
# JPEG2000 extraction + DALI decode
# -----------------------------
def convert_dicom_to_j2k(dicom_root: str, image_path: str, save_folder: str = "") -> Optional[str]:
    """If JPEG2000, extract embedded JP2 stream to save_folder and return path; else None."""
    dcm_path = resolve_dcm_path(dicom_root, image_path)

    # Missing paths are already logged; just skip here
    if not os.path.exists(dcm_path):
        return None

    try:
        dcmfile = pydicom.dcmread(dcm_path, stop_before_pixels=False)
    except Exception:
        return None

    try:
        ts = str(dcmfile.file_meta.TransferSyntaxUID)
    except Exception:
        return None

    if ts != JPEG2000_UID:
        return None

    with open(dcm_path, "rb") as fp:
        raw = DicomBytesIO(fp.read())
        ds = pydicom.dcmread(raw)

    offset = ds.PixelData.find(b"\x00\x00\x00\x0C")
    if offset < 0:
        return None

    hackedbitstream = bytearray()
    hackedbitstream.extend(ds.PixelData[offset:])

    # key based on safe relative path (stable + avoids collisions)
    rel_for_key = safe_relpath(dcm_path, dicom_root)
    key = make_key(rel_for_key)

    jp2_path = os.path.join(save_folder, f"{key}.jp2")
    ensure_parent_dir(jp2_path)

    with open(jp2_path, "wb") as f:
        f.write(hackedbitstream)

    return jp2_path


@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.ANY_DATA,
        dtype=DALIDataType.UINT16,
    )
    return images


# -----------------------------
# Breast extraction (unchanged)
# -----------------------------
def np_CountUpContinuingOnes(b_arr):
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def np_ExtractBreast(img):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)
    height, _ = img.shape

    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    return img_copy[row_ind][:, col_ind]


def torch_CountUpContinuingOnes(b_arr):
    left = torch.arange(len(b_arr), device=b_arr.device)
    left[b_arr > 0] = 0
    left = torch.cummax(left, dim=-1)[0]

    rev_arr = torch.flip(b_arr, [-1])
    right = torch.arange(len(rev_arr), device=b_arr.device)
    right[rev_arr > 0] = 0
    right = torch.cummax(right, dim=-1)[0]
    right = len(rev_arr) - 1 - torch.flip(right, [-1])

    return right - left - 1


def torch_ExtractBreast(img_ori):
    img = torch.where(img_ori <= 40, torch.zeros_like(img_ori), img_ori)
    height, _ = img.shape

    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].to(torch.float32).std(dim=0) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    col_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].to(torch.float32).std(dim=1) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    row_ind = torch.where(continuing_ones == continuing_ones.max())[0]

    return img_ori[row_ind][:, col_ind]


# -----------------------------
# CPU path
# -----------------------------
def save_array(
    dicom_root,
    SIZE,
    output_root,
    image_path,
    fix_monochrome=True,
    skip_if_exists=True,
    fallback="percentile",
    p_lo=0.01,
    p_hi=0.99,
):
    dcm_path = resolve_dcm_path(dicom_root, image_path)
    if not os.path.exists(dcm_path):
        return

    rel_out = safe_relpath(dcm_path, dicom_root)
    out_png = os.path.join(output_root, os.path.splitext(rel_out)[0] + ".png")
    ensure_parent_dir(out_png)

    if skip_if_exists and os.path.exists(out_png):
        return

    # Read metadata for transfer syntax + photometric + windowing tags
    try:
        ds_meta = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        ts = str(ds_meta.file_meta.TransferSyntaxUID)
        photometric = getattr(ds_meta, "PhotometricInterpretation", "")
    except Exception:
        return

    if ts == JPEG2000_UID:
        # best-effort CPU decode; requires JPEG2000 handler installed for pydicom
        try:
            ds = pydicom.dcmread(dcm_path)
            data = ds.pixel_array
            if data.ndim == 3:
                data = data[0]
            data = data.astype(np.float32)
            photometric = getattr(ds, "PhotometricInterpretation", photometric)
        except Exception:
            return

        data = data[5:-5, 5:-5]
        data_u8f = window_uint16_to_uint8_numpy(data, ds, fallback=fallback, p_lo=p_lo, p_hi=p_hi)

        if fix_monochrome and photometric == "MONOCHROME1":
            data_u8f = 255.0 - data_u8f

        data_u8 = data_u8f.astype(np.uint8)
        data_u8 = np_ExtractBreast(data_u8)
        data_u8 = cv2.resize(data_u8, SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_png, data_u8)
        return

    # Non-J2K: dicomsdl
    try:
        dcm = dicomsdl.open(dcm_path)
        data = dcm.pixelData()
        if data.ndim == 3:
            data = data[0]
        data = data.astype(np.float32)
        info = dcm.getPixelDataInfo() or {}
        photometric_sdl = info.get("PhotometricInterpretation", photometric)
    except Exception:
        return

    data = data[5:-5, 5:-5]
    data_u8f = window_uint16_to_uint8_numpy(data, ds_meta, fallback=fallback, p_lo=p_lo, p_hi=p_hi)

    if fix_monochrome and photometric_sdl == "MONOCHROME1":
        data_u8f = 255.0 - data_u8f

    data_u8 = data_u8f.astype(np.uint8)
    data_u8 = np_ExtractBreast(data_u8)
    data_u8 = cv2.resize(data_u8, SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_png, data_u8)


# -----------------------------
# Main conversion routine
# -----------------------------
def convert_dicom_to_png(
    SIZE,
    dicom_root,
    output_root,
    j2k_folder,
    df,
    path_col="image_path",
    n_jobs=2,
    dali_threads=2,
    device_id=0,
    fix_monochrome=True,
    fallback="percentile",
    p_lo=0.01,
    p_hi=0.99,
):
    df = df.copy()
    df[path_col] = df[path_col].astype(str).apply(normalize_path)

    os.makedirs(output_root, exist_ok=True)
    print("Number of images to process:", len(df))

    # Build key mapping based on safe relpath so GPU JP2 <-> DICOM mapping is stable
    dcm_paths = [resolve_dcm_path(dicom_root, p) for p in df[path_col].values]
    rel_outs = [safe_relpath(p, dicom_root) for p in dcm_paths]
    keys = [make_key(r) for r in rel_outs]

    key_to_dcmpath = dict(zip(keys, dcm_paths))
    key_to_relout = dict(zip(keys, rel_outs))

    # -------- GPU path (JPEG2000) --------
    if torch.cuda.is_available():
        N_CHUNKS = 4 if len(df) > 100 else 1
        CHUNKS = [(len(df) / N_CHUNKS * k, len(df) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
        CHUNKS = np.array(CHUNKS).astype(int)

        for chunk in tqdm(CHUNKS, desc="GPU stage: chunks"):
            os.makedirs(j2k_folder, exist_ok=True)
            image_paths = df.iloc[chunk[0] : chunk[1]][path_col].values

            # Progress bar for JP2 extraction (COMPLETED tasks)
            with tqdm_joblib(tqdm(total=len(image_paths), desc="GPU stage: extract JP2", leave=False)):
                jp2_paths = Parallel(n_jobs=n_jobs)(
                    delayed(convert_dicom_to_j2k)(dicom_root, p, save_folder=j2k_folder)
                    for p in image_paths
                )

            jp2_paths = [p for p in jp2_paths if p is not None]
            if not jp2_paths:
                shutil.rmtree(j2k_folder, ignore_errors=True)
                continue

            pipe = j2k_decode_pipeline(
                jp2_paths, batch_size=1, num_threads=dali_threads, device_id=device_id, debug=False
            )
            pipe.build()

            # Progress bar for GPU decode+save
            for jp2 in tqdm(jp2_paths, desc="GPU stage: decode+save", leave=False):
                out = pipe.run()
                img = out[0][0]  # DALI tensor (UINT16)

                key = os.path.splitext(os.path.basename(jp2))[0]
                dcm_path = key_to_dcmpath.get(key, None)
                rel_out = key_to_relout.get(key, None)
                if dcm_path is None or rel_out is None:
                    continue

                out_png = os.path.join(output_root, os.path.splitext(rel_out)[0] + ".png")
                ensure_parent_dir(out_png)
                if os.path.exists(out_png):
                    continue

                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    photometric = getattr(ds, "PhotometricInterpretation", "")
                except Exception:
                    continue

                img_i16 = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
                feed_ndarray(img, img_i16, cuda_stream=torch.cuda.current_stream(device=device_id))

                img_f = transport_int16_to_u16_float(img_i16)
                img_f = maybe_hw(img_f)

                # keep your border crop
                img_f = img_f[5:-5, 5:-5]

                # preserve contrast like viewer
                img_u8f = window_uint16_to_uint8_torch(img_f, ds, fallback=fallback, p_lo=p_lo, p_hi=p_hi)

                # photometric inversion in display space
                if fix_monochrome and photometric == "MONOCHROME1":
                    img_u8f = 255.0 - img_u8f

                # keep your "zoom on breast"
                img_u8f = torch_ExtractBreast(img_u8f)

                img_np = img_u8f.detach().clamp(0, 255).cpu().numpy().astype(np.uint8)
                img_np = cv2.resize(img_np, SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(out_png, img_np)

            shutil.rmtree(j2k_folder, ignore_errors=True)

    # -------- CPU path (everything else; shows completion progress) --------
    with tqdm_joblib(tqdm(total=len(df), desc="CPU stage: convert (completed)")):
        _ = Parallel(n_jobs=n_jobs)(
            delayed(save_array)(
                dicom_root,
                SIZE,
                output_root,
                p,
                fix_monochrome=fix_monochrome,
                skip_if_exists=True,
                fallback=fallback,
                p_lo=p_lo,
                p_hi=p_hi,
            )
            for p in df[path_col].values
        )


def main():
    parser = argparse.ArgumentParser(description="Convert DICOMs to PNGs using metadata column image_path.")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--width", type=int, default=912)
    parser.add_argument("--height", type=int, default=1520)

    parser.add_argument(
        "--base_folder",
        type=str,
        default="/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data/NLBS_Data",
        help="Base folder containing metadata CSV and DICOM tree (used for relpath + output structure).",
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help='If not set, uses "<base_folder>/<phase>.csv". Must contain an "image_path" column.',
    )

    parser.add_argument("--output_root", type=str, default=None, help="Folder to write PNGs")
    parser.add_argument("--j2k_folder", type=str, default=None, help="Temp folder for extracted JP2 files")

    parser.add_argument("--path_col", type=str, default="image_path", help="Column name with DICOM file path")
    parser.add_argument("--missing_csv", type=str, default=None, help="Where to save missing paths CSV (default: <output_root>/missing_paths.csv)")

    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--dali_threads", type=int, default=2)
    parser.add_argument("--device_id", type=int, default=0)

    parser.add_argument(
        "--fallback",
        type=str,
        default="percentile",
        choices=["percentile", "minmax"],
        help="If WindowCenter/Width missing, use percentile or minmax scaling.",
    )
    parser.add_argument("--p_lo", type=float, default=0.01)
    parser.add_argument("--p_hi", type=float, default=0.99)

    args = parser.parse_args()

    dicom_root = args.base_folder
    output_root = "/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data/NLBS_Data_png_v1"
    j2k_folder = os.path.join("/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data", "tmp/j2k/")

    metadata_csv = args.metadata_csv or os.path.join(args.base_folder, f"{args.phase}.csv")
    df = pd.read_csv(metadata_csv)

    if args.path_col not in df.columns:
        raise ValueError(f'Metadata CSV must contain a "{args.path_col}" column. Found: {list(df.columns)}')

    # 1) Write missing paths to CSV + 2) filter them out before processing
    df = write_missing_paths_csv_and_filter(
        df=df,
        dicom_root=dicom_root,
        path_col=args.path_col,
        output_root=output_root,
        missing_csv_path=args.missing_csv,
    )

    SIZE = (args.width, args.height)

    print("torch version:", torch.__version__)
    print("timm version:", timm.__version__)
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("dicom_root:", dicom_root)
    print("metadata_csv:", metadata_csv)
    print("output_root:", output_root)
    print("j2k_folder:", j2k_folder)
    print("path_col:", args.path_col)
    print("fallback:", args.fallback, "p_lo:", args.p_lo, "p_hi:", args.p_hi)
    print("images after filtering missing:", len(df))

    convert_dicom_to_png(
        SIZE=SIZE,
        dicom_root=dicom_root,
        output_root=output_root,
        j2k_folder=j2k_folder,
        df=df,
        path_col=args.path_col,
        n_jobs=args.n_jobs,
        dali_threads=args.dali_threads,
        device_id=args.device_id,
        fix_monochrome=True,
        fallback=args.fallback,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
    )


if __name__ == "__main__":
    main()
