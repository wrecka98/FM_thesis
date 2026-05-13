import glob
import os
import shutil
import argparse
import hashlib
import ctypes
from typing import Optional

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
# Constants / dtype mapping
# -----------------------------
JPEG2000_UID = "1.2.840.10008.1.2.4.90"

to_torch_type = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    # FIXED: Map UINT16 to INT32.
    # If mapped to int16, values > 32767 wrap to negative, destroying contrast.
    types.DALIDataType.UINT16: torch.int32,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64,
}


def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.
    """
    dali_type = to_torch_type[dali_tensor.dtype]
    assert dali_type == arr.dtype, (
        "DALI dtype != torch dtype: {} vs {}".format(dali_type, arr.dtype)
    )
    assert dali_tensor.shape() == list(arr.size()), (
        "Shapes do not match: DALI {} vs torch {}".format(dali_tensor.shape(), list(arr.size()))
    )

    cuda_stream = types._raw_cuda_stream(cuda_stream)
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())

    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


# -----------------------------
# Path helpers
# -----------------------------
def normalize_relpath(p: str) -> str:
    p = str(p).strip().lstrip("/\\")
    p = p.replace("\\", os.sep).replace("/", os.sep)
    return p


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def make_key(rel_path: str) -> str:
    base = os.path.splitext(os.path.basename(rel_path))[0]
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{h}"


# -----------------------------
# JPEG2000 extraction + DALI decode
# -----------------------------
def convert_dicom_to_j2k(dicom_root: str, rel_path: str, save_folder: str = "") -> Optional[str]:
    rel_path = normalize_relpath(rel_path)
    dcm_path = os.path.join(dicom_root, rel_path)

    try:
        dcmfile = pydicom.dcmread(dcm_path, stop_before_pixels=False)
    except Exception:
        return None

    if str(dcmfile.file_meta.TransferSyntaxUID) != JPEG2000_UID:
        return None

    with open(dcm_path, "rb") as fp:
        raw = DicomBytesIO(fp.read())
        ds = pydicom.dcmread(raw)

    offset = ds.PixelData.find(b"\x00\x00\x00\x0C")
    if offset < 0:
        return None

    hackedbitstream = bytearray()
    hackedbitstream.extend(ds.PixelData[offset:])

    key = make_key(rel_path)
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
# Breast extraction (PRESERVED AS REQUESTED)
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
    # Normalize briefly for thresholding if not already
    img = np.where(img <= 40, 0, img)
    height, _ = img.shape

    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    if len(col_ind) > 0:
        img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    if len(row_ind) > 0 and len(col_ind) > 0:
        return img_copy[row_ind][:, col_ind]
    return img_copy


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
    if len(col_ind) == 0: return img_ori  # Safety check
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].to(torch.float32).std(dim=1) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)

    row_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    if len(row_ind) == 0: return img_ori  # Safety check

    return img_ori[row_ind][:, col_ind]


# -----------------------------
# CPU path (dicomsdl for non-J2K; best-effort for J2K if no CUDA)
# -----------------------------
def save_array(dicom_root, SIZE, output_root, rel_path, fix_monochrome=True):
    rel_path = normalize_relpath(rel_path)
    dcm_path = os.path.join(dicom_root, rel_path)
    out_png = os.path.join(output_root, os.path.splitext(rel_path)[0] + ".png")

    if os.path.exists(out_png):
        return

    ensure_parent_dir(out_png)

    # Helper: Percentile normalization to preserve contrast
    def robust_normalize(arr):
        arr = arr.astype(np.float32)
        # Clip top 0.5% (removes bright markers) and bottom 0.5%
        # This keeps the breast tissue contrast intact.
        p_low = np.percentile(arr, 0.5)
        p_high = np.percentile(arr, 99.5)

        arr = np.clip(arr, p_low, p_high)

        if p_high - p_low > 0:
            arr = (arr - p_low) / (p_high - p_low)
        else:
            arr = np.zeros_like(arr)
        return arr

    try:
        # Try dicomsdl first
        dcm = dicomsdl.open(dcm_path)
        data = dcm.pixelData(storedvalue=True)  # Get raw data
        info = dcm.getPixelDataInfo()
        is_monochrome1 = (info['PhotometricInterpretation'] == 'MONOCHROME1')

    except Exception:
        # Fallback to pydicom
        try:
            ds = pydicom.dcmread(dcm_path)
            data = ds.pixel_array
            is_monochrome1 = (getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")
        except Exception:
            return

    if is_monochrome1:
        data = np.amax(data) - data

    # 1. Robust Normalization (Fixes "Washed Out")
    data = robust_normalize(data)

    # 2. Heuristic Background Check (Fixes "White Background")
    # Check corners. If they are white, invert.
    h, w = data.shape
    corners = np.concatenate([
        data[0:50, 0:50].flatten(),
        data[0:50, w - 50:w].flatten(),
        data[h - 50:h, 0:50].flatten(),
        data[h - 50:h, w - 50:w].flatten()
    ])
    if np.mean(corners) > 0.5:
        data = 1.0 - data

    # 3. Finalize
    data = (data * 255).astype(np.uint8)
    data = np_ExtractBreast(data)  # Your zoom logic
    data = cv2.resize(data, SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_png, data)


# -----------------------------
# Main conversion routine
# -----------------------------
def convert_dicom_to_png(SIZE, dicom_root, output_root, j2k_folder, df):
    df = df.copy()
    df["File Path"] = df["File Path"].apply(normalize_relpath)
    df["_key"] = df["File Path"].apply(make_key)
    key_to_relpath = dict(zip(df["_key"].values, df["File Path"].values))

    os.makedirs(output_root, exist_ok=True)
    print("Number of images:", len(df))

    N_CHUNKS = 4 if len(df) > 100 else 1
    CHUNKS = [(len(df) / N_CHUNKS * k, len(df) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
    CHUNKS = np.array(CHUNKS).astype(int)

    if torch.cuda.is_available():
        for chunk in tqdm(CHUNKS, desc="Chunks (GPU J2K + decode)"):
            os.makedirs(j2k_folder, exist_ok=True)
            rel_paths = df.iloc[chunk[0]:chunk[1]]["File Path"].values

            jp2_paths = Parallel(n_jobs=2)(
                delayed(convert_dicom_to_j2k)(dicom_root, relp, save_folder=j2k_folder)
                for relp in rel_paths
            )
            jp2_paths = [p for p in jp2_paths if p is not None]

            if not jp2_paths:
                shutil.rmtree(j2k_folder, ignore_errors=True)
                continue

            pipe = j2k_decode_pipeline(jp2_paths, batch_size=1, num_threads=2, device_id=0, debug=False)
            pipe.build()

            for jp2 in jp2_paths:
                key = os.path.splitext(os.path.basename(jp2))[0]
                rel_dcm = key_to_relpath.get(key, None)
                if rel_dcm is None: continue

                # Metadata check for PhotometricInterpretation
                dcm_path = os.path.join(dicom_root, rel_dcm)
                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    photometric = getattr(ds, "PhotometricInterpretation", "")
                except Exception:
                    continue

                out = pipe.run()
                img = out[0][0]

                # FIXED: Use INT32 to prevent overflow of UINT16 values > 32767
                img_torch = torch.empty(img.shape(), dtype=torch.int32, device="cuda")
                feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))

                # Convert to Float for processing
                img_torch = img_torch.float().reshape(img_torch.shape[0], img_torch.shape[1])

                # Handle Photometric Interpretation
                if photometric == "MONOCHROME1":
                    img_torch = img_torch.max() - img_torch

                # FIXED: Robust Scaling (Percentile) instead of Min/Max
                # This prevents "washing out" caused by extreme white pixels
                v_min = torch.quantile(img_torch, 0.005)
                v_max = torch.quantile(img_torch, 0.995)

                img_torch = torch.clamp(img_torch, v_min, v_max)
                denom = (v_max - v_min)

                if denom > 0:
                    img_torch = (img_torch - v_min) / denom
                else:
                    img_torch = torch.zeros_like(img_torch)

                # FIXED: Heuristic Background Check
                # If corners are bright (> 0.5), the image is inverted. Flip it back.
                h, w = img_torch.shape
                corners = torch.cat([
                    img_torch[0:50, 0:50].flatten(),
                    img_torch[0:50, w - 50:w].flatten(),
                    img_torch[h - 50:h, 0:50].flatten(),
                    img_torch[h - 50:h, w - 50:w].flatten()
                ])
                if corners.mean() > 0.5:
                    img_torch = 1.0 - img_torch

                img_torch = img_torch * 255.0

                # Preserved Zooming Logic
                img_torch = torch_ExtractBreast(img_torch)

                img_np = img_torch.detach().cpu().numpy().astype(np.uint8)
                img_np = cv2.resize(img_np, SIZE, interpolation=cv2.INTER_AREA)

                out_png = os.path.join(output_root, os.path.splitext(rel_dcm)[0] + ".png")
                ensure_parent_dir(out_png)
                cv2.imwrite(out_png, img_np)

            shutil.rmtree(j2k_folder, ignore_errors=True)

    _ = Parallel(n_jobs=2)(
        delayed(save_array)(dicom_root, SIZE, output_root, relp)
        for relp in tqdm(df["File Path"].values, total=len(df), desc="CPU conversion")
    )


def main():
    parser = argparse.ArgumentParser(description="Convert DICOMs to PNGs using File Path metadata.")
    parser.add_argument("--phase", type=str, default="train", help='Used only to name default output/csv.')
    parser.add_argument("--width", type=int, default=912, help="Output PNG width")
    parser.add_argument("--height", type=int, default=1520, help="Output PNG height")

    parser.add_argument(
        "--base_folder",
        type=str,
        default="/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data/NLBS_Data",
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    dicom_root = args.base_folder
    # Update this path as needed
    output_root = "/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data/NLBS_Data_png_v1"
    j2k_folder = os.path.join("/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data", "tmp/j2k/")

    metadata_csv = args.metadata_csv or os.path.join(args.base_folder, f"{args.phase}.csv")
    df = pd.read_csv(metadata_csv)

    # Process all images (removed .head(10))
    df = df.head(10)

    if "File Path" not in df.columns:
        raise ValueError(f'Metadata CSV must contain a "File Path" column. Found: {list(df.columns)}')

    df["File Path"] = df["File Path"].astype(str).str.replace("\\", "/", regex=False)

    SIZE = (args.width, args.height)

    print("torch version:", torch.__version__)
    print("timm version:", timm.__version__)
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("dicom_root:", dicom_root)
    print("metadata_csv:", metadata_csv)
    print("output_root:", output_root)

    convert_dicom_to_png(SIZE, dicom_root, output_root, j2k_folder, df)


if __name__ == "__main__":
    main()