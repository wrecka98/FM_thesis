#!/usr/bin/env python3
from __future__ import annotations

import argparse
import plistlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pydicom
from skimage.draw import polygon


IMAGE_SUFFIXES = {".dcm", ".dicom"}


def load_dicom_to_uint8(
    path: Path,
    invert_monochrome1: bool = True,
    percentile_clip: Optional[tuple[float, float]] = (1.0, 99.0),
) -> np.ndarray:
    """
    Convert a mammography DICOM to a 2D uint8 image in [0, 255].

    This satisfies Swin-LiteMedSAM's 2D input expectations:
      - image saved in NPZ as HxWx3
      - values in [0, 255] (repo checks max < 256)

    The repository does not prescribe a DICOM windowing policy, so this function uses:
      1) pixel_array
      2) RescaleSlope / RescaleIntercept if present
      3) MONOCHROME1 inversion if requested
      4) optional percentile clipping
      5) min-max scaling to uint8
    """
    dcm = pydicom.dcmread(str(path), stop_before_pixels=False)
    img = dcm.pixel_array.astype(np.float32)

    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    if invert_monochrome1 and getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    if percentile_clip is not None:
        lo_pct, hi_pct = percentile_clip
        lo, hi = np.percentile(img, [lo_pct, hi_pct])
        img = np.clip(img, lo, hi)
    else:
        lo, hi = float(img.min()), float(img.max())

    denom = float(hi - lo)
    if denom <= 0:
        img = np.zeros_like(img, dtype=np.uint8)
    else:
        img = ((img - lo) / (denom + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)

    return img



def load_inbreast_mass_mask_and_boxes(
    xml_path: Path,
    image_shape: tuple[int, int],
    roi_class_name: Optional[str] = "Mass",
    bbox_padding: int = 0,
):
    def load_point(point_string: str) -> tuple[float, float]:
        x, y = [float(num) for num in point_string.strip("()").split(",")]
        return y, x  # row, col

    h, w = image_shape
    union_mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []

    with open(xml_path, "rb") as f:
        plist_dict = plistlib.load(f, fmt=plistlib.FMT_XML)["Images"][0]

    rois = plist_dict.get("ROIs", [])
    for roi in rois:
        if roi_class_name is not None:
            name = roi.get("Name", "")
            if name != roi_class_name:
                continue

        points = roi.get("Point_px", [])
        if not points:
            continue

        coords = [load_point(p) for p in points]
        rows = np.array([p[0] for p in coords], dtype=np.float32)
        cols = np.array([p[1] for p in coords], dtype=np.float32)

        # fill union mask
        if len(coords) <= 2:
            for r, c in coords:
                rr = int(round(r))
                cc = int(round(c))
                if 0 <= rr < h and 0 <= cc < w:
                    union_mask[rr, cc] = 1
        else:
            rr, cc = polygon(rows, cols, shape=(h, w))
            union_mask[rr, cc] = 1

        # compute one box for this ROI directly from its own points
        y_min = max(0, int(np.floor(rows.min())) - bbox_padding)
        y_max = min(h - 1, int(np.ceil(rows.max())) + bbox_padding)
        x_min = max(0, int(np.floor(cols.min())) - bbox_padding)
        x_max = min(w - 1, int(np.ceil(cols.max())) + bbox_padding)

        boxes.append([x_min, y_min, x_max, y_max])

    if boxes:
        boxes = np.asarray(boxes, dtype=np.int32)
    else:
        boxes = np.zeros((0, 4), dtype=np.int32)

    return union_mask, boxes


def find_annotation_for_image(
    image_path: Path,
    annotation_dir: Optional[Path],
    prefix_match: bool = True,
) -> Optional[Path]:
    search_root = annotation_dir if annotation_dir is not None else image_path.parent
    prefix = image_path.name.split("_")[0] if prefix_match else image_path.stem
    xmls = sorted(search_root.glob(f"{prefix}*.xml")) if prefix_match else [search_root / f"{image_path.stem}.xml"]
    xmls = [p for p in xmls if p.exists()]
    return xmls[0] if xmls else None


def iter_dicom_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def make_three_channel(img2d_uint8: np.ndarray) -> np.ndarray:
    return np.repeat(img2d_uint8[..., None], 3, axis=-1)


def convert_case(
    dcm_path: Path,
    output_dir: Path,
    annotation_dir: Optional[Path],
    include_masks: bool,
    skip_empty: bool,
    bbox_padding: int,
    invert_monochrome1: bool,
    percentile_clip: Optional[tuple[float, float]],
    prefix_match_annotations: bool,
) -> str:
    img2d = load_dicom_to_uint8(
        dcm_path,
        invert_monochrome1=invert_monochrome1,
        percentile_clip=percentile_clip,
    )
    h, w = img2d.shape
    img3c = make_three_channel(img2d)

    xml_path = find_annotation_for_image(
        dcm_path,
        annotation_dir=annotation_dir,
        prefix_match=prefix_match_annotations,
    )

    if xml_path is None:
        if skip_empty:
            return f"SKIP  {dcm_path.name}: no matching XML"
        mass_mask = np.zeros((h, w), dtype=np.uint8)
        boxes=[None]
    else:
        mass_mask,boxes = load_inbreast_mass_mask_and_boxes(xml_path, image_shape=(h, w), roi_class_name="Mass")

    if bbox_padding > 0 and boxes.size > 0:
        boxes = boxes.copy()
        boxes[:, 0] = np.maximum(0, boxes[:, 0] - bbox_padding)
        boxes[:, 1] = np.maximum(0, boxes[:, 1] - bbox_padding)
        boxes[:, 2] = np.minimum(w - 1, boxes[:, 2] + bbox_padding)
        boxes[:, 3] = np.minimum(h - 1, boxes[:, 3] + bbox_padding)

    if boxes.shape[0] == 0 and skip_empty:
        return f"SKIP  {dcm_path.name}: no Mass ROI"

    out_path = output_dir / f"{dcm_path.stem}.npz"

    if include_masks:
        # Extra arrays are ignored by the authors' infer.py; it only reads imgs and boxes.
        np.savez_compressed(
            out_path,
            imgs=img3c,
            boxes=boxes.astype(np.int32),
            gts=mass_mask.astype(np.uint8),
            image_path=str(dcm_path),
            annotation_path=str(xml_path) if xml_path is not None else "",
        )
    else:
        np.savez_compressed(
            out_path,
            imgs=img3c,
            boxes=boxes.astype(np.int32),
        )

    return f"OK    {dcm_path.name} -> {out_path.name} | boxes={len(boxes)}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert INbreast DICOM + XML mass annotations to Swin-LiteMedSAM 2D NPZ input files."
    )
    p.add_argument("--dicom_dir", type=Path, required=True, help="Directory containing INbreast DICOM files (e.g. MassDICOMs).")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory where NPZ files will be written.")
    p.add_argument("--annotation_dir", type=Path, default=None, help="Directory containing XML files. Defaults to the DICOM file's parent folder.")
    p.add_argument("--include_masks", action="store_true", help="Also save the binary mass mask as key 'gts'. Docker inference will ignore it.")
    p.add_argument("--skip_empty", action="store_true", help="Skip cases without a matching XML or without a Mass ROI.")
    p.add_argument("--bbox_padding", type=int, default=0, help="Optional padding added to each box in original pixel coordinates.")
    p.add_argument("--no_invert_monochrome1", action="store_true", help="Disable MONOCHROME1 inversion.")
    p.add_argument("--no_percentile_clip", action="store_true", help="Disable percentile clipping before uint8 scaling.")
    p.add_argument("--clip_low", type=float, default=1.0, help="Low percentile for clipping before uint8 scaling.")
    p.add_argument("--clip_high", type=float, default=99.0, help="High percentile for clipping before uint8 scaling.")
    p.add_argument("--no_prefix_match_annotations", action="store_true", help="Match XML by exact stem instead of INbreast prefix.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    percentile_clip = None if args.no_percentile_clip else (args.clip_low, args.clip_high)

    dicom_files = list(iter_dicom_files(args.dicom_dir))
    if not dicom_files:
        raise SystemExit(f"No DICOM files found under {args.dicom_dir}")

    for dcm_path in dicom_files:
        msg = convert_case(
            dcm_path=dcm_path,
            output_dir=args.output_dir,
            annotation_dir=args.annotation_dir,
            include_masks=args.include_masks,
            skip_empty=args.skip_empty,
            bbox_padding=args.bbox_padding,
            invert_monochrome1=not args.no_invert_monochrome1,
            percentile_clip=percentile_clip,
            prefix_match_annotations=not args.no_prefix_match_annotations,
        )
        print(msg)

    print(f"Done. NPZ files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
