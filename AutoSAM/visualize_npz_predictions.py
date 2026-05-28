import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def normalize_image(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resolve_path(data_root, value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(data_root) / path


def squeeze_array(array):
    array = np.asarray(array)
    return np.squeeze(array)


def choose_slice_index(pred, gt, mode="middle", index=None):
    pred = squeeze_array(pred)
    gt = squeeze_array(gt)
    reference = pred if mode == "max-pred" else gt

    if reference.ndim <= 2:
        return None
    if index is not None:
        return max(0, min(int(index), reference.shape[0] - 1))
    if mode in ("max-pred", "max-gt"):
        areas = reference.reshape(reference.shape[0], -1).sum(axis=1)
        return int(np.argmax(areas))
    return reference.shape[0] // 2


def volume_to_slice(array, index=None):
    array = squeeze_array(array)
    if array.ndim <= 2 or index is None:
        return array

    # AutoSAM debug volumes are usually saved as D x H x W.
    index = max(0, min(int(index), array.shape[0] - 1))
    return array[index]


def to_binary(mask, threshold):
    mask = np.asarray(mask, dtype=np.float32)
    return mask > float(threshold)


def resize_mask(mask, shape):
    if mask.shape == shape[:2]:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def overlay_masks(image, pred_mask=None, gt_mask=None, alpha=0.38):
    canvas = image.copy()
    color_layer = np.zeros_like(canvas)

    if gt_mask is not None:
        color_layer[gt_mask] = (0, 220, 0)
    if pred_mask is not None:
        color_layer[pred_mask] = (0, 0, 255)
    if pred_mask is not None and gt_mask is not None:
        color_layer[pred_mask & gt_mask] = (0, 220, 255)

    colored = color_layer.any(axis=2)
    canvas[colored] = cv2.addWeighted(canvas, 1 - alpha, color_layer, alpha, 0)[colored]

    if gt_mask is not None:
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    if pred_mask is not None:
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
    return canvas


def add_label(image, text):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (max(220, 12 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(labeled, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def load_test_rows(args):
    if not args.csv_file:
        return None
    import pandas as pd

    df = pd.read_csv(args.csv_file)
    split = df[args.split_col].astype(str).str.lower()
    test_values = {value.strip().lower() for value in args.test_split_values.split(",") if value.strip()}
    mask = split.isin(test_values)

    if args.fold_col in df.columns:
        fold_values = df[args.fold_col].astype(str)
        fold_test = mask & (fold_values == str(args.fold))
        heldout_test = mask & fold_values.isin(["-1", "heldout", "held-out"])
        if fold_test.any():
            mask = fold_test
        elif heldout_test.any():
            mask = heldout_test

    rows = df[mask].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No test rows found in {args.csv_file}")
    return rows


def parse_debug_index(path):
    match = re.search(r"debug_volume(\d+)\.npz$", path.name)
    return int(match.group(1)) if match else None


def make_panel(image, pred_mask, gt_mask):
    gt_panel = add_label(overlay_masks(image, gt_mask=gt_mask), "Ground truth")
    pred_panel = add_label(overlay_masks(image, pred_mask=pred_mask), "Prediction")
    both_panel = add_label(overlay_masks(image, pred_mask=pred_mask, gt_mask=gt_mask), "GT + prediction")
    return cv2.hconcat([gt_panel, pred_panel, both_panel])


def visualize_file(npz_path, output_dir, args, test_rows=None):
    data = np.load(npz_path)
    slice_index = choose_slice_index(
        data["mask"],
        data["gt"],
        mode=args.slice_mode,
        index=args.slice_index,
    )
    pred = volume_to_slice(data["mask"], index=slice_index)
    gt = volume_to_slice(data["gt"], index=slice_index)

    debug_index = parse_debug_index(npz_path)
    if test_rows is not None and debug_index is not None and debug_index < len(test_rows):
        row = test_rows.iloc[debug_index]
        image = normalize_image(read_gray(resolve_path(args.data_root, row[args.image_col])))
        gt = read_gray(resolve_path(args.data_root, row[args.mask_col])) > 0
    else:
        image = normalize_image(volume_to_slice(data["image"], index=slice_index))
        gt = to_binary(gt, args.gt_threshold)

    pred = to_binary(pred, args.pred_threshold)
    pred = resize_mask(pred, image.shape)
    gt = resize_mask(gt, image.shape)

    panel = make_panel(image, pred, gt)
    output_path = output_dir / f"{npz_path.stem}_overlay.png"
    cv2.imwrite(str(output_path), panel)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize AutoSAM .npz prediction debug volumes.")
    parser.add_argument("--npz-dir", required=True, help="Directory containing debug_volume*.npz files.")
    parser.add_argument("--output-dir", required=True, help="Directory where overlay PNGs will be saved.")
    parser.add_argument("--csv-file", default="", help="Optional CSV used to reload original images/masks.")
    parser.add_argument("--data-root", default=".", help="Root for relative CSV image/mask paths.")
    parser.add_argument("--image-col", default="image_path", help="CSV image path column.")
    parser.add_argument("--mask-col", default="mask_path", help="CSV mask path column.")
    parser.add_argument("--split-col", default="split", help="CSV split column.")
    parser.add_argument("--fold-col", default="fold", help="CSV fold column.")
    parser.add_argument("--fold", default="0", help="Fold used for mapping test rows.")
    parser.add_argument("--test-split-values", default="test", help="Comma-separated test split values.")
    parser.add_argument("--pred-threshold", type=float, default=0.5, help="Prediction binarization threshold.")
    parser.add_argument("--gt-threshold", type=float, default=0.5, help="GT binarization threshold for NPZ GT.")
    parser.add_argument("--slice-mode", choices=["middle", "max-pred", "max-gt"], default="max-pred")
    parser.add_argument("--slice-index", type=int, default=None, help="Optional explicit slice index.")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_rows = load_test_rows(args) if args.csv_file else None
    npz_files = sorted(npz_dir.glob("debug_volume*.npz"), key=lambda path: parse_debug_index(path) or -1)
    if not npz_files:
        raise FileNotFoundError(f"No debug_volume*.npz files found in {npz_dir}")

    for npz_path in npz_files:
        output_path = visualize_file(npz_path, output_dir, args, test_rows=test_rows)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
