#!/usr/bin/env python3
"""
Prepare a 2D mammography dataset for nnU-Net v2 from train/test pickle splits.

Expected pickle files:
    train.pkl
    test.pkl

Each pickle must contain columns:
    ImagePath   relative path to mammogram PNG
    ROIPath     relative path to binary mask PNG

The relative paths are resolved against --base-path.

The script creates REAL PNG copies for both images and masks:
- Mammograms are converted to true single-channel grayscale PNGs.
  If the source image has multiple channels, all channels must be identical.
  Otherwise the script stops with an error.
- Masks are converted to true single-channel binary PNGs with values 0 and 1.
  Any nonzero source pixel becomes foreground class 1.

Output layout:
    nnUNet_raw/
        Dataset001_Mammography/
            imagesTr/
                <case_id>_0000.png
            labelsTr/
                <case_id>.png
            imagesTs/
                <case_id>_0000.png
            labelsTs/              # optional, created unless --no-test-labels
                <case_id>.png
            dataset.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sanitize_case_id(value: str) -> str:
    """Make a string safe for use as an nnU-Net case identifier."""
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value.strip("_")


def resolve_source(base_path: Path, relative_path) -> Path:
    """Resolve a dataframe path against the provided base path."""
    if pd.isna(relative_path):
        raise ValueError("Encountered an empty/NaN path in the pickle file")

    path = Path(str(relative_path))

    if not path.is_absolute():
        path = base_path / path

    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Referenced file does not exist: {path}")

    return path


def derive_case_id(
    image_path: Path,
    base_path: Path,
    row_index,
    prefix: str,
) -> str:
    """
    Derive a stable unique case ID from the relative image path.

    Example:
        patient_001/L_CC/image.png
        -> train_patient_001_L_CC_image
    """
    try:
        rel = image_path.relative_to(base_path)
    except ValueError:
        rel = image_path

    without_suffix = rel.with_suffix("")
    parts = [sanitize_case_id(p) for p in without_suffix.parts]
    parts = [p for p in parts if p]

    stem = "_".join(parts)
    if not stem:
        stem = f"case_{row_index}"

    return f"{prefix}_{stem}"


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    required = {"ImagePath", "ROIPath"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def prepare_destination(destination: Path, overwrite: bool) -> None:
    """Ensure destination can be written."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.unlink()


def rewrite_mammogram_single_channel(
    source: Path,
    destination: Path,
    overwrite: bool = False,
) -> None:
    """
    Write a mammogram as a true single-channel PNG.

    Accepted inputs:
    - 2D grayscale image: written directly
    - multi-channel image: all channels must be identical

    This preserves the original grayscale pixel values exactly when RGB
    channels are identical.
    """
    prepare_destination(destination, overwrite)

    with Image.open(source) as img:
        array = np.asarray(img)

    if array.ndim == 2:
        gray = array

    elif array.ndim == 3:
        # Require every channel to be identical to the first one.
        first = array[..., 0]

        for channel_idx in range(1, array.shape[-1]):
            if not np.array_equal(first, array[..., channel_idx]):
                raise ValueError(
                    f"Image has non-identical channels and cannot be reduced "
                    f"losslessly to one channel: {source}"
                )

        gray = first

    else:
        raise ValueError(
            f"Unsupported image shape {array.shape} for mammogram: {source}"
        )

    # PIL supports uint8 and uint16 grayscale PNGs directly.
    if gray.dtype == np.bool_:
        gray = gray.astype(np.uint8)

    if gray.dtype not in (np.uint8, np.uint16):
        if np.issubdtype(gray.dtype, np.integer):
            min_val = int(gray.min())
            max_val = int(gray.max())

            if min_val < 0:
                raise ValueError(
                    f"Negative grayscale values found in mammogram: {source}"
                )

            if max_val <= 255:
                gray = gray.astype(np.uint8)
            elif max_val <= 65535:
                gray = gray.astype(np.uint16)
            else:
                raise ValueError(
                    f"Grayscale values exceed uint16 range in: {source}"
                )
        else:
            raise ValueError(
                f"Unsupported mammogram dtype {gray.dtype} in: {source}"
            )

    Image.fromarray(gray).save(destination)

    # Defensive check: output must be a true 2D image.
    with Image.open(destination) as written_img:
        written = np.asarray(written_img)

    if written.ndim != 2:
        raise RuntimeError(
            f"Rewritten mammogram is not single-channel: "
            f"{destination}, shape={written.shape}"
        )


def rewrite_binary_mask(
    source: Path,
    destination: Path,
    overwrite: bool = False,
) -> None:
    """
    Rewrite a binary segmentation mask as a single-channel PNG with values 0/1.

    Any nonzero source pixel becomes class 1.
    For multi-channel masks, channels must either be identical or the mask is
    collapsed by foreground presence across channels.
    """
    prepare_destination(destination, overwrite)

    with Image.open(source) as img:
        mask = np.asarray(img)

    if mask.ndim == 2:
        binary = mask != 0

    elif mask.ndim == 3:
        # If channels are identical, preserve that interpretation directly.
        first = mask[..., 0]
        identical = all(
            np.array_equal(first, mask[..., i])
            for i in range(1, mask.shape[-1])
        )

        if identical:
            binary = first != 0
        else:
            # For RGB/RGBA binary masks, any nonzero channel means foreground.
            binary = np.any(mask != 0, axis=-1)

    else:
        raise ValueError(
            f"Unsupported mask shape {mask.shape} for: {source}"
        )

    binary = binary.astype(np.uint8)

    Image.fromarray(binary, mode="L").save(destination)

    # Defensive checks for nnU-Net.
    with Image.open(destination) as written_img:
        written = np.asarray(written_img)

    if written.ndim != 2:
        raise RuntimeError(
            f"Rewritten mask is not single-channel: "
            f"{destination}, shape={written.shape}"
        )

    unique_values = np.unique(written)

    if not np.all(np.isin(unique_values, [0, 1])):
        raise RuntimeError(
            f"Rewritten mask contains unexpected values "
            f"{unique_values}: {destination}"
        )


def process_split(
    df: pd.DataFrame,
    split_name: str,
    base_path: Path,
    images_dir: Path,
    labels_dir: Path | None,
    overwrite: bool,
    used_case_ids: set,
) -> int:
    count = 0

    for row_index, row in df.iterrows():
        image_source = resolve_source(base_path, row["ImagePath"])
        mask_source = resolve_source(base_path, row["ROIPath"])

        if image_source.suffix.lower() != ".png":
            raise ValueError(f"Image is not a PNG: {image_source}")

        if mask_source.suffix.lower() != ".png":
            raise ValueError(f"Mask is not a PNG: {mask_source}")

        case_id = derive_case_id(
            image_path=image_source,
            base_path=base_path,
            row_index=row_index,
            prefix=split_name,
        )

        original_case_id = case_id
        suffix = 1

        while case_id in used_case_ids:
            suffix += 1
            case_id = f"{original_case_id}_{suffix}"

        used_case_ids.add(case_id)

        image_destination = images_dir / f"{case_id}_0000.png"

        rewrite_mammogram_single_channel(
            source=image_source,
            destination=image_destination,
            overwrite=overwrite,
        )

        label_destination = None

        if labels_dir is not None:
            label_destination = labels_dir / f"{case_id}.png"

            try:
                rewrite_binary_mask(
                    source=mask_source,
                    destination=label_destination,
                    overwrite=overwrite,
                )
            except Exception:
                if image_destination.exists():
                    image_destination.unlink()
                raise

        count += 1

        print(f"[{split_name} {count:05d}] {case_id}")
        print(f"  image: {image_source}")
        print(f"      -> single-channel PNG: {image_destination}")

        if label_destination is not None:
            print(f"  mask:  {mask_source}")
            print(f"      -> single-channel 0/1 PNG: {label_destination}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create an nnU-Net v2 dataset from train.pkl/test.pkl. "
            "Images are rewritten as single-channel PNGs and masks as "
            "single-channel 0/1 PNGs."
        )
    )

    parser.add_argument(
        "--base-path",
        required=True,
        type=Path,
        help="Base path against which ImagePath and ROIPath are resolved.",
    )

    parser.add_argument(
        "--train-pkl",
        required=True,
        type=Path,
        help="Path to train.pkl.",
    )

    parser.add_argument(
        "--test-pkl",
        required=True,
        type=Path,
        help="Path to test.pkl.",
    )

    parser.add_argument(
        "--nnunet-raw",
        required=True,
        type=Path,
        help="Parent nnUNet_raw directory.",
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        help="nnU-Net dataset ID. Default: 1",
    )

    parser.add_argument(
        "--dataset-name",
        default="Mammography",
        help="nnU-Net dataset name. Default: Mammography",
    )

    parser.add_argument(
        "--no-test-labels",
        action="store_true",
        help="Do not create labelsTs masks for the test split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing rewritten images, masks, and dataset.json."
        ),
    )

    args = parser.parse_args()

    base_path = args.base_path.expanduser().resolve()
    train_pkl = args.train_pkl.expanduser().resolve()
    test_pkl = args.test_pkl.expanduser().resolve()
    nnunet_raw = args.nnunet_raw.expanduser().resolve()

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path does not exist: {base_path}")

    if not train_pkl.is_file():
        raise FileNotFoundError(f"train.pkl not found: {train_pkl}")

    if not test_pkl.is_file():
        raise FileNotFoundError(f"test.pkl not found: {test_pkl}")

    if args.dataset_id < 1:
        raise ValueError("--dataset-id must be a positive integer")

    train_df = pd.read_pickle(train_pkl)
    test_df = pd.read_pickle(test_pkl)

    validate_dataframe(train_df, "train.pkl")
    validate_dataframe(test_df, "test.pkl")

    dataset_name = sanitize_case_id(args.dataset_name)
    dataset_dir = (
        nnunet_raw
        / f"Dataset{args.dataset_id:03d}_{dataset_name}"
    )

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = (
        None
        if args.no_test_labels
        else dataset_dir / "labelsTs"
    )

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    if labels_ts is not None:
        labels_ts.mkdir(parents=True, exist_ok=True)

    used_case_ids = set()

    print("Processing training split...")

    n_train = process_split(
        df=train_df,
        split_name="train",
        base_path=base_path,
        images_dir=images_tr,
        labels_dir=labels_tr,
        overwrite=args.overwrite,
        used_case_ids=used_case_ids,
    )

    print("\nProcessing test split...")

    n_test = process_split(
        df=test_df,
        split_name="test",
        base_path=base_path,
        images_dir=images_ts,
        labels_dir=labels_ts,
        overwrite=args.overwrite,
        used_case_ids=used_case_ids,
    )

    dataset_json = {
        "channel_names": {
            "0": "grayscale"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": n_train,
        "file_ending": ".png"
    }

    dataset_json_path = dataset_dir / "dataset.json"

    if dataset_json_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{dataset_json_path} already exists. "
            f"Use --overwrite to replace it."
        )

    with dataset_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)
        f.write("\n")

    print("\nDone.")
    print(f"Dataset:          {dataset_dir}")
    print(f"Training cases:   {n_train}")
    print(f"Test cases:       {n_test}")
    print(f"Training images:  {images_tr} (single-channel PNG copies)")
    print(f"Training labels:  {labels_tr} (single-channel 0/1 PNG copies)")
    print(f"Test images:      {images_ts} (single-channel PNG copies)")

    if labels_ts is not None:
        print(
            f"Test labels:      {labels_ts} "
            f"(single-channel 0/1 PNG copies)"
        )
    else:
        print("Test labels:      not created")

    print(f"dataset.json:     {dataset_json_path}")


if __name__ == "__main__":
    main()
