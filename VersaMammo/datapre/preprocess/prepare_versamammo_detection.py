"""
Prepare ZGT/Mammo-FM mass annotations for VersaMammo detection.

The input is the original dataframe pickle used by ZGT_file_preprocess.ipynb.
The output is a set of VersaMammo-compatible folder trees:

    VersaMammo/datapre/segdetdata/<dataset>_fold0/
      Train/<image_id>/img.jpg
      Train/<image_id>/bboxes.npy
      Eval/<image_id>/img.jpg
      Eval/<image_id>/bboxes.npy
      Test/<image_id>/img.jpg
      Test/<image_id>/bboxes.npy

Images are symlinked by default. Bounding boxes are always materialized as
`bboxes.npy` because that is the direct input expected by VersaMammo's
downstream/Detection/preprocess.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd


FM_THESIS_ROOT = Path(__file__).resolve().parents[3]


def clean_token(value: object) -> str:
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan"}:
        return "none"
    return (
        str(value)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def safe_join(root: Path, path: object) -> Path:
    path = Path(str(path))
    if path.is_absolute():
        return path
    return root / path


def read_image_size(image_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    return height, width


def mask_to_bbox(mask_path: Path) -> Optional[Tuple[int, int, int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    ys, xs = (mask > 0).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_from_row(row: pd.Series) -> Optional[Tuple[float, float, float, float]]:
    required = ["x_min", "y_min", "x_max", "y_max"]
    if not all(col in row.index for col in required):
        return None

    vals = [row[col] for col in required]
    if any(pd.isna(v) for v in vals):
        return None

    x_min, y_min, x_max, y_max = map(float, vals)
    if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def merge_masks(mask_paths: list[Path], dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return

    merged = None
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        merged = mask if merged is None else np.maximum(merged, mask)

    if merged is None:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), merged)


def assign_patient_splits(
    patients: np.ndarray,
    n_folds: int,
    test_size: float,
    random_state: int,
) -> tuple[set[str], dict[str, int]]:
    rng = np.random.default_rng(random_state)
    shuffled = np.array(sorted(patients), dtype=object)
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_size)))
    if len(shuffled) - n_test < n_folds:
        raise ValueError(
            f"Not enough trainval patients for {n_folds} folds after holding out "
            f"{n_test} test patients."
        )

    test_patients = set(map(str, shuffled[:n_test]))
    trainval_patients = shuffled[n_test:]
    fold_chunks = np.array_split(trainval_patients, n_folds)

    patient_to_fold: dict[str, int] = {}
    for fold, chunk in enumerate(fold_chunks):
        for patient in chunk:
            patient_to_fold[str(patient)] = fold

    return test_patients, patient_to_fold


def build_annotation_table(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_pickle(args.input_df)
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        image_path = safe_join(args.source_root, row[args.image_col])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mask_path = None
        if args.mask_col in row.index and not pd.isna(row[args.mask_col]):
            candidate = safe_join(args.source_root, row[args.mask_col])
            if candidate.exists():
                mask_path = candidate

        patient = clean_token(row[args.patient_col])
        view = clean_token(row[args.view_col])
        roi = clean_token(row[args.roi_col])
        image_id = f"{patient}_{view}"
        unique_id = f"{image_id}_{roi}"

        height, width = read_image_size(image_path)
        bbox = mask_to_bbox(mask_path) if mask_path is not None else bbox_from_row(row)
        class_name = args.class_name if bbox is not None else "No finding"

        rows.append(
            {
                "patient_id": patient,
                "image_id": image_id,
                "unique_id": unique_id,
                "view": view,
                "roi_num": roi,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "class_name": class_name,
                "x_min": bbox[0] if bbox is not None else -1,
                "y_min": bbox[1] if bbox is not None else -1,
                "x_max": bbox[2] if bbox is not None else -1,
                "y_max": bbox[3] if bbox is not None else -1,
                "height": height,
                "width": width,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path) if mask_path is not None else "",
            }
        )

    annotations = pd.DataFrame(rows)
    test_patients, patient_to_fold = assign_patient_splits(
        annotations["patient_id"].unique(),
        args.n_folds,
        args.test_size,
        args.random_state,
    )

    annotations["split"] = annotations["patient_id"].apply(
        lambda patient: "test" if patient in test_patients else "trainval"
    )
    annotations["fold"] = annotations["patient_id"].apply(
        lambda patient: -1 if patient in test_patients else patient_to_fold[patient]
    )

    return annotations


def write_case(
    image_rows: pd.DataFrame,
    case_dir: Path,
    link_mode: str,
    overwrite: bool,
    write_masks: bool,
) -> None:
    image_path = Path(str(image_rows.iloc[0]["image_path"]))
    place_file(image_path, case_dir / "img.jpg", link_mode, overwrite)

    valid_rows = image_rows[
        (image_rows["x_min"] >= 0)
        & (image_rows["y_min"] >= 0)
        & (image_rows["x_max"] > image_rows["x_min"])
        & (image_rows["y_max"] > image_rows["y_min"])
    ]
    bboxes = valid_rows[["x_min", "y_min", "x_max", "y_max"]].to_numpy(
        dtype=np.float32
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "bboxes.npy", bboxes)

    if not write_masks:
        return

    mask_paths = [
        Path(str(path))
        for path in valid_rows["mask_path"].tolist()
        if isinstance(path, str) and path
    ]
    if len(mask_paths) == 1:
        place_file(mask_paths[0], case_dir / "mask.png", link_mode, overwrite)
    elif len(mask_paths) > 1:
        merge_masks(mask_paths, case_dir / "mask.png", overwrite)


def materialize_fold(
    annotations: pd.DataFrame,
    args: argparse.Namespace,
    fold: int,
) -> pd.DataFrame:
    dataset = f"{args.dataset_name}_fold{fold}"
    dataset_dir = args.output_root / dataset

    fold_df = annotations.copy()
    fold_df["versamammo_dataset"] = dataset
    fold_df["versamammo_split"] = fold_df.apply(
        lambda row: "Test"
        if row["split"] == "test"
        else ("Eval" if int(row["fold"]) == fold else "Train"),
        axis=1,
    )
    fold_df["versamammo_case_dir"] = fold_df.apply(
        lambda row: str(
            Path(dataset)
            / row["versamammo_split"]
            / clean_token(row["image_id"])
        ),
        axis=1,
    )

    for (split, image_id), image_rows in fold_df.groupby(
        ["versamammo_split", "image_id"], sort=False
    ):
        case_dir = dataset_dir / split / clean_token(image_id)
        write_case(
            image_rows,
            case_dir,
            args.link_mode,
            args.overwrite,
            args.write_masks,
        )

    return fold_df


def write_manifests(all_fold_rows: list[pd.DataFrame], args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)

    base = pd.concat(all_fold_rows, ignore_index=True)
    base.to_csv(args.output_root / f"{args.dataset_name}_versamammo_rows.csv", index=False)

    split_rows = (
        base[["versamammo_dataset", "image_id", "versamammo_split"]]
        .drop_duplicates()
        .rename(
            columns={
                "versamammo_dataset": "dataset",
                "image_id": "data_name",
                "versamammo_split": "data_split",
            }
        )
    )
    split_rows.to_csv(
        args.output_root / f"{args.dataset_name}_versamammo_split.csv",
        index=False,
    )

    for fold, fold_df in enumerate(all_fold_rows):
        fold_df.to_csv(
            args.output_root / f"{args.dataset_name}_fold{fold}_rows.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create VersaMammo detection folders from df_masses.pkl."
    )
    parser.add_argument(
        "--input-df",
        type=Path,
        default=FM_THESIS_ROOT / "df_masses.pkl",
        help="Input dataframe pickle.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data/spathak"),
        help="Root used to resolve relative ImagePath/ROIPath values.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FM_THESIS_ROOT / "VersaMammo" / "datapre" / "segdetdata",
        help="Directory where <dataset>_fold*/ folders will be written.",
    )
    parser.add_argument("--dataset-name", default="ZGT_VersaMammo")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place image/mask files into VersaMammo folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-masks",
        dest="write_masks",
        action="store_false",
        help="Do not write optional mask.png files.",
    )
    parser.set_defaults(write_masks=True)

    parser.add_argument("--image-col", default="ImagePath")
    parser.add_argument("--mask-col", default="ROIPath")
    parser.add_argument("--patient-col", default="PatientID")
    parser.add_argument("--view-col", default="View")
    parser.add_argument("--roi-col", default="ROINum")
    parser.add_argument("--class-name", default="Mass")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = build_annotation_table(args)
    all_fold_rows = [
        materialize_fold(annotations, args, fold) for fold in range(args.n_folds)
    ]
    write_manifests(all_fold_rows, args)

    print(f"Prepared {args.n_folds} VersaMammo folds under: {args.output_root}")
    for fold in range(args.n_folds):
        dataset = f"{args.dataset_name}_fold{fold}"
        counts = (
            all_fold_rows[fold][["image_id", "versamammo_split"]]
            .drop_duplicates()["versamammo_split"]
            .value_counts()
            .to_dict()
        )
        print(f"{dataset}: {counts}")


if __name__ == "__main__":
    main()
