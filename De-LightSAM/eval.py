from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    from albumentations.pytorch.transforms import ToTensor as ToTensorV2
from monai.metrics import compute_hausdorff_distance
from tqdm import tqdm

from dataloader import BinaryLoader, CSVMammoBinaryLoader
from model import ESPMedSAM


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "pipelines and experiments" / "results" / "delightsam_segmentation"


def parse_folds(value: str) -> List[int]:
    folds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            folds.extend(range(start, end + 1))
        else:
            folds.append(int(part))
    return folds


def parse_values(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_transforms(input_size: int):
    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = ESPMedSAM().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().cpu().bool().reshape(-1)
    target = target.detach().cpu().bool().reshape(-1)

    tp = torch.logical_and(pred, target).sum().item()
    tn = torch.logical_and(~pred, ~target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()

    dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 1.0
    pred_area = tp + fp
    gt_area = tp + fn
    relative_area_diff = abs(pred_area - gt_area) / gt_area if gt_area > 0 else float(pred_area > 0)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "relative_area_diff": float(relative_area_diff),
    }


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        pred = pred.detach().float().cpu().unsqueeze(0).unsqueeze(0)
        target = target.detach().float().cpu().unsqueeze(0).unsqueeze(0)
        value = compute_hausdorff_distance(pred, target)
        value = torch.mean(value).item()
        if np.isfinite(value):
            return float(value)
    except Exception:
        pass
    return float("nan")


def summarize(rows: List[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    metric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"image_id", "fold", "seconds_per_image"}
    ]
    summary = {}
    for key in metric_keys:
        values = np.array([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=np.float32)
        summary[f"{prefix}{key}"] = float(np.mean(values)) if len(values) else float("nan")
        summary[f"{prefix}{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if "seconds_per_image" in rows[0]:
        seconds = np.array([float(row["seconds_per_image"]) for row in rows], dtype=np.float32)
        summary[f"{prefix}fps"] = float(1.0 / np.mean(seconds)) if len(seconds) and np.mean(seconds) > 0 else 0.0
    return summary


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_for_fold(args: argparse.Namespace, fold: int) -> pd.DataFrame:
    csv_path = args.csv_file or Path(str(args.csv_template).format(fold=fold))
    df = pd.read_csv(csv_path)

    if args.fold_col and args.fold_col in df.columns and args.csv_file:
        df = df[df[args.fold_col].astype(str) == str(fold)]

    if args.split_col and args.split_col in df.columns:
        split_values = parse_values(args.test_split_values)
        df = df[df[args.split_col].astype(str).str.lower().isin([value.lower() for value in split_values])]

    if df.empty:
        raise ValueError(f"No rows selected for fold {fold} from {csv_path}")

    return df.reset_index(drop=True)


def build_dataset(args: argparse.Namespace, df: Optional[pd.DataFrame] = None):
    transforms = build_transforms(args.input_size)
    if df is not None:
        return CSVMammoBinaryLoader(
            df,
            transforms=transforms,
            data_root=args.data_root,
            image_col=args.image_col,
            mask_col=args.mask_col,
            id_col=args.id_col,
            domain_class=args.domain_class,
        )

    with open(args.jsonfile, "r") as f:
        split = json.load(f)
    return BinaryLoader(args.dataset, split["test"], transforms)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    fold: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    gt_dir = output_dir / "ground_truth"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in tqdm(dataset, desc=f"fold {fold}" if fold is not None else "test"):
        if len(sample) == 5:
            _, image, mask, image_id, domain_class = sample
        else:
            image, mask, image_id, domain_class = sample

        image = image.unsqueeze(0).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0).float().to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        logits, _, _ = model(x=image, domain_seq=int(domain_class))
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds_per_image = time.time() - start

        pred = (torch.sigmoid(logits) >= args.threshold).float()
        metric_row = binary_metrics(pred, mask)
        metric_row["hausdorff"] = hausdorff_distance(pred.squeeze(), mask.squeeze())
        metric_row["image_id"] = str(image_id)
        metric_row["seconds_per_image"] = float(seconds_per_image)
        if fold is not None:
            metric_row["fold"] = float(fold)
        rows.append(metric_row)

        if args.save_predictions:
            pred_np = (pred.squeeze().detach().cpu().numpy().astype(np.uint8) * 255)
            mask_np = (mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255)
            safe_id = str(image_id).replace("/", "_")
            cv2.imwrite(str(prediction_dir / f"{safe_id}.png"), pred_np)
            cv2.imwrite(str(gt_dir / f"{safe_id}.png"), mask_np)

    metrics = summarize(rows)
    save_csv(output_dir / "per_image_metrics.csv", rows)
    save_json(output_dir / "metrics.json", metrics)
    return metrics, rows


def aggregate_fold_metrics(fold_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = [
        key
        for key in fold_results[0].keys()
        if key not in {"fold"} and not key.endswith("_std")
    ]
    aggregate = {}
    for key in keys:
        values = np.array([float(row[key]) for row in fold_results if np.isfinite(float(row[key]))], dtype=np.float32)
        aggregate[key] = {
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate De-LightSAM on JSON or CSV fold metadata.")
    parser.add_argument("--dataset", default="private_mammo")
    parser.add_argument("--model", required=True, type=Path, help="Path to a De-LightSAM checkpoint.")

    parser.add_argument("--csv-template", default=None, help="CSV path template, e.g. folds/fold{fold}.csv.")
    parser.add_argument("--csv-file", type=Path, default=None, help="Single CSV containing all folds.")
    parser.add_argument("--folds", default="0-4")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--mask-col", default="mask_path")
    parser.add_argument("--id-col", default="unique_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--test-split-values", default="test")

    parser.add_argument("--jsonfile", default=None, help="Legacy JSON split file. Used only when no CSV is provided.")
    parser.add_argument("--domain-class", type=int, default=1, help="Mask token/domain class. 1 is the X-ray token.")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv_template is None and args.csv_file is None and args.jsonfile is None:
        args.jsonfile = f"{args.dataset}_data_split.json"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = args.experiment_name or args.dataset
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")
    print(f"Using domain_class={args.domain_class}")

    model = load_model(args.model, device)

    fold_results = []
    all_rows = []
    if args.csv_template or args.csv_file:
        for fold in parse_folds(args.folds):
            df = load_csv_for_fold(args, fold)
            dataset = build_dataset(args, df)
            output_dir = args.results_dir / experiment_name / f"fold{fold}"
            metrics, rows = evaluate_dataset(model, dataset, device, args, output_dir, fold)
            fold_result = {"fold": float(fold), **metrics}
            fold_results.append(fold_result)
            all_rows.extend(rows)
    else:
        dataset = build_dataset(args)
        output_dir = args.results_dir / experiment_name
        metrics, rows = evaluate_dataset(model, dataset, device, args, output_dir)
        fold_results.append(metrics)
        all_rows.extend(rows)

    save_csv(args.results_dir / experiment_name / "all_per_image_metrics.csv", all_rows)
    save_csv(args.results_dir / experiment_name / "fold_results.csv", fold_results)
    aggregate = aggregate_fold_metrics(fold_results)
    save_json(args.results_dir / experiment_name / "aggregate_metrics.json", aggregate)

    print("Aggregate metrics:")
    for metric in ["dice", "iou", "sensitivity", "precision", "hausdorff"]:
        if metric in aggregate:
            values = aggregate[metric]
            print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
