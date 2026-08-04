from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader

from UNetEfficientNetB5 import UNetEfficientNetB5
from run_versamammo_segmentation_dataframe import (
    DEFAULT_PATH_ROOT,
    PngDataframeDataset,
    extract_state_dict,
    validate_dataframe,
)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = UNetEfficientNetB5(checkpoint_path=None, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not load {checkpoint_path} as a complete VersaMammo segmentation "
            "checkpoint. Supply the final 'VersaMammo (Enb5).pth' written by "
            "run_versamammo_segmentation_dataframe.py, not a backbone-only checkpoint."
        ) from error
    return model.to(device).eval()


def calculate_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = prediction.bool().reshape(-1)
    target = target.bool().reshape(-1)
    tp = torch.logical_and(prediction, target).sum().item()
    tn = torch.logical_and(~prediction, ~target).sum().item()
    fp = torch.logical_and(prediction, ~target).sum().item()
    fn = torch.logical_and(~prediction, target).sum().item()
    pred_area = tp + fp
    target_area = tp + fn
    return {
        "dice": 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "sensitivity": tp / (tp + fn) if tp + fn else 1.0,
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "relative_area_diff": abs(pred_area - target_area) / target_area
        if target_area
        else float(pred_area > 0),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for batch in loader:
        images = batch["images"].float().to(device, non_blocking=True)
        targets = batch["masks"] >= 0.5
        predictions = model(images).cpu() >= threshold

        for row_number, source_index, image_path, mask_path, prediction, target in zip(
            batch["row_number"],
            batch["source_index"],
            batch["image_path"],
            batch["mask_path"],
            predictions,
            targets,
        ):
            row_number = int(row_number)
            row: Dict[str, object] = {
                "row_number": row_number,
                "source_index": source_index,
                "image_path": image_path,
                "mask_path": mask_path,
            }
            row.update(calculate_metrics(prediction, target))
            rows.append(row)

            mask = prediction.squeeze(0).numpy().astype(np.uint8) * 255
            io.imsave(
                prediction_dir / f"row_{row_number}.png",
                mask,
                check_contrast=False,
            )

    metric_names = [
        key
        for key in rows[0]
        if key not in {"row_number", "source_index", "image_path", "mask_path"}
    ]
    summary = {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in rows])),
            "std": float(np.std([float(row[metric]) for row in rows], ddof=1))
            if len(rows) > 1
            else 0.0,
        }
        for metric in metric_names
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VersaMammo segmentation model on PNG paths in a pickle."
    )
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument(
        "--image-scale",
        choices=("auto", "255", "unit", "minmax"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataframe.is_file():
        raise FileNotFoundError(f"Dataframe not found: {args.dataframe}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1 or args.input_size < 1:
        raise ValueError("--batch-size and --input-size must be positive.")

    dataframe = pd.read_pickle(args.dataframe)
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{args.dataframe} does not contain a pandas DataFrame.")
    validate_dataframe(dataframe, args.path_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PngDataframeDataset(
        dataframe,
        args.path_root,
        args.input_size,
        args.image_scale,
        args.mask_threshold,
        augment=False,
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Evaluating all {len(dataset)} dataframe rows using {device}.")
    model = load_model(args.checkpoint, device)
    rows, summary = evaluate(
        model,
        loader,
        device,
        args.prediction_threshold,
        args.output_dir,
    )
    write_csv(args.output_dir / "per_image_metrics.csv", rows)
    with (args.output_dir / "aggregate_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "dataframe": str(args.dataframe.resolve()),
                "path_root": str(args.path_root.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "samples": len(dataset),
                "input_size": args.input_size,
                "image_scale": args.image_scale,
                "mask_threshold": args.mask_threshold,
                "prediction_threshold": args.prediction_threshold,
                "device_used": str(device),
            },
            handle,
            indent=2,
        )

    print("Aggregate metrics:")
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
