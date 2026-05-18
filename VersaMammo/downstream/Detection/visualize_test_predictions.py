from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = CURRENT_DIR.parent
VERSAMAMMO_ROOT = DOWNSTREAM_DIR.parent
DEFAULT_DATA_ROOT = VERSAMAMMO_ROOT / "datapre" / "segdetdata"
DEFAULT_METRICS_DIR = CURRENT_DIR / "metrics"


def load_json(path: Path) -> Dict[str, float]:
    with open(path) as f:
        return json.load(f)


def find_best_fold(metrics_dir: Path, dataset_prefix: str, metric: str) -> str:
    candidates = []
    for metrics_path in metrics_dir.glob(f"{dataset_prefix}_fold*/test_metrics.json"):
        metrics = load_json(metrics_path)
        if metric not in metrics:
            continue
        candidates.append((float(metrics[metric]), metrics_path.parent.name))

    if not candidates:
        raise FileNotFoundError(
            f"No fold metrics with key '{metric}' found under {metrics_dir}"
        )

    candidates.sort(reverse=True)
    return candidates[0][1]


def load_predictions(prediction_csv: Path, score_threshold: float) -> Dict[str, List[dict]]:
    predictions = defaultdict(list)
    with open(prediction_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["score"])
            if score < score_threshold:
                continue
            predictions[row["image_name"]].append(
                {
                    "score": score,
                    "box": np.array(
                        [
                            float(row["x_min"]),
                            float(row["y_min"]),
                            float(row["x_max"]),
                            float(row["y_max"]),
                        ],
                        dtype=np.float32,
                    ),
                }
            )

    return predictions


def read_image(path: Path) -> np.ndarray:
    image = plt.imread(str(path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def rescale_prediction_box(box: np.ndarray, image_shape, input_size: int) -> np.ndarray:
    height, width = image_shape[:2]
    scale = np.array(
        [width / input_size, height / input_size, width / input_size, height / input_size],
        dtype=np.float32,
    )
    return box * scale


def draw_box(ax, box: np.ndarray, color: str, linewidth: float, label: Optional[str] = None):
    x_min, y_min, x_max, y_max = box.tolist()
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x_min,
            max(0, y_min - 4),
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 1.5, "edgecolor": "none"},
        )


def save_overlay(
    image_name: str,
    case_dir: Path,
    predictions: List[dict],
    output_dir: Path,
    input_size: int,
    dpi: int,
) -> None:
    image_path = case_dir / "img.jpg"
    gt_path = case_dir / "bboxes.npy"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT boxes: {gt_path}")

    image = read_image(image_path)
    gt_boxes = np.load(gt_path, allow_pickle=True).astype(np.float32).reshape(-1, 4)

    fig, ax = plt.subplots(figsize=(10, 10))
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    for gt_box in gt_boxes:
        draw_box(ax, gt_box, color="lime", linewidth=2.0, label="GT")

    for pred in predictions:
        pred_box = rescale_prediction_box(pred["box"], image.shape, input_size)
        draw_box(
            ax,
            pred_box,
            color="red",
            linewidth=1.5,
            label=f"{pred['score']:.2f}",
        )

    ax.set_axis_off()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{image_name}.png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay GT and saved predicted boxes for the best VersaMammo fold."
    )
    parser.add_argument("--dataset-prefix", default="ZGT_VersaMammo")
    parser.add_argument("--fold", default=None, help="Fold directory name or number. Defaults to best by metric.")
    parser.add_argument("--best-metric", default="map_50")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def normalize_fold_name(dataset_prefix: str, fold: Optional[str], metrics_dir: Path, metric: str) -> str:
    if fold is None:
        return find_best_fold(metrics_dir, dataset_prefix, metric)
    if fold.isdigit():
        return f"{dataset_prefix}_fold{fold}"
    return fold


def main() -> None:
    args = parse_args()
    fold_name = normalize_fold_name(
        args.dataset_prefix,
        args.fold,
        args.metrics_dir,
        args.best_metric,
    )

    metrics_path = args.metrics_dir / fold_name / "test_metrics.json"
    prediction_csv = args.metrics_dir / fold_name / "test_predictions.csv"
    dataset_dir = args.data_root / fold_name / "Test"
    output_dir = args.output_dir or args.metrics_dir / fold_name / "test_overlays"

    metrics = load_json(metrics_path)
    predictions = load_predictions(prediction_csv, args.score_threshold)
    image_names = sorted(predictions.keys())
    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    print(
        f"Visualizing {fold_name} ({args.best_metric}={metrics.get(args.best_metric, float('nan')):.4f})"
    )
    print(f"Reading predictions from: {prediction_csv}")
    print(f"Reading raw test cases from: {dataset_dir}")
    print(f"Writing overlays to: {output_dir}")

    for idx, image_name in enumerate(image_names, start=1):
        save_overlay(
            image_name=image_name,
            case_dir=dataset_dir / image_name,
            predictions=predictions[image_name],
            output_dir=output_dir,
            input_size=args.input_size,
            dpi=args.dpi,
        )
        if idx % 10 == 0 or idx == len(image_names):
            print(f"Saved {idx}/{len(image_names)} overlays")


if __name__ == "__main__":
    main()
