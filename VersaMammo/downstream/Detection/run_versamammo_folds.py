from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import nms

from dataloader import myDataset, myNormalize, myRandomHFlip, myRandomVFlip
from detection_metrics import evaluate_detection_records
from Faster_R_CNN_FPN import get_model
from preprocess import preprocess


CURRENT_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = CURRENT_DIR.parent
VERSAMAMMO_ROOT = DOWNSTREAM_DIR.parent
DEFAULT_DATA_ROOT = VERSAMAMMO_ROOT / "datapre" / "segdetdata"
DEFAULT_SOTAS_DIR = DOWNSTREAM_DIR / "Sotas"
MODEL_DISPLAY_NAME = "VersaMammo (Enb5)"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_detection(batch):
    return tuple(zip(*batch))


def ensure_cache(dataset: str, split: str, data_root: Path, input_size: int) -> Path:
    raw_path = data_root / dataset / split
    cache_path = data_root / dataset / f"{split}_cache_{input_size}"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing VersaMammo {split} folder: {raw_path}")
    if not cache_path.exists():
        preprocess(str(raw_path), str(cache_path), [input_size, input_size])

    return cache_path


def build_dataloader(
    dataset_path: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    train: bool,
) -> DataLoader:
    transforms = []
    if train:
        transforms.extend([myRandomVFlip(), myRandomHFlip()])
    transforms.append(myNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    dataset = myDataset(str(dataset_path), transforms)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection,
    )


def build_versamammo_detector(args: argparse.Namespace) -> nn.Module:
    checkpoint_path = args.pretrained_checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.sotas_dir / f"{MODEL_DISPLAY_NAME}.pth"

    return get_model(
        backbone_name="VersaMammo",
        checkpoint_path=str(checkpoint_path),
        pretrained=True,
        finetune=args.finetune,
        input_size=args.input_size,
    )


def move_batch_to_device(images, targets, device: torch.device):
    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    return images, targets


def filtered_prediction(output: Dict[str, torch.Tensor], args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    boxes = output["boxes"]
    scores = output["scores"]
    labels = output["labels"]

    keep = labels == 1
    keep = keep & (scores >= args.min_eval_score)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(boxes) > 0 and args.nms_threshold is not None:
        keep_idx = nms(boxes, scores, args.nms_threshold)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

    return {"boxes": boxes, "scores": scores, "labels": labels}


@torch.no_grad()
def collect_records(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, np.ndarray]]:
    model.eval()
    records = []

    for _, image_names, images, targets in dataloader:
        images, targets = move_batch_to_device(images, targets, device)
        outputs = model(images)

        for image_name, target, output in zip(image_names, targets, outputs):
            output = filtered_prediction(output, args)
            records.append(
                {
                    "image_name": str(image_name),
                    "gt_boxes": target["boxes"].detach().cpu().numpy(),
                    "pred_boxes": output["boxes"].detach().cpu().numpy(),
                    "scores": output["scores"].detach().cpu().numpy(),
                }
            )

    return records


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], List[Dict[str, np.ndarray]]]:
    records = collect_records(model, dataloader, device, args)
    metrics = evaluate_detection_records(
        records,
        pr_iou_threshold=args.pr_iou_threshold,
        score_threshold=args.score_threshold,
    )
    return metrics, records


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_records(path: Path, records: List[Dict[str, np.ndarray]]) -> None:
    rows = []
    for record in records:
        for box, score in zip(record["pred_boxes"], record["scores"]):
            rows.append(
                {
                    "image_name": record["image_name"],
                    "score": float(score),
                    "x_min": float(box[0]),
                    "y_min": float(box[1]),
                    "x_max": float(box[2]),
                    "y_max": float(box[3]),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name", "score", "x_min", "y_min", "x_max", "y_max"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_one_dataset(
    args: argparse.Namespace,
    dataset: str,
    fold: Optional[int] = None,
) -> Dict[str, float]:
    fold_dir = args.metrics_dir / dataset
    checkpoint_path = args.save_dir / dataset / f"{MODEL_DISPLAY_NAME}.pth"
    train_cache = ensure_cache(dataset, "Train", args.data_root, args.input_size)
    eval_cache = ensure_cache(dataset, "Eval", args.data_root, args.input_size)
    test_cache = ensure_cache(dataset, "Test", args.data_root, args.input_size)

    train_loader = build_dataloader(
        train_cache,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        train=True,
    )
    eval_loader = build_dataloader(
        eval_cache,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
        train=False,
    )
    test_loader = build_dataloader(
        test_cache,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
        train=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")
    model = build_versamammo_detector(args).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    best_metric = -1.0
    best_eval_metrics: Dict[str, float] = {}
    stale_validations = 0
    iteration = 0
    log_rows = []

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        for _, _, images, targets in train_loader:
            iteration += 1
            images, targets = move_batch_to_device(images, targets, device)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            loss.backward()
            optimizer.step()

            if iteration % args.log_every == 0:
                print(
                    f"{dataset} epoch={epoch + 1} iter={iteration} "
                    f"loss={loss.item():.5f}"
                )

            if iteration % args.eval_every == 0:
                eval_metrics, _ = evaluate_model(model, eval_loader, device, args)
                model.train()
                selection_metric = eval_metrics[args.selection_metric]
                log_row = {
                    "dataset": dataset,
                    "fold": float(fold) if fold is not None else "",
                    "epoch": epoch + 1,
                    "iteration": iteration,
                    "train_loss": float(loss.item()),
                    **eval_metrics,
                }
                log_rows.append(log_row)
                save_metrics_csv(fold_dir / "validation_history.csv", log_rows)

                print(
                    f"{dataset} validation iter={iteration} "
                    f"{args.selection_metric}={selection_metric:.4f} "
                    f"map_50_95={eval_metrics['map_50_95']:.4f}"
                )

                if selection_metric > best_metric:
                    stale_validations = 0
                    best_metric = selection_metric
                    best_eval_metrics = eval_metrics
                    torch.save(model.state_dict(), checkpoint_path)
                    save_json(fold_dir / "best_validation_metrics.json", eval_metrics)
                    print(f"Saved best checkpoint: {checkpoint_path}")
                else:
                    stale_validations += 1

                if stale_validations >= args.early_stop:
                    break

            if iteration >= args.max_iter:
                break

        if stale_validations >= args.early_stop or iteration >= args.max_iter:
            break

    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)
        best_eval_metrics, _ = evaluate_model(model, eval_loader, device, args)
        save_json(fold_dir / "best_validation_metrics.json", best_eval_metrics)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    test_metrics, test_records = evaluate_model(model, test_loader, device, args)
    save_json(fold_dir / "test_metrics.json", test_metrics)
    save_records(fold_dir / "test_predictions.csv", test_records)

    result = {
        "dataset": dataset,
        "fold": float(fold) if fold is not None else "",
        "training_seconds": float(time.time() - start),
        "checkpoint": str(checkpoint_path),
        **{f"val_{key}": value for key, value in best_eval_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(fold_dir / "fold_result.json", result)
    return result


def train_one_fold(args: argparse.Namespace, fold: int) -> Dict[str, float]:
    dataset = f"{args.dataset_prefix}_fold{fold}"
    return train_one_dataset(args, dataset, fold)


def evaluate_saved_dataset(
    args: argparse.Namespace,
    dataset: str,
    fold: Optional[int] = None,
) -> Dict[str, float]:
    fold_dir = args.metrics_dir / dataset
    checkpoint_path = args.save_dir / dataset / f"{MODEL_DISPLAY_NAME}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")

    test_cache = ensure_cache(dataset, "Test", args.data_root, args.input_size)
    test_loader = build_dataloader(
        test_cache,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
        train=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")
    model = build_versamammo_detector(args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    test_metrics, test_records = evaluate_model(model, test_loader, device, args)
    save_json(fold_dir / "test_metrics.json", test_metrics)
    save_records(fold_dir / "test_predictions.csv", test_records)

    result = {
        "dataset": dataset,
        "fold": float(fold) if fold is not None else "",
        "checkpoint": str(checkpoint_path),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(fold_dir / "fold_result.json", result)
    return result


def evaluate_saved_fold(args: argparse.Namespace, fold: int) -> Dict[str, float]:
    dataset = f"{args.dataset_prefix}_fold{fold}"
    return evaluate_saved_dataset(args, dataset, fold)


def aggregate_fold_results(
    results: List[Dict[str, float]],
    args: argparse.Namespace,
    aggregate_name: str,
) -> Dict[str, Dict[str, float]]:
    metric_keys = [
        key
        for key in results[0].keys()
        if key.startswith("test_") or key.startswith("val_")
    ]
    aggregate = {}
    for key in metric_keys:
        values = np.array([float(result[key]) for result in results], dtype=np.float32)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.metrics_dir / f"{aggregate_name}_aggregate_metrics.json", aggregate)

    rows = [
        {"metric": metric, "mean": values["mean"], "std": values["std"]}
        for metric, values in aggregate.items()
    ]
    save_metrics_csv(args.metrics_dir / f"{aggregate_name}_aggregate_metrics.csv", rows)
    return aggregate


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


def parse_datasets(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate VersaMammo detection on fold datasets or exact dataset folders."
    )
    parser.add_argument("--dataset-prefix", default="ZGT_VersaMammo")
    parser.add_argument("--folds", default="0-4", help="Comma/range list, e.g. 0-4 or 0,2,4.")
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated exact dataset folder names under data-root, e.g. INbreast. "
            "When set, --dataset-prefix and --folds are ignored."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Name used for aggregate metrics files. Defaults to dataset-prefix or datasets.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sotas-dir", type=Path, default=DEFAULT_SOTAS_DIR)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=CURRENT_DIR / "saved_model")
    parser.add_argument("--metrics-dir", type=Path, default=CURRENT_DIR / "metrics")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--finetune", choices=["lp", "fft"], default="lp")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=10000000)
    parser.add_argument("--batch-size-train", type=int, default=8)
    parser.add_argument("--batch-size-eval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-workers-eval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--selection-metric", default="map_50")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--pr-iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-eval-score", type=float, default=0.0)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")

    args = parser.parse_args()
    args.folds = parse_folds(args.folds)
    if args.datasets:
        args.datasets = parse_datasets(args.datasets)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.datasets:
        jobs = [(dataset, None) for dataset in args.datasets]
        aggregate_name = args.experiment_name or "_".join(args.datasets)
    else:
        jobs = [(f"{args.dataset_prefix}_fold{fold}", fold) for fold in args.folds]
        aggregate_name = args.experiment_name or args.dataset_prefix

    results = []
    for dataset, fold in jobs:
        label = f"Fold {fold}: {dataset}" if fold is not None else dataset
        print(f"=== {label} ===")
        if args.eval_only:
            results.append(evaluate_saved_dataset(args, dataset, fold))
        else:
            results.append(train_one_dataset(args, dataset, fold))

    save_metrics_csv(args.metrics_dir / f"{aggregate_name}_fold_results.csv", results)
    aggregate = aggregate_fold_results(results, args, aggregate_name)
    print("Aggregate test metrics:")
    for metric in ["test_map_50", "test_map_50_95", "test_precision_iou50_score50", "test_recall_iou50_score50"]:
        if metric in aggregate:
            values = aggregate[metric]
            print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
