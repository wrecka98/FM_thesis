from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from skimage import io


DEFAULT_TRAINING_MODULES = (
    "versamammo_train_seg_v2",
    "versamammo_train_seg",
)


def import_training_module(module_name: str | None):
    """Import the training module that owns the cache/model definitions."""
    candidates = (module_name,) if module_name else DEFAULT_TRAINING_MODULES
    errors = []

    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
        except ImportError as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        required = (
            "ensure_cache",
            "build_dataloader",
            "UNetEfficientNetB5",
            "batch_metrics",
            "summarize_metric_rows",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            errors.append(f"{candidate}: missing {missing}")
            continue

        print(f"Using training utilities from: {module.__file__}")
        return module

    details = "\n".join(errors)
    raise ImportError(
        "Could not import a compatible VersaMammo training module. Place this "
        "script in the same directory as versamammo_train_seg_v2.py or "
        "versamammo_train_seg.py, or pass --training-module.\n" + details
    )


def safe_filename(value: object) -> str:
    name = str(value)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "case"


def load_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load checkpoints saved either as a raw state dict or a wrapper dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} does not contain a model state dictionary."
        )

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def image_to_display(image: torch.Tensor) -> np.ndarray:
    """Convert normalized CxHxW input tensor to an RGB float image in [0, 1]."""
    image = image.detach().cpu().float()

    if image.ndim != 3:
        raise ValueError(f"Expected CxHxW image tensor, got {tuple(image.shape)}")

    # myNormalize(mean=.5, std=.5) transforms [0,1] to approximately [-1,1].
    image = image * 0.5 + 0.5
    image = image.clamp(0.0, 1.0)

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 2:
        image = torch.cat([image, image[:1]], dim=0)
    elif image.shape[0] > 3:
        image = image[:3]

    return image.permute(1, 2, 0).numpy()


def make_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Build an RGB overlay.

    Ground truth only: green
    Prediction only: red
    Overlap: yellow
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("overlay alpha must be between 0 and 1")

    image = np.asarray(image, dtype=np.float32)
    gt = np.asarray(ground_truth, dtype=bool).squeeze()
    pred = np.asarray(prediction, dtype=bool).squeeze()

    if image.shape[:2] != gt.shape or gt.shape != pred.shape:
        raise ValueError(
            f"Overlay shape mismatch: image={image.shape}, gt={gt.shape}, pred={pred.shape}"
        )

    gt_only = gt & ~pred
    pred_only = pred & ~gt
    overlap = gt & pred

    overlay = image.copy()
    color_layer = image.copy()
    color_layer[gt_only] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    color_layer[pred_only] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    color_layer[overlap] = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    mask_union = gt | pred
    overlay[mask_union] = (
        (1.0 - alpha) * image[mask_union]
        + alpha * color_layer[mask_union]
    )
    return np.clip(overlay, 0.0, 1.0)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def tensor_stats(tensor: torch.Tensor, prefix: str) -> Dict[str, object]:
    """Return compact diagnostics without assuming a particular value range."""
    values = tensor.detach().cpu().float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    stats: Dict[str, object] = {
        f"{prefix}_shape": "x".join(str(size) for size in values.shape),
        f"{prefix}_numel": values.numel(),
        f"{prefix}_finite_fraction": float(finite.float().mean()) if values.numel() else 0.0,
    }
    if finite_values.numel():
        stats.update(
            {
                f"{prefix}_min": float(finite_values.min()),
                f"{prefix}_max": float(finite_values.max()),
                f"{prefix}_mean": float(finite_values.mean()),
                f"{prefix}_std": float(finite_values.std(unbiased=False)),
                f"{prefix}_nonzero_pixels": int(torch.count_nonzero(finite_values)),
            }
        )
    return stats


def load_cached_tensors(cache_dir: Path, image_name: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Load tensors before myDataset normalization for scale debugging."""
    case_dir = cache_dir / str(image_name)
    return (
        torch.load(case_dir / "img.pt", map_location="cpu", weights_only=True),
        torch.load(case_dir / "mask.pt", map_location="cpu", weights_only=True),
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    training = import_training_module(args.training_module)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "prediction_masks"
    overlay_dir = output_dir / "overlays"
    input_dir = output_dir / "debug_inputs"
    ground_truth_dir = output_dir / "debug_ground_truth_masks"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_images:
        input_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_dir.mkdir(parents=True, exist_ok=True)

    test_cache = training.ensure_cache(
        dataset_path=dataset_dir,
        split="Test",
        input_size=args.input_size,
        loader=args.loader,
        image_npz_key=args.image_npz_key,
        mask_npz_key=args.mask_npz_key,
        intensity_normalization=args.intensity_normalization,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )

    test_loader = training.build_dataloader(
        dataset_path=test_cache,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        train=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")

    model = training.UNetEfficientNetB5(
        checkpoint_path=str(args.pretrained_checkpoint),
        pretrained=True,
    ).to(device)

    state_dict = load_state_dict(checkpoint_path, device)
    incompatibility = model.load_state_dict(state_dict, strict=True)
    print(incompatibility)
    model.eval()

    metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    warning_counts = {
        "empty_cached_image": 0,
        "empty_cached_mask": 0,
        "empty_processed_mask": 0,
        "empty_prediction": 0,
        "possible_double_scaling": 0,
    }

    for batch_index, batch in enumerate(test_loader, start=1):
        images = batch["images"].float().to(device, non_blocking=True)
        masks = batch["masks"].float().to(device, non_blocking=True)
        probabilities = model(images)

        predictions = probabilities >= args.threshold
        batch_rows = training.batch_metrics(probabilities, masks, args.threshold)
        image_names = list(batch["image_name"])

        for index, (image_name, metrics) in enumerate(zip(image_names, batch_rows)):
            filename = safe_filename(image_name)
            image = image_to_display(images[index])
            gt = masks[index].detach().cpu().numpy().squeeze() >= 0.5
            pred = predictions[index].detach().cpu().numpy().squeeze()
            raw_image, raw_mask = load_cached_tensors(test_cache, image_name)

            raw_image_nonzero = int(torch.count_nonzero(raw_image))
            raw_mask_nonzero = int(torch.count_nonzero(raw_mask))
            gt_pixels = int(np.count_nonzero(gt))
            pred_pixels = int(np.count_nonzero(pred))
            raw_image_max = float(raw_image.detach().float().max()) if raw_image.numel() else 0.0
            raw_mask_max = float(raw_mask.detach().float().max()) if raw_mask.numel() else 0.0
            processed_mask_max = (
                float(masks[index].detach().float().max())
                if masks[index].numel()
                else 0.0
            )
            # A [0,1] cache is valid. Flag it only if a positive cached mask
            # lost all foreground pixels during dataloader preprocessing.
            possible_double_scaling = (
                0.0 < raw_mask_max <= 1.0
                and processed_mask_max < 0.5
                and gt_pixels == 0
            )

            warning_counts["empty_cached_image"] += int(raw_image_nonzero == 0)
            warning_counts["empty_cached_mask"] += int(raw_mask_nonzero == 0)
            warning_counts["empty_processed_mask"] += int(gt_pixels == 0)
            warning_counts["empty_prediction"] += int(pred_pixels == 0)
            warning_counts["possible_double_scaling"] += int(possible_double_scaling)

            diagnostic_rows.append(
                {
                    "image_name": str(image_name),
                    **tensor_stats(raw_image, "cached_image"),
                    **tensor_stats(raw_mask, "cached_mask"),
                    **tensor_stats(images[index], "model_input"),
                    **tensor_stats(masks[index], "processed_mask"),
                    **tensor_stats(probabilities[index], "probability"),
                    "gt_positive_pixels_at_0.5": gt_pixels,
                    "prediction_positive_pixels": pred_pixels,
                    "possible_double_scaling": possible_double_scaling,
                }
            )

            prediction_png = (pred.astype(np.uint8) * 255)
            io.imsave(
                prediction_dir / f"{filename}.png",
                prediction_png,
                check_contrast=False,
            )

            overlay = make_overlay(image, gt, pred, args.overlay_alpha)
            io.imsave(
                overlay_dir / f"{filename}.png",
                (overlay * 255).round().astype(np.uint8),
                check_contrast=False,
            )
            if args.save_debug_images:
                io.imsave(
                    input_dir / f"{filename}.png",
                    (image * 255).round().astype(np.uint8),
                    check_contrast=False,
                )
                io.imsave(
                    ground_truth_dir / f"{filename}.png",
                    gt.astype(np.uint8) * 255,
                    check_contrast=False,
                )

            if args.verbose_diagnostics:
                print(
                    f"CHECK {image_name}: cached_image=[{float(raw_image.min()):.4g}, "
                    f"{raw_image_max:.4g}], cached_mask_max={raw_mask_max:.4g}, "
                    f"GT_pixels={gt_pixels}, probability=[{float(probabilities[index].min()):.4g}, "
                    f"{float(probabilities[index].max()):.4g}], pred_pixels={pred_pixels}"
                )

            metric_rows.append(
                {
                    "image_name": str(image_name),
                    **metrics,
                }
            )

        if batch_index % args.log_every == 0:
            print(
                f"Processed {batch_index * args.batch_size} test samples "
                f"(batches completed: {batch_index})"
            )

    numeric_rows = [
        {key: value for key, value in row.items() if key != "image_name"}
        for row in metric_rows
    ]
    summary = training.summarize_metric_rows(numeric_rows)
    summary.update(
        {
            "num_test_samples": len(metric_rows),
            "threshold": args.threshold,
            "intensity_normalization": args.intensity_normalization,
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
            "checkpoint": str(checkpoint_path),
            "dataset_dir": str(dataset_dir),
        }
    )

    save_csv(output_dir / "test_per_image_metrics.csv", metric_rows)
    save_csv(output_dir / "intermediate_diagnostics.csv", diagnostic_rows)
    save_json(output_dir / "test_metrics.json", summary)
    save_json(output_dir / "diagnostic_summary.json", warning_counts)

    if warning_counts["possible_double_scaling"]:
        print(
            "\nWARNING: foreground present in a [0,1] cached mask disappeared "
            "during dataloader preprocessing. Check that the evaluator imported "
            "the updated dataloader and training module."
        )
    print("Diagnostic counts:", warning_counts)

    print("\nFinal Test metrics")
    print("------------------")
    for key in (
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "relative_area_diff",
    ):
        if key in summary:
            print(f"{key}: {summary[key]:.4f}")
    print(f"Samples: {len(metric_rows)}")
    print(f"Metrics: {output_dir / 'test_metrics.json'}")
    print(f"Per-image metrics: {output_dir / 'test_per_image_metrics.csv'}")
    print(f"Predictions: {prediction_dir}")
    print(f"Overlays: {overlay_dir}")
    print(f"Intermediate diagnostics: {output_dir / 'intermediate_diagnostics.csv'}")
    print(f"Diagnostic summary: {output_dir / 'diagnostic_summary.json'}")
    print("Overlay legend: green=ground truth, red=prediction, yellow=overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained VersaMammo segmentation model on Test only, "
            "save metrics, binary masks, and image/GT/prediction overlays."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing Test/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained segmentation checkpoint to evaluate.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Original VersaMammo EfficientNet-B5 pretrained checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for metrics, predictions, and overlays.",
    )
    parser.add_argument(
        "--training-module",
        default=None,
        help=(
            "Module containing the corrected cache/model utilities. By default, "
            "tries versamammo_train_seg_v2 and then versamammo_train_seg."
        ),
    )
    parser.add_argument(
        "--loader",
        choices=["image", "numpy", "auto"],
        default="auto",
    )
    parser.add_argument("--image-npz-key", default=None)
    parser.add_argument("--mask-npz-key", default=None)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument(
        "--intensity-normalization",
        choices=["percentile", "minmax", "legacy"],
        default="percentile",
        help="Must match the normalization used when training the checkpoint.",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--verbose-diagnostics",
        action="store_true",
        help="Print cached/input/mask/prediction ranges for every test case.",
    )
    parser.add_argument(
        "--no-debug-images",
        dest="save_debug_images",
        action="store_false",
        help="Do not save model-input and ground-truth debug PNGs.",
    )
    parser.set_defaults(save_debug_images=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
