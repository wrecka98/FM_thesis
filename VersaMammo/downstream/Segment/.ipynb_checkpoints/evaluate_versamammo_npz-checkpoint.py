from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.data import DataLoader, Dataset

from UNetEfficientNetB5 import UNetEfficientNetB5


def load_npz_array(path: Path, key: Optional[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if key is not None:
            if key not in archive:
                raise KeyError(f"{path} does not contain key {key!r}; available keys: {keys}")
            return np.asarray(archive[key])
        if len(keys) != 1:
            raise ValueError(
                f"{path} contains {len(keys)} arrays ({keys}). "
                "Specify the appropriate --image-key or --mask-key."
            )
        return np.asarray(archive[keys[0]])


def image_to_chw(image: np.ndarray, path: Path) -> torch.Tensor:
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.repeat(image[None, ...], 3, axis=0)
    elif image.ndim == 3:
        if image.shape[0] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = np.moveaxis(image, -1, 0)
        else:
            raise ValueError(
                f"Cannot infer channels for {path}, shape={image.shape}; "
                "expected HxW, 1xHxW, 3xHxW, HxWx1, or HxWx3."
            )
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
    else:
        raise ValueError(f"Unsupported image shape in {path}: {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(image)).float()


def mask_to_chw(mask: np.ndarray, path: Path) -> torch.Tensor:
    mask = np.squeeze(mask)
    if mask.ndim == 3 and mask.shape[-1] in (1, 3):
        mask = mask[..., 0]
    elif mask.ndim == 3 and mask.shape[0] in (1, 3):
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape in {path}: {mask.shape}")
    return torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()


def scale_image(image: torch.Tensor, mode: str) -> torch.Tensor:
    minimum = float(image.min())
    maximum = float(image.max())
    if mode == "unit":
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"--image-scale unit requires values in [0, 1], got [{minimum}, {maximum}]"
            )
        scaled = image
    elif mode == "255":
        scaled = image / 255.0
    elif mode == "minmax":
        scaled = (image - minimum) / (maximum - minimum) if maximum > minimum else image * 0.0
    else:
        if minimum >= 0.0 and maximum <= 1.0:
            scaled = image
        elif minimum >= 0.0 and maximum <= 255.0:
            scaled = image / 255.0
        else:
            scaled = (
                (image - minimum) / (maximum - minimum) if maximum > minimum else image * 0.0
            )
    return (scaled - 0.5) / 0.5


def binarize_mask(mask: torch.Tensor, threshold: Optional[float]) -> torch.Tensor:
    if threshold is None:
        threshold = 0.5 if float(mask.max()) <= 1.0 else 0.0
    return (mask > threshold).float()


class PatientNpzDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_filename: str,
        mask_filename: str,
        image_key: Optional[str],
        mask_key: Optional[str],
        input_size: int,
        image_scale: str,
        mask_threshold: Optional[float],
    ) -> None:
        self.data_dir = data_dir
        self.image_filename = image_filename
        self.mask_filename = mask_filename
        self.image_key = image_key
        self.mask_key = mask_key
        self.input_size = input_size
        self.image_scale = image_scale
        self.mask_threshold = mask_threshold

        image_paths = sorted(data_dir.rglob(image_filename))
        self.samples: List[Tuple[Path, Path, str]] = []
        for image_path in image_paths:
            mask_path = image_path.parent / mask_filename
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Found image without matching {mask_filename}: {image_path}"
                )
            sample_id = image_path.parent.relative_to(data_dir).as_posix()
            self.samples.append((image_path, mask_path, sample_id))
        if not self.samples:
            raise FileNotFoundError(
                f"No {image_filename} files with matching {mask_filename} found under {data_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, mask_path, sample_id = self.samples[index]
        image = image_to_chw(load_npz_array(image_path, self.image_key), image_path)
        mask = mask_to_chw(load_npz_array(mask_path, self.mask_key), mask_path)
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"Image/mask shape mismatch for {sample_id}: "
                f"{tuple(image.shape[-2:])} vs {tuple(mask.shape[-2:])}"
            )

        image = scale_image(image, self.image_scale)
        mask = binarize_mask(mask, self.mask_threshold)
        size = (self.input_size, self.input_size)
        image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask[None], size=size, mode="nearest")[0]
        return {"image": image, "mask": mask, "sample_id": sample_id}


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state-dict or a dictionary containing one.")
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("No tensor state-dict was found in the checkpoint.")
    return state_dict


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = UNetEfficientNetB5(pretrained=False, checkpoint_path=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not load {checkpoint_path} as a complete VersaMammo segmentation model. "
            "Supply the trained segmentation checkpoint containing both encoder and decoder weights, "
            "not the original backbone-only pretraining checkpoint."
        ) from error
    return model.to(device).eval()


def calculate_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = prediction.bool().reshape(-1)
    target = target.bool().reshape(-1)
    tp = torch.logical_and(prediction, target).sum().item()
    tn = torch.logical_and(~prediction, ~target).sum().item()
    fp = torch.logical_and(prediction, ~target).sum().item()
    fn = torch.logical_and(~prediction, target).sum().item()

    return {
        "dice": 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "sensitivity": tp / (tp + fn) if tp + fn else 1.0,
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
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
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        probabilities = model(images).cpu()
        predictions = probabilities >= threshold

        for sample_id, prediction, mask in zip(batch["sample_id"], predictions, masks):
            row: Dict[str, object] = {"sample_id": sample_id}
            row.update(calculate_metrics(prediction, mask >= 0.5))
            rows.append(row)

            output_path = prediction_dir / f"{sample_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(
                output_path,
                prediction.squeeze(0).numpy().astype(np.uint8) * 255,
                check_contrast=False,
            )

    metric_names = [key for key in rows[0] if key != "sample_id"]
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
        description="Evaluate a trained VersaMammo segmentation model on patient-level NPZ data."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-filename", default="img.npz")
    parser.add_argument("--mask-filename", default="mask.npz")
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--mask-key", default=None)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="Default: 0.5 for masks in [0,1], otherwise >0 is foreground.",
    )
    parser.add_argument(
        "--image-scale",
        choices=("auto", "unit", "255", "minmax"),
        default="auto",
        help="Intensity conversion before VersaMammo's [-1,1] normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = PatientNpzDataset(
        args.data_dir,
        args.image_filename,
        args.mask_filename,
        args.image_key,
        args.mask_key,
        args.input_size,
        args.image_scale,
        args.mask_threshold,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Evaluating {len(dataset)} samples on {device}.")
    model = load_model(args.checkpoint, device)
    rows, summary = evaluate(
        model, loader, device, args.prediction_threshold, args.output_dir
    )

    write_csv(args.output_dir / "per_image_metrics.csv", rows)
    with (args.output_dir / "aggregate_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                **vars(args),
                "data_dir": str(args.data_dir.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "samples": len(dataset),
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
