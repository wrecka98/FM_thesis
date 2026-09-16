from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
from torch.utils.data import DataLoader

from dataloader import myDataset, myNormalize, myRandomHFlip, myRandomVFlip
from preprocess import preprocess
from UNetEfficientNetB5 import UNetEfficientNetB5


CURRENT_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = CURRENT_DIR.parent
VERSAMAMMO_ROOT = DOWNSTREAM_DIR.parent
REPO_ROOT = VERSAMAMMO_ROOT.parent
DEFAULT_DATA_ROOT = VERSAMAMMO_ROOT / "datapre" / "segdetdata"
DEFAULT_SOTAS_DIR = DOWNSTREAM_DIR / "Sotas"
DEFAULT_RESULTS_DIR = REPO_ROOT / "pipelines_and_experiments" / "results" / "SEG_MamaMIA_versamammo_segmentation"
MODEL_DISPLAY_NAME = "VersaMammo_SEG_MamaMIA.pth"
SCRIPT_VERSION = "dataframe-patient-cv-v4"


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_array(
    path: Path,
    loader: str,
    npz_key: Optional[str] = None,
    array_role: str = "array",
) -> np.ndarray:
    """Load an image/mask as a NumPy array.

    ``np.load`` returns an ``NpzFile`` archive for .npz files, so this
    function explicitly extracts an array from that archive. If no key is
    supplied, a sensible role-specific key is tried and a single-array
    archive is accepted automatically.
    """
    if loader == "image":
        array = io.imread(path)
        return np.asarray(array)

    if loader not in {"numpy", "auto"}:
        raise ValueError(f"Unsupported loader: {loader}")

    if loader == "auto" and path.suffix.lower() not in {".npy", ".npz"}:
        return np.asarray(io.imread(path))

    loaded = np.load(path, allow_pickle=False)

    if isinstance(loaded, np.ndarray):  # .npy
        return loaded

    # .npz: loaded is numpy.lib.npyio.NpzFile and has no .shape attribute.
    try:
        available_keys = list(loaded.files)
        if not available_keys:
            raise ValueError(f"NPZ archive contains no arrays: {path}")

        if npz_key is not None:
            if npz_key not in available_keys:
                raise KeyError(
                    f"Key {npz_key!r} not found in {path}. "
                    f"Available keys: {available_keys}"
                )
            selected_key = npz_key
        elif len(available_keys) == 1:
            selected_key = available_keys[0]
        else:
            preferred_keys = (
                ("image", "img", "arr_0", "data")
                if array_role == "image"
                else ("mask", "segmentation", "label", "arr_0", "data")
            )
            selected_key = next(
                (key for key in preferred_keys if key in available_keys),
                None,
            )
            if selected_key is None:
                raise ValueError(
                    f"NPZ archive {path} contains multiple arrays "
                    f"{available_keys}. Supply the appropriate command-line "
                    f"NPZ key explicitly."
                )

        return np.asarray(loaded[selected_key])
    finally:
        loaded.close()


def spatial_shape(array: np.ndarray, name: str) -> Tuple[int, int]:
    """Return (height, width) for 2-D, HWC, or CHW arrays."""
    array = np.asarray(array)

    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])

    if array.ndim == 3:
        # Treat a small leading dimension as channels-first (C, H, W).
        if array.shape[0] in {1, 2, 3, 4} and array.shape[-1] not in {1, 2, 3, 4}:
            return int(array.shape[1]), int(array.shape[2])
        # Otherwise assume channels-last (H, W, C).
        return int(array.shape[0]), int(array.shape[1])

    raise ValueError(
        f"{name} must be a 2-D or 3-D array, but got shape {array.shape}."
    )


def case_file_paths(case_dir: Path, loader: str) -> Tuple[Path, Path]:
    """Resolve image and mask filenames for the selected loader."""
    candidates = {
        "image": [
            ("img.jpg", "mask.png"),
            ("image.jpg", "mask.png"),
            ("image.png", "mask.png"),
        ],
        "numpy": [
            ("image.npz", "mask.npz"),
            ("image.npy", "mask.npy"),
            ("img.npz", "mask.npz"),
            ("img.npy", "mask.npy"),
        ],
    }

    pairs = (
        candidates["image"] + candidates["numpy"]
        if loader == "auto"
        else candidates[loader]
    )

    for image_name, mask_name in pairs:
        image_path = case_dir / image_name
        mask_path = case_dir / mask_name
        if image_path.exists() and mask_path.exists():
            return image_path, mask_path

    expected = ", ".join(f"{a} + {b}" for a, b in pairs)
    raise FileNotFoundError(
        f"Could not find an image/mask pair in {case_dir}. "
        f"Expected one of: {expected}"
    )


def _to_chw_image(
    array: np.ndarray,
    path: Path,
    normalization: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> torch.Tensor:
    """Convert a 2-D/3-D NumPy image to a float32 CxHxW tensor."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        if array.shape[0] in {1, 2, 3, 4} and array.shape[-1] not in {1, 2, 3, 4}:
            pass  # already CHW
        else:
            array = np.moveaxis(array, -1, 0)  # HWC -> CHW
    else:
        raise ValueError(f"Unsupported image shape {array.shape} in {path}")

    tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)

    # EfficientNet expects three channels. Repeat grayscale images; for arrays
    # with more than three channels, retain the first three.
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 2:
        tensor = torch.cat([tensor, tensor[:1]], dim=0)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]

    if normalization == "percentile":
        # MRI arrays often occupy only a small part of an integer dtype's
        # theoretical range. Estimate the useful range from finite, nonzero
        # foreground voxels so zero-valued padding does not dominate the lower
        # percentile. Values outside the interval are clipped as outliers.
        finite = tensor[torch.isfinite(tensor)]
        foreground = finite[finite != 0]
        values = foreground if foreground.numel() else finite
        if values.numel():
            low = torch.quantile(values, percentile_low / 100.0)
            high = torch.quantile(values, percentile_high / 100.0)
            if high > low:
                tensor = ((tensor - low) / (high - low)).clamp(0.0, 1.0)
            else:
                tensor = torch.zeros_like(tensor)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    elif normalization == "minmax":
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel():
            minimum = finite.min()
            maximum = finite.max()
            tensor = (
                (tensor - minimum) / (maximum - minimum)
                if maximum > minimum
                else torch.zeros_like(tensor)
            )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    elif normalization == "legacy":
        if np.issubdtype(array.dtype, np.integer):
            dtype_max = float(np.iinfo(array.dtype).max)
            if dtype_max > 0:
                tensor = tensor / dtype_max
        elif tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
            minimum = tensor.min()
            maximum = tensor.max()
            if maximum > minimum:
                tensor = (tensor - minimum) / (maximum - minimum)
    else:
        raise ValueError(f"Unsupported intensity normalization: {normalization}")

    return tensor.contiguous()


def _to_chw_mask(array: np.ndarray, path: Path) -> torch.Tensor:
    """Convert a segmentation mask to a binary float32 1xHxW tensor."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        if array.shape[0] in {1, 2, 3, 4} and array.shape[-1] not in {1, 2, 3, 4}:
            pass
        else:
            array = np.moveaxis(array, -1, 0)
        array = array[:1]
    else:
        raise ValueError(f"Unsupported mask shape {array.shape} in {path}")

    tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)
    # Masks may be encoded as 0/1, 0/255, or other positive labels.
    tensor = (tensor > 0).float()
    return tensor.contiguous()


def _build_numpy_cache_case(
    image_path: Path,
    mask_path: Path,
    cached_case: Path,
    input_size: int,
    loader: str,
    image_npz_key: Optional[str],
    mask_npz_key: Optional[str],
    intensity_normalization: str,
    percentile_low: float,
    percentile_high: float,
) -> None:
    """Create img.pt and mask.pt directly from NPY/NPZ source arrays."""
    image = load_array(image_path, loader, image_npz_key, "image")
    mask = load_array(mask_path, loader, mask_npz_key, "mask")

    image_tensor = _to_chw_image(
        image,
        image_path,
        intensity_normalization,
        percentile_low,
        percentile_high,
    )
    mask_tensor = _to_chw_mask(mask, mask_path)

    target_size = (input_size, input_size)
    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask_tensor = F.interpolate(
        mask_tensor.unsqueeze(0),
        size=target_size,
        mode="nearest",
    ).squeeze(0)

    cached_case.mkdir(parents=True, exist_ok=True)
    torch.save(image_tensor.contiguous(), cached_case / "img.pt")
    torch.save(mask_tensor.contiguous(), cached_case / "mask.pt")


def ensure_cache(
    dataset_path: Path,
    split: str,
    input_size: int,
    loader: str,
    image_npz_key: Optional[str] = None,
    mask_npz_key: Optional[str] = None,
    intensity_normalization: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Path:
    raw_path = dataset_path / split
    if not 0.0 <= percentile_low < percentile_high <= 100.0:
        raise ValueError(
            "Percentiles must satisfy 0 <= percentile_low < percentile_high <= 100."
        )
    normalization_tag = intensity_normalization
    if intensity_normalization == "percentile":
        normalization_tag += f"_{percentile_low:g}_{percentile_high:g}"
    cache_path = dataset_path / f"{split}_cache_{input_size}_{normalization_tag}"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing VersaMammo {split} folder: {raw_path}")

    case_records = []
    for case_dir in sorted(raw_path.iterdir()):
        if not case_dir.is_dir():
            continue

        image_path, mask_path = case_file_paths(case_dir, loader)
        image = load_array(image_path, loader, image_npz_key, "image")
        mask = load_array(mask_path, loader, mask_npz_key, "mask")
        image_hw = spatial_shape(image, f"Image in {case_dir}")
        mask_hw = spatial_shape(mask, f"Mask in {case_dir}")
        if image_hw != mask_hw:
            raise ValueError(
                f"Image/mask shape mismatch in {case_dir}: image={image.shape} "
                f"(spatial={image_hw}), mask={mask.shape} (spatial={mask_hw})."
            )
        case_records.append((case_dir, image_path, mask_path))

    cache_path.mkdir(parents=True, exist_ok=True)

    if loader in {"numpy", "auto"}:
        for case_dir, image_path, mask_path in case_records:
            cached_case = cache_path / case_dir.name
            cached_image = cached_case / "img.pt"
            cached_mask = cached_case / "mask.pt"
            stale = (
                not cached_image.exists()
                or not cached_mask.exists()
                or image_path.stat().st_mtime > cached_image.stat().st_mtime
                or mask_path.stat().st_mtime > cached_mask.stat().st_mtime
            )
            if stale:
                print(f"Building cache: {split}/{case_dir.name}")
                _build_numpy_cache_case(
                    image_path,
                    mask_path,
                    cached_case,
                    input_size,
                    loader,
                    image_npz_key,
                    mask_npz_key,
                    intensity_normalization,
                    percentile_low,
                    percentile_high,
                )
    else:
        needs_preprocess = any(
            not (cache_path / case_dir.name / "img.pt").exists()
            or not (cache_path / case_dir.name / "mask.pt").exists()
            for case_dir, _, _ in case_records
        )
        if needs_preprocess:
            preprocess(str(raw_path), str(cache_path), [input_size, input_size])

    # Never allow an incomplete cache to reach a DataLoader worker, where the
    # resulting exception is much harder to diagnose.
    missing = []
    for case_dir, _, _ in case_records:
        cached_case = cache_path / case_dir.name
        for filename in ("img.pt", "mask.pt"):
            if not (cached_case / filename).exists():
                missing.append(str(cached_case / filename))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Cache generation for {split} is incomplete. Missing "
            f"{len(missing)} file(s), including:\n{preview}"
        )

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
    )


def configure_finetune(model: nn.Module, finetune: str) -> None:
    if finetune in {"full", "fft"}:
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        if name.startswith("backbone.") or name.startswith("encoder"):
            param.requires_grad = False
        else:
            param.requires_grad = True


def build_model(args: argparse.Namespace) -> nn.Module:
    checkpoint_path = args.pretrained_checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.sotas_dir / f"{MODEL_DISPLAY_NAME}.pth"

    model = UNetEfficientNetB5(checkpoint_path=str(checkpoint_path), pretrained=True)
    configure_finetune(model, args.finetune)
    return model


def segmentation_loss(preds: torch.Tensor, targets: torch.Tensor, dice_loss: DiceLoss) -> torch.Tensor:
    bce_loss = nn.functional.binary_cross_entropy(preds, targets)
    return bce_loss + dice_loss(preds, targets)


def batch_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float) -> List[Dict[str, float]]:
    pred_bin = preds.detach().cpu() >= threshold
    target_bin = targets.detach().cpu() >= 0.5
    rows = []

    for pred, target in zip(pred_bin, target_bin):
        pred = pred.bool().reshape(-1)
        target = target.bool().reshape(-1)
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

        rows.append(
            {
                "dice": float(dice),
                "iou": float(iou),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "accuracy": float(accuracy),
                "relative_area_diff": float(relative_area_diff),
            }
        )

    return rows


def summarize_metric_rows(rows: List[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {
        f"{prefix}{key}": float(np.mean([row[key] for row in rows]))
        for key in keys
    }


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_prediction_masks(output_dir: Path, image_names: List[str], preds: torch.Tensor, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_masks = (preds.detach().cpu().numpy() >= threshold).astype(np.uint8) * 255
    for image_name, pred_mask in zip(image_names, pred_masks):
        io.imsave(output_dir / f"{image_name}.png", np.squeeze(pred_mask), check_contrast=False)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    prediction_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    dice_loss = DiceLoss()
    metric_rows: List[Dict[str, float]] = []
    losses = []

    for batch in dataloader:
        images = batch["images"].float().to(device)
        masks = batch["masks"].float().to(device)
        preds = model(images)
        loss = segmentation_loss(preds, masks, dice_loss)
        losses.append(float(loss.item()))

        if "image_name" in batch:
            image_names = [str(value) for value in batch["image_name"]]
        elif "source_index" in batch:
            # Dataframe rows can share ImagePath when they describe different
            # ROIs. Prefix with the source index to avoid overwriting masks.
            image_names = [f"row_{value}" for value in batch["source_index"]]
        else:
            image_names = [f"batch_item_{index}" for index in range(len(images))]
        batch_rows = batch_metrics(preds, masks, args.threshold)
        for image_name, row in zip(image_names, batch_rows):
            metric_rows.append({"image_name": image_name, **row})

        if prediction_dir is not None:
            save_prediction_masks(prediction_dir, image_names, preds, args.threshold)

    metrics = summarize_metric_rows([{k: v for k, v in row.items() if k != "image_name"} for row in metric_rows])
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, metric_rows


def train_one_dataset(
    args: argparse.Namespace,
    dataset: str,
    dataset_path: Path,
    fold: Optional[int] = None,
) -> Dict[str, float]:
    dataset_dir = args.results_dir / dataset
    checkpoint_path = args.save_dir / dataset / f"{MODEL_DISPLAY_NAME}.pth"

    cache_options = {
        "image_npz_key": args.image_npz_key,
        "mask_npz_key": args.mask_npz_key,
        "intensity_normalization": args.intensity_normalization,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
    }
    train_cache = ensure_cache(
        dataset_path, "Train", args.input_size, args.loader, **cache_options
    )

    eval_raw_path = dataset_path / "Eval"
    if args.validation == "required" and not eval_raw_path.exists():
        raise FileNotFoundError(
            f"Validation was required, but the Eval folder is missing: {eval_raw_path}"
        )

    use_validation = args.validation == "required" or (
        args.validation == "auto" and eval_raw_path.exists()
    )
    eval_cache = (
        ensure_cache(
            dataset_path, "Eval", args.input_size, args.loader,
            **cache_options,
        )
        if use_validation
        else None
    )

    if use_validation:
        print(f"Validation enabled using: {eval_raw_path}")
    else:
        print(
            "Validation disabled. The final training state will be saved as the checkpoint; "
            "early stopping and best-validation checkpoint selection are inactive."
        )

    test_cache = ensure_cache(
        dataset_path, "Test", args.input_size, args.loader, **cache_options
    )

    train_loader = build_dataloader(
        train_cache,
        args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        train=True,
    )
    eval_loader = (
        build_dataloader(
            eval_cache,
            args.batch_size_eval,
            shuffle=False,
            num_workers=args.num_workers_eval,
            train=False,
        )
        if eval_cache is not None
        else None
    )
    test_loader = build_dataloader(
        test_cache,
        args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
        train=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")
    model = build_model(args).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    dice_loss = DiceLoss()

    best_metric = -1.0
    best_eval_metrics: Dict[str, float] = {}
    stale_validations = 0
    iteration = 0
    log_rows = []
    start = time.time()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            iteration += 1
            images = batch["images"].float().to(device)
            masks = batch["masks"].float().to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = segmentation_loss(preds, masks, dice_loss)
            loss.backward()
            optimizer.step()

            if iteration % args.log_every == 0:
                print(f"{dataset} epoch={epoch + 1} iter={iteration} loss={loss.item():.5f}")

            if use_validation and iteration % args.eval_every == 0:
                assert eval_loader is not None
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
                save_metrics_csv(dataset_dir / "validation_history.csv", log_rows)
                print(
                    f"{dataset} validation iter={iteration} "
                    f"{args.selection_metric}={selection_metric:.4f} "
                    f"dice={eval_metrics['dice']:.4f}"
                )

                if selection_metric > best_metric:
                    stale_validations = 0
                    best_metric = selection_metric
                    best_eval_metrics = eval_metrics
                    torch.save(model.state_dict(), checkpoint_path)
                    save_json(dataset_dir / "best_validation_metrics.json", eval_metrics)
                    print(f"Saved best checkpoint: {checkpoint_path}")
                else:
                    stale_validations += 1

                if stale_validations >= args.early_stop:
                    break

            if iteration >= args.max_iter:
                break

        if stale_validations >= args.early_stop or iteration >= args.max_iter:
            break

    if not use_validation:
        # Without validation there is no "best" checkpoint. Save the final
        # training state unconditionally, replacing any stale checkpoint.
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved final checkpoint: {checkpoint_path}")
    elif not checkpoint_path.exists():
        # Validation was enabled, but training ended before the first scheduled
        # validation. Save the current state and evaluate it once.
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint before first scheduled validation: {checkpoint_path}")
        assert eval_loader is not None
        best_eval_metrics, _ = evaluate_model(model, eval_loader, device, args)
        save_json(dataset_dir / "best_validation_metrics.json", best_eval_metrics)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    prediction_dir = dataset_dir / "test_predictions"
    test_metrics, test_rows = evaluate_model(model, test_loader, device, args, prediction_dir)
    save_json(dataset_dir / "test_metrics.json", test_metrics)
    save_metrics_csv(dataset_dir / "test_per_image_metrics.csv", test_rows)

    result = {
        "dataset": dataset,
        "fold": float(fold) if fold is not None else "",
        "training_seconds": float(time.time() - start),
        "checkpoint": str(checkpoint_path),
        **{f"val_{key}": value for key, value in best_eval_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(dataset_dir / "fold_result.json", result)
    return result


def evaluate_saved_dataset(
    args: argparse.Namespace,
    dataset: str,
    dataset_path: Path,
    fold: Optional[int] = None,
) -> Dict[str, float]:
    dataset_dir = args.results_dir / dataset
    checkpoint_path = args.eval_checkpoint or args.save_dir / dataset / f"{MODEL_DISPLAY_NAME}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")

    test_cache = ensure_cache(
        dataset_path,
        "Test",
        args.input_size,
        args.loader,
        image_npz_key=args.image_npz_key,
        mask_npz_key=args.mask_npz_key,
        intensity_normalization=args.intensity_normalization,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )
    test_loader = build_dataloader(
        test_cache,
        args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
        train=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")
    model = build_model(args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    prediction_dir = dataset_dir / "test_predictions"
    test_metrics, test_rows = evaluate_model(model, test_loader, device, args, prediction_dir)
    save_json(dataset_dir / "test_metrics.json", test_metrics)
    save_metrics_csv(dataset_dir / "test_per_image_metrics.csv", test_rows)

    result = {
        "dataset": dataset,
        "fold": float(fold) if fold is not None else "",
        "checkpoint": str(checkpoint_path),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(dataset_dir / "fold_result.json", result)
    return result


def aggregate_results(
    results: List[Dict[str, float]],
    results_dir: Path,
    aggregate_name: str,
) -> Dict[str, Dict[str, float]]:
    metric_keys = [key for key in results[0].keys() if key.startswith("test_") or key.startswith("val_")]
    aggregate = {}
    for key in metric_keys:
        values = np.array([float(result[key]) for result in results], dtype=np.float32)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    save_json(results_dir / f"{aggregate_name}_aggregate_metrics.json", aggregate)
    rows = [
        {"metric": metric, "mean": values["mean"], "std": values["std"]}
        for metric, values in aggregate.items()
    ]
    save_metrics_csv(results_dir / f"{aggregate_name}_aggregate_metrics.csv", rows)
    return aggregate


def dataframe_loader(
    dataframe,
    args: argparse.Namespace,
    *,
    train: bool,
) -> DataLoader:
    """Build the existing PNG dataframe loader without duplicating its IO logic."""
    from run_versamammo_segmentation_dataframe import PngDataframeDataset

    dataset = PngDataframeDataset(
        dataframe=dataframe,
        path_root=args.path_root,
        input_size=args.input_size,
        image_scale=args.image_scale,
        mask_threshold=args.mask_threshold,
        augment=train and not args.no_augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size_train if train else args.batch_size_eval,
        shuffle=train,
        num_workers=args.num_workers if train else args.num_workers_eval,
        pin_memory=torch.cuda.is_available(),
    )


def train_dataframe_fold(
    args: argparse.Namespace,
    train_dataframe,
    validation_dataframe,
    test_dataframe,
    fold: int,
) -> Dict[str, float]:
    """Train one patient-level CV fold and evaluate the fixed held-out test set."""
    dataset = f"{args.experiment_name}_fold{fold}"
    dataset_dir = args.results_dir / dataset
    checkpoint_path = args.save_dir / dataset / f"{MODEL_DISPLAY_NAME}.pth"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # These manifests make the exact CV partition independently auditable.
    train_dataframe.to_csv(dataset_dir / "train_split.csv", index=True)
    validation_dataframe.to_csv(dataset_dir / "validation_split.csv", index=True)

    train_loader = dataframe_loader(train_dataframe, args, train=True)
    validation_loader = dataframe_loader(validation_dataframe, args, train=False)
    test_loader = dataframe_loader(test_dataframe, args, train=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")
    # The dataframe trainer loads VersaMammo encoder weights by state-dict
    # structure rather than by checkpoint filename and avoids network access.
    from run_versamammo_segmentation_dataframe import build_model as build_dataframe_model

    model = build_dataframe_model(args).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    dice_loss = DiceLoss()

    best_metric = -1.0
    best_validation_metrics: Dict[str, float] = {}
    stale_validations = 0
    iteration = 0
    validation_history = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            iteration += 1
            images = batch["images"].float().to(device, non_blocking=True)
            masks = batch["masks"].float().to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(images)
            loss = segmentation_loss(predictions, masks, dice_loss)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if iteration % args.log_every == 0:
                print(f"{dataset} epoch={epoch + 1} iter={iteration} loss={loss.item():.5f}")
            if iteration >= args.max_iter:
                break

        # Validate at every epoch end. This is safer for small folds than the
        # legacy iteration-only cadence, which may never fire.
        validation_metrics, _ = evaluate_model(model, validation_loader, device, args)
        selection_metric = validation_metrics[args.selection_metric]
        validation_history.append(
            {
                "dataset": dataset,
                "fold": fold,
                "epoch": epoch + 1,
                "iteration": iteration,
                "train_loss": float(np.mean(epoch_losses)),
                **validation_metrics,
            }
        )
        save_metrics_csv(dataset_dir / "validation_history.csv", validation_history)
        print(
            f"{dataset} validation epoch={epoch + 1} "
            f"{args.selection_metric}={selection_metric:.4f}"
        )

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_validation_metrics = validation_metrics
            stale_validations = 0
            torch.save(model.state_dict(), checkpoint_path)
            save_json(dataset_dir / "best_validation_metrics.json", validation_metrics)
        else:
            stale_validations += 1

        if stale_validations >= args.early_stop or iteration >= args.max_iter:
            break

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()})
    test_metrics, test_rows = evaluate_model(
        model, test_loader, device, args, dataset_dir / "test_predictions"
    )
    save_json(dataset_dir / "test_metrics.json", test_metrics)
    save_metrics_csv(dataset_dir / "test_per_image_metrics.csv", test_rows)

    result = {
        "dataset": dataset,
        "fold": float(fold),
        "n_train_rows": len(train_dataframe),
        "n_validation_rows": len(validation_dataframe),
        "n_test_rows": len(test_dataframe),
        "n_train_patients": int(train_dataframe[args.group_column].nunique()),
        "n_validation_patients": int(validation_dataframe[args.group_column].nunique()),
        "n_test_patients": int(test_dataframe[args.group_column].nunique()),
        "epochs_completed": len(validation_history),
        "iterations_completed": iteration,
        "training_seconds": float(time.time() - start),
        "checkpoint": str(checkpoint_path),
        **{f"val_{key}": value for key, value in best_validation_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(dataset_dir / "fold_result.json", result)
    return result


def run_dataframe_cross_validation(args: argparse.Namespace) -> None:
    """Create patient-level folds from train.pkl and keep test.pkl untouched."""
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold
    from run_versamammo_segmentation_dataframe import validate_dataframe

    train_dataframe = pd.read_pickle(args.train_dataframe)
    test_dataframe = pd.read_pickle(args.test_dataframe)
    for path, dataframe in (
        (args.train_dataframe, train_dataframe),
        (args.test_dataframe, test_dataframe),
    ):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"{path} does not contain a pandas DataFrame.")
        validate_dataframe(dataframe, args.path_root)
        if args.group_column not in dataframe.columns:
            raise ValueError(f"Missing grouping column {args.group_column!r} in {path}")

    train_groups = set(train_dataframe[args.group_column].astype(str))
    test_groups = set(test_dataframe[args.group_column].astype(str))
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(
            f"Patient leakage: {len(overlap)} {args.group_column} value(s) occur in both "
            f"train and test pickles. Examples: {overlap[:10]}"
        )
    if len(train_groups) < args.cv_splits:
        raise ValueError(
            f"--cv-splits={args.cv_splits} exceeds the {len(train_groups)} unique "
            f"training patients."
        )

    patient_table = train_dataframe[[args.group_column]].drop_duplicates().copy()
    patient_table[args.group_column] = patient_table[args.group_column].astype(str)
    if args.stratify_column and args.stratify_column in train_dataframe.columns:
        labels_per_patient = train_dataframe.groupby(args.group_column)[
            args.stratify_column
        ].nunique(dropna=False)
        inconsistent = labels_per_patient[labels_per_patient != 1]
        if not inconsistent.empty:
            raise ValueError(
                f"{args.stratify_column!r} must be constant within each patient. "
                f"Inconsistent {args.group_column} values: {list(inconsistent.index[:10])}"
            )
        patient_labels = (
            train_dataframe[[args.group_column, args.stratify_column]]
            .drop_duplicates(subset=[args.group_column])
            .assign(**{args.group_column: lambda frame: frame[args.group_column].astype(str)})
        )
        patient_table = patient_table.merge(
            patient_labels, on=args.group_column, how="left", validate="one_to_one"
        )
        splitter = StratifiedKFold(
            n_splits=args.cv_splits, shuffle=True, random_state=args.seed
        )
        splits = splitter.split(patient_table, y=patient_table[args.stratify_column])
        split_method = "patient-level StratifiedKFold"
    else:
        if args.stratify_column:
            print(
                f"Warning: {args.stratify_column!r} is absent; using GroupKFold "
                "without label stratification."
            )
        splitter = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
        splits = splitter.split(patient_table)
        split_method = "patient-level KFold"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.results_dir / f"{args.experiment_name}_cv_config.json",
        {
            "train_dataframe": str(args.train_dataframe.resolve()),
            "test_dataframe": str(args.test_dataframe.resolve()),
            "path_root": str(args.path_root.resolve()),
            "split_method": split_method,
            "cv_splits": args.cv_splits,
            "group_column": args.group_column,
            "stratify_column": args.stratify_column,
            "seed": args.seed,
            "fixed_test_set": True,
        },
    )

    requested_folds = set(args.folds)
    results = []
    for fold, (train_patient_indices, validation_patient_indices) in enumerate(splits):
        if fold not in requested_folds:
            continue
        train_patient_ids = set(
            patient_table.iloc[train_patient_indices][args.group_column].astype(str)
        )
        validation_patient_ids = set(
            patient_table.iloc[validation_patient_indices][args.group_column].astype(str)
        )
        patient_ids = train_dataframe[args.group_column].astype(str)
        fold_train = train_dataframe.loc[patient_ids.isin(train_patient_ids)].copy()
        fold_validation = train_dataframe.loc[
            patient_ids.isin(validation_patient_ids)
        ].copy()
        fold_train_groups = set(fold_train[args.group_column].astype(str))
        fold_validation_groups = set(fold_validation[args.group_column].astype(str))
        if fold_train_groups & fold_validation_groups:
            raise RuntimeError(f"Patient leakage detected while constructing fold {fold}.")
        print(
            f"=== Fold {fold}: {len(fold_train)} train rows / "
            f"{len(fold_validation)} validation rows / {len(test_dataframe)} fixed test rows ==="
        )
        results.append(
            train_dataframe_fold(
                args, fold_train, fold_validation, test_dataframe, fold
            )
        )

    if not results:
        raise ValueError(
            f"No folds selected. --folds requested {sorted(requested_folds)}, "
            f"but --cv-splits creates folds 0-{args.cv_splits - 1}."
        )
    save_metrics_csv(
        args.results_dir / f"{args.experiment_name}_fold_results.csv", results
    )
    aggregate = aggregate_results(results, args.results_dir, args.experiment_name)
    print("Aggregate CV metrics:")
    for metric in ("val_dice", "test_dice", "test_iou", "test_sensitivity", "test_precision"):
        if metric in aggregate:
            values = aggregate[metric]
            print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate VersaMammo segmentation over patient-level folds."
    )
    parser.add_argument("--dataset-prefix", default="ZGT_VersaMammo")
    parser.add_argument("--folds", default="0-4", help="Comma/range list, e.g. 0-4 or 0,2,4.")
    parser.add_argument(
        "--train-dataframe",
        type=Path,
        default=None,
        help="Pickled training DataFrame. Enables patient-level cross-validation mode.",
    )
    parser.add_argument(
        "--test-dataframe",
        type=Path,
        default=None,
        help="Fixed held-out test DataFrame used only after selecting each fold checkpoint.",
    )
    parser.add_argument(
        "--path-root",
        type=Path,
        default=Path("/mnt/data/spathak"),
        help="Root against which relative ImagePath and ROIPath values are resolved.",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--group-column", default="PatientID")
    parser.add_argument(
        "--stratify-column",
        default="PatientGroundtruth",
        help="Patient outcome used by StratifiedGroupKFold; falls back to GroupKFold if absent.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated exact dataset folder names under data-root. Overrides --dataset-prefix/--folds.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Direct path to one dataset containing Train/Test and optionally Eval, such as the "
            "VersaMammo_format directory produced by INbreast.ipynb. Skips folds."
        ),
    )
    parser.add_argument("--sotas-dir", type=Path, default=DEFAULT_SOTAS_DIR)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=CURRENT_DIR / "saved_model")
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument(
        "--image-scale",
        choices=["auto", "255", "unit", "minmax"],
        default="auto",
        help="PNG intensity scaling in dataframe mode.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--intensity-normalization",
        choices=["percentile", "minmax", "legacy"],
        default="percentile",
        help=(
            "Image scaling before caching. 'percentile' clips and scales finite "
            "nonzero intensities (recommended for MRI); 'minmax' uses the full "
            "observed range; 'legacy' retains dtype-maximum scaling."
        ),
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument(
        "--loader",
        choices=["image", "numpy", "auto"],
        default="auto",
        help=(
            "Raw data loader: 'image' uses skimage.io.imread for JPG/PNG; "
            "'numpy' loads NPY/NPZ; 'auto' selects from the file extension. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--image-npz-key",
        default=None,
        help=(
            "Array key inside image NPZ files. Not needed for single-array NPZ "
            "archives. Example: --image-npz-key image"
        ),
    )
    parser.add_argument(
        "--mask-npz-key",
        default=None,
        help=(
            "Array key inside mask NPZ files. Not needed for single-array NPZ "
            "archives. Example: --mask-npz-key mask"
        ),
    )
    parser.add_argument("--finetune", choices=["head", "full", "lp", "fft"], default="head")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=10000000)
    parser.add_argument("--batch-size-train", type=int, default=8)
    parser.add_argument("--batch-size-eval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-workers-eval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--validation",
        choices=["auto", "required", "disabled"],
        default="auto",
        help=(
            "Validation behavior: 'auto' uses Eval when present and otherwise skips it; "
            "'required' raises an error if Eval is missing; 'disabled' never uses Eval. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Validate every N training iterations when validation is enabled. Default: 500.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--selection-metric", default="dice")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")

    args = parser.parse_args()
    args.folds = parse_folds(args.folds)
    if args.datasets:
        args.datasets = parse_datasets(args.datasets)
    return args


def main() -> None:
    print(f"Running versamammo_train_seg.py version: {SCRIPT_VERSION}")
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataframe_mode = args.train_dataframe is not None or args.test_dataframe is not None
    if dataframe_mode:
        if args.train_dataframe is None or args.test_dataframe is None:
            raise ValueError(
                "Dataframe cross-validation requires both --train-dataframe and "
                "--test-dataframe."
            )
        if not args.train_dataframe.is_file() or not args.test_dataframe.is_file():
            raise FileNotFoundError(
                f"Missing train/test pickle: {args.train_dataframe}, {args.test_dataframe}"
            )
        if args.cv_splits < 2:
            raise ValueError("--cv-splits must be at least 2.")
        if args.eval_only:
            raise ValueError(
                "--eval-only is not supported in dataframe CV mode; use the saved "
                "fold checkpoints with the dedicated evaluation script."
            )
        if args.validation == "disabled":
            raise ValueError(
                "Dataframe CV mode requires validation folds; do not pass "
                "--validation disabled."
            )
        if args.pretrained_checkpoint is None or not args.pretrained_checkpoint.is_file():
            raise FileNotFoundError(
                "Dataframe cross-validation requires an existing --pretrained-checkpoint."
            )
        args.experiment_name = args.experiment_name or args.train_dataframe.stem
        run_dataframe_cross_validation(args)
        return

    if args.dataset_dir:
        dataset_path = args.dataset_dir.expanduser().resolve()
        jobs = [(dataset_path.name, dataset_path, None)]
        aggregate_name = args.experiment_name or dataset_path.name
    elif args.datasets:
        jobs = [(dataset, args.data_root / dataset, None) for dataset in args.datasets]
        aggregate_name = args.experiment_name or "_".join(args.datasets)
    else:
        jobs = [
            (f"{args.dataset_prefix}_fold{fold}", args.data_root / f"{args.dataset_prefix}_fold{fold}", fold)
            for fold in args.folds
        ]
        aggregate_name = args.experiment_name or args.dataset_prefix

    results = []
    for dataset, dataset_path, fold in jobs:
        label = f"Fold {fold}: {dataset}" if fold is not None else dataset
        print(f"=== {label} ===")
        if args.eval_only:
            results.append(evaluate_saved_dataset(args, dataset, dataset_path, fold))
        else:
            results.append(train_one_dataset(args, dataset, dataset_path, fold))

    save_metrics_csv(args.results_dir / f"{aggregate_name}_fold_results.csv", results)
    aggregate = aggregate_results(results, args.results_dir, aggregate_name)

    print("Aggregate test metrics:")
    for metric in ["test_dice", "test_iou", "test_sensitivity", "test_precision"]:
        if metric in aggregate:
            values = aggregate[metric]
            print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
