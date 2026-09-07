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
DEFAULT_RESULTS_DIR = REPO_ROOT / "pipelines_and_experiments" / "results" / "SEG_INbreast_versamammo_segmentation"
MODEL_DISPLAY_NAME = "VersaMammo (Enb5)"
SCRIPT_VERSION = "npz-cache-fix-v2"


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


def _to_chw_image(array: np.ndarray, path: Path) -> torch.Tensor:
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

    # Keep already normalized floating-point data unchanged. Integer images are
    # scaled by their dtype range; floating images above one are min-max scaled.
    if np.issubdtype(array.dtype, np.integer):
        dtype_max = float(np.iinfo(array.dtype).max)
        if dtype_max > 0:
            tensor = tensor / dtype_max
    elif tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
        minimum = tensor.min()
        maximum = tensor.max()
        if maximum > minimum:
            tensor = (tensor - minimum) / (maximum - minimum)

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
) -> None:
    """Create img.pt and mask.pt directly from NPY/NPZ source arrays."""
    image = load_array(image_path, loader, image_npz_key, "image")
    mask = load_array(mask_path, loader, mask_npz_key, "mask")

    image_tensor = _to_chw_image(image, image_path)
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
) -> Path:
    raw_path = dataset_path / split
    cache_path = dataset_path / f"{split}_cache_{input_size}"

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

        image_names = list(batch["image_name"])
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

    train_cache = ensure_cache(dataset_path, "Train", args.input_size, args.loader, args.image_npz_key, args.mask_npz_key)

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
            args.image_npz_key, args.mask_npz_key
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

    test_cache = ensure_cache(dataset_path, "Test", args.input_size, args.loader, args.image_npz_key, args.mask_npz_key)

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

    test_cache = ensure_cache(dataset_path, "Test", args.input_size, args.loader, args.image_npz_key, args.mask_npz_key)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate VersaMammo segmentation over patient-level folds."
    )
    parser.add_argument("--dataset-prefix", default="ZGT_VersaMammo")
    parser.add_argument("--folds", default="0-4", help="Comma/range list, e.g. 0-4 or 0,2,4.")
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
