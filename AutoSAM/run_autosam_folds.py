from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.model_single import ModelEmb
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "pipelines and experiments" / "results" / "autosam_segmentation"


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
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CSVMammoAutoSAMDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_root: Path,
        sam_transform: ResizeLongestSide,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        id_col: str = "unique_id",
        augment: bool = False,
        loop: int = 1,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.sam_transform = sam_transform
        self.image_col = image_col
        self.mask_col = mask_col
        self.id_col = id_col
        self.augment = augment
        self.loop = max(1, int(loop))

    def __len__(self) -> int:
        return len(self.df) * self.loop


    def _resolve_path(self, folder, value):

        return self.data_root / folder / value

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[2] > 3:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return image

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        return (mask > 0).astype(np.float32)

    def _augment_pair(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))
        if random.random() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=0))
            mask = np.ascontiguousarray(np.flip(mask, axis=0))
        return image, mask

    def __getitem__(self, index: int):
        row = self.df.iloc[index % len(self.df)]
        image_path = self._resolve_path(folder="images_png",value=row[self.image_col])
        mask_path = self._resolve_path(folder="masks",value=row[self.mask_col])

        image = self._read_image(image_path)
        mask = self._read_mask(mask_path)
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        original_size = torch.tensor(image.shape[:2], dtype=torch.float32)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).float()

        image_t = self.sam_transform.apply_image_torch(image_t)
        mask_t = self.sam_transform.apply_image_torch(mask_t)
        mask_t = (mask_t > 0.5).float()
        image_size = torch.tensor(image_t.shape[-2:], dtype=torch.float32)

        image_t = self.sam_transform.preprocess(image_t)
        mask_t = self.sam_transform.preprocess(mask_t)

        image_id = str(row[self.id_col]) if self.id_col in row.index else image_path.stem
        return image_t, mask_t, original_size, image_size, image_id


def load_fold_csv(args: argparse.Namespace, fold: int) -> pd.DataFrame:
    csv_path = args.csv_file or Path(str(args.csv_template).format(fold=fold))
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found for fold {fold} in {csv_path}")
    return df.reset_index(drop=True)


def split_fold_dataframe(
    df: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_col = args.split_col
    train_values = parse_values(args.train_split_values)
    val_values = parse_values(args.val_split_values)
    test_values = parse_values(args.test_split_values)

    split_series = df[split_col].astype(str).str.lower() if split_col in df.columns else pd.Series([""] * len(df))
    fold_series = df[args.fold_col].astype(str) if args.fold_col in df.columns else None
    train_mask = split_series.isin(train_values)
    val_mask = split_series.isin(val_values)
    test_mask = split_series.isin(test_values)

    if fold_series is not None:
        cv_mask = split_series.isin(train_values) & (fold_series == str(args.current_fold))
        if cv_mask.any():
            val_mask = val_mask | cv_mask
            train_mask = split_series.isin(train_values) & (fold_series != str(args.current_fold))

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found using split values: {args.train_split_values}")
    if test_df.empty:
        raise ValueError(f"No test rows found using split values: {args.test_split_values}")

    if val_df.empty and args.val_fraction > 0:
        rng = np.random.default_rng(seed)
        group_col = args.group_col if args.group_col in train_df.columns else args.id_col
        groups = np.array(sorted(train_df[group_col].astype(str).unique()))
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * args.val_fraction)))
        val_groups = set(groups[:n_val])
        val_df = train_df[train_df[group_col].astype(str).isin(val_groups)].copy()
        train_df = train_df[~train_df[group_col].astype(str).isin(val_groups)].copy()

    if val_df.empty:
        print("Validation split is empty; using the test split for model selection.")
        val_df = test_df.copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    tp = torch.sum(target * pred, dim=(1, 2, 3))
    fn = torch.sum(target * (1 - pred), dim=(1, 2, 3))
    fp = torch.sum((1 - target) * pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + 0.5 * fn + 0.5 * fp + smooth)
    return 1 - torch.mean(tversky_class)


def norm_batch(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    flat = x.view(b, -1)
    min_value = flat.min(dim=1)[0].view(b, 1, 1, 1)
    max_value = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (x - min_value) / (max_value - min_value + 1e-6)


def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> List[Dict[str, float]]:
    rows = []
    pred = pred.detach().cpu().bool()
    target = target.detach().cpu().bool()
    for pred_i, target_i in zip(pred, target):
        pred_flat = pred_i.reshape(-1)
        target_flat = target_i.reshape(-1)
        tp = torch.logical_and(pred_flat, target_flat).sum().item()
        tn = torch.logical_and(~pred_flat, ~target_flat).sum().item()
        fp = torch.logical_and(pred_flat, ~target_flat).sum().item()
        fn = torch.logical_and(~pred_flat, target_flat).sum().item()
        dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0
        iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
        sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 1.0
        rows.append(
            {
                "dice": float(dice),
                "iou": float(iou),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "accuracy": float(accuracy),
            }
        )
    return rows


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [key for key in rows[0].keys() if key not in {"image_id", "fold"}]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def freeze_sam(sam: nn.Module) -> None:
    sam.eval()
    for param in sam.parameters():
        param.requires_grad = False


def autosam_forward(
    model: nn.Module,
    sam: nn.Module,
    images: torch.Tensor,
    adapter_size: int,
) -> torch.Tensor:
    images_small = F.interpolate(images, (adapter_size, adapter_size), mode="bilinear", align_corners=True)
    dense_embeddings = model(images_small)
    with torch.no_grad():
        image_embeddings = sam.image_encoder(images)
        image_pe = sam.prompt_encoder.get_dense_pe()

    low_res_masks = []
    for idx in range(images.shape[0]):
        with torch.no_grad():
            sparse_embeddings, _ = sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_mask, _ = sam.mask_decoder(
            image_embeddings=image_embeddings[idx : idx + 1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings[idx : idx + 1],
            multimask_output=False,
        )
        low_res_masks.append(low_res_mask)
    return norm_batch(torch.cat(low_res_masks, dim=0))


def train_one_epoch(
    model: nn.Module,
    sam: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> float:
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"train epoch {epoch}")
    for images, masks, _, _, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        pred_low = autosam_forward(model, sam, images, args.adapter_size)
        target_low = F.interpolate(masks.unsqueeze(1), pred_low.shape[-2:], mode="nearest").float()
        loss = F.binary_cross_entropy(pred_low, target_low) + dice_loss(pred_low, target_low)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        pbar.set_postfix(loss=np.mean(losses))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sam: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Optional[Path] = None,
    fold: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    model.eval()
    rows = []
    prediction_dir = output_dir / "predictions" if output_dir and args.save_predictions else None
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for images, masks, _, _, image_ids in tqdm(loader, desc="eval"):
        images = images.to(device)
        masks = masks.to(device)
        pred_low = autosam_forward(model, sam, images, args.adapter_size)
        pred = F.interpolate(pred_low, masks.shape[-2:], mode="bilinear", align_corners=True)
        pred_bin = (pred.squeeze(1) >= args.threshold).float()
        metric_rows = binary_metrics(pred_bin, masks)

        for image_id, metric_row, pred_mask in zip(image_ids, metric_rows, pred_bin):
            row = {"image_id": str(image_id), **metric_row}
            if fold is not None:
                row["fold"] = float(fold)
            rows.append(row)
            if prediction_dir:
                safe_id = str(image_id).replace("/", "_")
                pred_np = pred_mask.detach().cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(str(prediction_dir / f"{safe_id}.png"), pred_np)

    metrics = summarize(rows)
    return metrics, rows


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sam_transform: ResizeLongestSide,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = CSVMammoAutoSAMDataset(
        train_df,
        data_root=args.data_root,
        sam_transform=sam_transform,
        image_col=args.image_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
        augment=args.augment_train,
        loop=args.train_loop,
    )
    val_dataset = CSVMammoAutoSAMDataset(
        val_df,
        data_root=args.data_root,
        sam_transform=sam_transform,
        image_col=args.image_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
        augment=False,
    )
    test_dataset = CSVMammoAutoSAMDataset(
        test_df,
        data_root=args.data_root,
        sam_transform=sam_transform,
        image_col=args.image_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
    )
    return train_loader, val_loader, test_loader


def train_one_fold(
    fold: int,
    sam: nn.Module,
    sam_transform: ResizeLongestSide,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict:
    fold_dir = args.results_dir / args.experiment_name / f"fold{fold}"
    checkpoint_path = fold_dir / "net_best.pth"
    args.current_fold = fold
    df = load_fold_csv(args, fold)
    train_df, val_df, test_df = split_fold_dataframe(df, args, args.seed + fold)
    print(f"Fold {fold}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_loader, val_loader, test_loader = build_loaders(train_df, val_df, test_df, sam_transform, args)
    model_args = {
        "depth_wise": args.depth_wise,
        "order": args.order,
        "Idim": args.adapter_size,
    }
    model = ModelEmb(args=model_args).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_metrics: Dict[str, float] = {}
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, sam, train_loader, optimizer, device, args, epoch)
        val_metrics, _ = evaluate(model, sam, val_loader, device, args)
        selection_metric = val_metrics[args.selection_metric]
        history_row = {"fold": float(fold), "epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(history_row)
        save_csv(fold_dir / "validation_history.csv", history)

        print(
            f"Fold {fold} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_{args.selection_metric}={selection_metric:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_metrics = val_metrics
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            save_json(fold_dir / "best_validation_metrics.json", val_metrics)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    test_metrics, test_rows = evaluate(model, sam, test_loader, device, args, fold_dir, fold)
    save_json(fold_dir / "test_metrics.json", test_metrics)
    save_csv(fold_dir / "test_per_image_metrics.csv", test_rows)

    result = {
        "fold": float(fold),
        "checkpoint": str(checkpoint_path),
        "training_seconds": float(time.time() - start),
        **{f"val_{key}": value for key, value in best_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(fold_dir / "fold_result.json", result)
    return result


def aggregate_results(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    keys = [key for key in results[0].keys() if key.startswith("val_") or key.startswith("test_")]
    aggregate = {}
    for key in keys:
        values = np.array([float(result[key]) for result in results], dtype=np.float32)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate AutoSAM adapter over CSV folds.")
    parser.add_argument("--csv-template", default=None, help="CSV template, e.g. '../data csv formats/detection_mammofm_fold{fold}.csv'.")
    parser.add_argument("--csv-file", type=Path, default=None, help="Optional single CSV containing all folds.")
    parser.add_argument("--folds", default="0-4")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--mask-col", default="mask_path")
    parser.add_argument("--id-col", default="unique_id")
    parser.add_argument("--group-col", default="patient_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--train-split-values", default="training,train,trainval")
    parser.add_argument("--val-split-values", default="validation,valid,val,eval")
    parser.add_argument("--test-split-values", default="test")
    parser.add_argument("--val-fraction", type=float, default=0.15)

    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    parser.add_argument("--adapter-size", type=int, default=64)
    parser.add_argument("--order", type=int, default=85)
    parser.add_argument("--depth-wise", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--batch-size-eval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-workers-eval", type=int, default=0)
    parser.add_argument("--train-loop", type=int, default=1)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="dice")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--experiment-name", default="ZGT_AutoSAM")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv_template is None and args.csv_file is None:
        raise ValueError("Provide either --csv-template or --csv-file.")
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    freeze_sam(sam)
    sam_transform = ResizeLongestSide(sam.image_encoder.img_size)

    results = []
    for fold in parse_folds(args.folds):
        print(f"=== Fold {fold} ===")
        results.append(train_one_fold(fold, sam, sam_transform, device, args))

    experiment_dir = args.results_dir / args.experiment_name
    save_csv(experiment_dir / "fold_results.csv", results)
    aggregate = aggregate_results(results)
    save_json(experiment_dir / "aggregate_metrics.json", aggregate)

    print("Aggregate test metrics:")
    for metric in ["test_dice", "test_iou", "test_sensitivity", "test_precision"]:
        if metric in aggregate:
            values = aggregate[metric]
            print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
