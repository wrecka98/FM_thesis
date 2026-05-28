from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_autosam_medsam_folds import (
    CSVMammoAutoSAMDataset,
    MedSAMResizeLongestSide,
    ModelEmb,
    autosam_forward,
    evaluate,
    freeze_sam,
    parse_values,
    sam_model_registry,
    save_csv,
    save_json,
    set_seed,
    train_one_epoch,
)


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DEFAULT_RESULTS_DIR = (
    REPO_ROOT
    / "pipelines and experiments"
    / "results"
    / "autosam_medsam_overfit_10"
)


class SyntheticCircleMaskDataset(CSVMammoAutoSAMDataset):
    def __init__(self, *args, circle_area_fraction: float = 0.30, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.circle_area_fraction = float(circle_area_fraction)

    def _make_circle_mask(self, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        target_area = self.circle_area_fraction * height * width
        radius = int(np.sqrt(target_area / np.pi))
        radius = max(1, min(radius, height // 2, width // 2))
        yy, xx = np.ogrid[:height, :width]
        center_y = height // 2
        center_x = width // 2
        return (((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2).astype(np.float32)

    def __getitem__(self, index: int):
        row = self.df.iloc[index % len(self.df)]
        image_path = self._resolve_path(folder="images_png", value=row[self.image_col])

        image = self._read_image(image_path)
        mask = self._make_circle_mask(image.shape[:2])
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


def select_overfit_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv_file)
    if df.empty:
        raise ValueError(f"No rows found in {args.csv_file}")

    if args.split_col in df.columns:
        split_values = parse_values(args.split_values)
        df = df[df[args.split_col].astype(str).str.lower().isin(split_values)].copy()

    if args.fold is not None:
        if args.fold_col not in df.columns:
            raise ValueError(f"Cannot filter by fold; column '{args.fold_col}' is missing.")
        df = df[df[args.fold_col].astype(str) == str(args.fold)].copy()

    if df.empty:
        raise ValueError("No rows left after split/fold filtering.")

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed)

    selected = df.head(args.num_images).copy().reset_index(drop=True)
    if len(selected) < args.num_images:
        print(f"Only found {len(selected)} rows after filtering; continuing with those.")
    return selected


def _display_image(image: torch.Tensor, height: int, width: int) -> np.ndarray:
    image_np = image.detach().cpu()[:, :height, :width].permute(1, 2, 0).numpy()
    min_value = float(image_np.min())
    max_value = float(image_np.max())
    image_np = (image_np - min_value) / (max_value - min_value + 1e-8)
    return np.clip(image_np * 255.0, 0, 255).astype(np.uint8)


def _display_mask(mask: torch.Tensor, height: int, width: int, threshold: float = 0.5) -> np.ndarray:
    mask_np = mask.detach().cpu().squeeze()[:height, :width].numpy()
    return (mask_np >= threshold).astype(np.uint8)


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> np.ndarray:
    blended = image.copy()
    color_np = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    blended[mask_bool] = ((1.0 - alpha) * blended[mask_bool].astype(np.float32) + alpha * color_np).astype(np.uint8)
    return blended


def _draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    outlined = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined, contours, -1, color, 2)
    return outlined


def _title_panel(panel: np.ndarray, title: str) -> np.ndarray:
    titled = panel.copy()
    cv2.rectangle(titled, (0, 0), (titled.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(titled, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return titled


@torch.no_grad()
def save_overlays(
    model: torch.nn.Module,
    sam: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    overlay_dir: Path,
    max_images: int,
) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    for images, masks, _, image_sizes, image_ids in loader:
        images = images.to(device)
        masks = masks.to(device)
        pred_low = autosam_forward(model, sam, images, args.adapter_size)
        preds = F.interpolate(pred_low, masks.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)

        for idx, image_id in enumerate(image_ids):
            valid_h = int(image_sizes[idx][0].item())
            valid_w = int(image_sizes[idx][1].item())
            image_np = _display_image(images[idx], valid_h, valid_w)
            gt_np = _display_mask(masks[idx], valid_h, valid_w)
            pred_np = _display_mask(preds[idx], valid_h, valid_w, args.threshold)

            gt_panel = _draw_mask_contour(_blend_mask(image_np, gt_np, (40, 220, 90)), gt_np, (0, 255, 0))
            pred_panel = _draw_mask_contour(_blend_mask(image_np, pred_np, (255, 70, 210)), pred_np, (255, 0, 255))
            both_panel = _blend_mask(image_np, gt_np, (40, 220, 90), alpha=0.36)
            both_panel = _blend_mask(both_panel, pred_np, (255, 70, 210), alpha=0.36)
            both_panel = _draw_mask_contour(both_panel, gt_np, (0, 255, 0))
            both_panel = _draw_mask_contour(both_panel, pred_np, (255, 0, 255))

            top = np.concatenate(
                [
                    _title_panel(image_np, "image"),
                    _title_panel(gt_panel, "gt mask"),
                ],
                axis=1,
            )
            bottom = np.concatenate(
                [
                    _title_panel(pred_panel, "prediction"),
                    _title_panel(both_panel, "gt green / pred magenta"),
                ],
                axis=1,
            )
            montage = np.concatenate([top, bottom], axis=0)
            safe_id = str(image_id).replace("/", "_")
            cv2.imwrite(str(overlay_dir / f"{safe_id}.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= max_images:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit the MedSAM-based AutoSAM dense-prompt adapter on a tiny mammography subset."
    )
    parser.add_argument("--csv-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-size", type=int, default=64)
    parser.add_argument("--order", type=int, default=85)
    parser.add_argument("--depth-wise", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--mask-col", default="mask_path")
    parser.add_argument("--id-col", default="unique_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split-values", default="training,train,trainval")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--fold", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--train-loop", type=int, default=1)
    parser.add_argument("--synthetic-circle-mask", action="store_true")
    parser.add_argument("--circle-area-fraction", type=float, default=0.30)

    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--experiment-name", default="overfit_10")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--save-overlays-every", type=int, default=0)
    parser.add_argument("--max-overlay-images", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = args.results_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_df = select_overfit_rows(args)
    selected_df.to_csv(run_dir / "selected_images.csv", index=False)
    print(f"Overfitting on {len(selected_df)} images. Selected rows saved to {run_dir / 'selected_images.csv'}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    freeze_sam(sam)
    sam_transform = MedSAMResizeLongestSide(sam.image_encoder.img_size)

    dataset_cls = SyntheticCircleMaskDataset if args.synthetic_circle_mask else CSVMammoAutoSAMDataset
    dataset_kwargs = {}
    if args.synthetic_circle_mask:
        dataset_kwargs["circle_area_fraction"] = args.circle_area_fraction
        print(f"Using centered synthetic circle masks with area fraction {args.circle_area_fraction:.2f}.")

    dataset = dataset_cls(
        selected_df,
        data_root=args.data_root,
        sam_transform=sam_transform,
        image_col=args.image_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
        augment=args.augment_train,
        loop=args.train_loop,
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model_args: Dict[str, int] = {
        "depth_wise": args.depth_wise,
        "order": args.order,
        "Idim": args.adapter_size,
    }
    model = ModelEmb(args=model_args).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_path = run_dir / "net_best_overfit.pth"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, sam, loader, optimizer, device, args, epoch)
        row: Dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, _ = evaluate(model, sam, eval_loader, device, args)
            row.update({f"memorized_{key}": float(value) for key, value in metrics.items()})
            print(
                f"epoch={epoch} loss={train_loss:.6f} "
                f"dice={metrics.get('dice', np.nan):.4f} iou={metrics.get('iou', np.nan):.4f}"
            )
        else:
            print(f"epoch={epoch} loss={train_loss:.6f}")

        history.append(row)
        save_csv(run_dir / "overfit_history.csv", history)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_path)

        if args.save_overlays_every > 0 and epoch % args.save_overlays_every == 0:
            save_overlays(
                model,
                sam,
                eval_loader,
                device,
                args,
                run_dir / "overlays" / f"epoch_{epoch:04d}",
                args.max_overlay_images,
            )

    final_metrics, final_rows = evaluate(model, sam, eval_loader, device, args, run_dir)
    if args.save_overlays:
        save_overlays(
            model,
            sam,
            eval_loader,
            device,
            args,
            run_dir / "overlays" / "final",
            args.max_overlay_images,
        )
    save_json(
        run_dir / "summary.json",
        {
            "num_images": len(selected_df),
            "epochs": args.epochs,
            "synthetic_circle_mask": bool(args.synthetic_circle_mask),
            "circle_area_fraction": float(args.circle_area_fraction),
            "best_train_loss": best_loss,
            "training_seconds": float(time.time() - start),
            "final_memorization_metrics": final_metrics,
            "best_checkpoint": str(best_path),
        },
    )
    save_csv(run_dir / "final_per_image_metrics.csv", final_rows)
    torch.save(model.state_dict(), run_dir / "net_last_overfit.pth")
    print(f"Done. Best loss={best_loss:.6f}. Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
