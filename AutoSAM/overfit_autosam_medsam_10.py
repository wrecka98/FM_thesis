from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_autosam_medsam_folds import (
    CSVMammoAutoSAMDataset,
    MedSAMResizeLongestSide,
    ModelEmb,
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

    final_metrics, final_rows = evaluate(model, sam, eval_loader, device, args, run_dir)
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
