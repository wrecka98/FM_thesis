from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
from torch.utils.data import DataLoader, Dataset

from UNetEfficientNetB5 import UNetEfficientNetB5


DEFAULT_PATH_ROOT = Path("/mnt/data/spathak")
MODEL_NAME = "VersaMammo (Enb5)"


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_path(path_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else path_root / path


def image_to_tensor(array: np.ndarray, path: Path) -> torch.Tensor:
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported PNG image shape for {path}: {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()


def mask_to_tensor(array: np.ndarray, path: Path) -> torch.Tensor:
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported PNG mask shape for {path}: {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).float()


def scale_image(image: torch.Tensor, mode: str) -> torch.Tensor:
    minimum = float(image.min())
    maximum = float(image.max())
    if mode == "255":
        image = image / 255.0
    elif mode == "unit":
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"--image-scale unit expects [0,1], received [{minimum},{maximum}]"
            )
    elif mode == "minmax":
        image = (image - minimum) / (maximum - minimum) if maximum > minimum else image * 0
    else:
        if minimum >= 0.0 and maximum <= 1.0:
            pass
        elif minimum >= 0.0 and maximum <= 255.0:
            image = image / 255.0
        else:
            image = (
                (image - minimum) / (maximum - minimum) if maximum > minimum else image * 0
            )
    return (image - 0.5) / 0.5


class PngDataframeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        path_root: Path,
        input_size: int,
        image_scale: str,
        mask_threshold: float,
        augment: bool,
    ) -> None:
        self.dataframe = dataframe.reset_index().rename(columns={"index": "source_index"})
        self.path_root = path_root
        self.input_size = input_size
        self.image_scale = image_scale
        self.mask_threshold = mask_threshold
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.dataframe.iloc[index]
        image_path = resolve_data_path(self.path_root, row["ImagePath"])
        mask_path = resolve_data_path(self.path_root, row["ROIPath"])

        image = image_to_tensor(io.imread(image_path), image_path)
        mask = mask_to_tensor(io.imread(mask_path), mask_path)
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"Image/mask size mismatch at dataframe row {row['source_index']}: "
                f"{image_path}={tuple(image.shape[-2:])}, "
                f"{mask_path}={tuple(mask.shape[-2:])}"
            )

        image = scale_image(image, self.image_scale)
        mask = (mask > self.mask_threshold).float()
        output_size = (self.input_size, self.input_size)
        image = F.interpolate(
            image[None], output_size, mode="bilinear", align_corners=False
        )[0]
        mask = F.interpolate(mask[None], output_size, mode="nearest")[0]

        if self.augment:
            if random.random() >= 0.5:
                image, mask = torch.flip(image, [1]), torch.flip(mask, [1])
            if random.random() >= 0.5:
                image, mask = torch.flip(image, [2]), torch.flip(mask, [2])

        return {
            "images": image,
            "masks": mask,
            "row_number": index,
            "source_index": str(row["source_index"]),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def validate_dataframe(dataframe: pd.DataFrame, path_root: Path) -> None:
    required_columns = {"ImagePath", "ROIPath"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing dataframe columns: {sorted(missing_columns)}")
    if dataframe.empty:
        raise ValueError("The training dataframe is empty.")
    if dataframe[["ImagePath", "ROIPath"]].isnull().any().any():
        raise ValueError("ImagePath and ROIPath must not contain null values.")

    missing_paths: List[str] = []
    wrong_extensions: List[str] = []
    for column in ("ImagePath", "ROIPath"):
        for value in dataframe[column]:
            path = resolve_data_path(path_root, value)
            if path.suffix.lower() != ".png":
                wrong_extensions.append(str(path))
            elif not path.is_file():
                missing_paths.append(str(path))
            if len(missing_paths) >= 10 or len(wrong_extensions) >= 10:
                break
    if wrong_extensions:
        raise ValueError(
            "All ImagePath and ROIPath entries must reference PNG files "
            "(showing up to 10):\n" + "\n".join(wrong_extensions[:10])
        )
    if missing_paths:
        raise FileNotFoundError(
            "Missing input files (showing up to 10):\n" + "\n".join(missing_paths[:10])
        )


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("The pretrained checkpoint must contain a state dictionary.")
    for wrapper_key in ("state_dict", "model_state_dict", "model", "net"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, dict):
            checkpoint = wrapped
            break
    return {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def build_model(args: argparse.Namespace) -> nn.Module:
    # Avoid filename-based routing and network downloads in UNetEfficientNetB5.
    model = UNetEfficientNetB5(checkpoint_path=None, pretrained=False)
    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)

    backbone_state = {
        key.removeprefix("image_encoder."): value
        for key, value in state_dict.items()
        if key.startswith("image_encoder.")
    }
    if backbone_state:
        incompatible = model.backbone.load_state_dict(backbone_state, strict=False)
        if len(backbone_state) == len(incompatible.unexpected_keys):
            raise ValueError("No VersaMammo image-encoder weights matched the EfficientNet-B5 backbone.")
    else:
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        if not compatible:
            raise ValueError(
                "The checkpoint contains neither VersaMammo image_encoder weights nor "
                "weights compatible with UNetEfficientNetB5."
            )
        model.load_state_dict(compatible, strict=False)

    train_full_model = args.finetune in {"full", "fft"}
    for name, parameter in model.named_parameters():
        parameter.requires_grad = train_full_model or not (
            name.startswith("backbone.") or name.startswith("encoder")
        )
    return model


def segmentation_loss(
    predictions: torch.Tensor, targets: torch.Tensor, dice_loss: DiceLoss
) -> torch.Tensor:
    return F.binary_cross_entropy(predictions, targets) + dice_loss(predictions, targets)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VersaMammo segmentation on every row in a pickled dataframe."
    )
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument(
        "--image-scale",
        choices=("auto", "255", "unit", "minmax"),
        default="auto",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--finetune", choices=("head", "full", "lp", "fft"), default="head")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.dataframe.is_file():
        raise FileNotFoundError(f"Dataframe not found: {args.dataframe}")
    if not args.pretrained_checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
    if args.epochs < 1 or args.max_iter < 1:
        raise ValueError("--epochs and --max-iter must be positive.")
    if args.batch_size < 1 or args.log_every < 1:
        raise ValueError("--batch-size and --log-every must be positive.")

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
        augment=not args.no_augmentation,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = build_model(args).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    dice_loss = DiceLoss()
    history: List[Dict[str, object]] = []
    iteration = 0
    start_time = time.time()

    print(f"Training on all {len(dataset)} dataframe rows using {device}.")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch in loader:
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
                print(f"epoch={epoch + 1} iter={iteration} loss={loss.item():.5f}")
            if iteration >= args.max_iter:
                break

        epoch_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "last_iteration": iteration,
                "mean_training_loss": epoch_loss,
            }
        )
        write_csv(args.output_dir / "training_history.csv", history)
        print(f"Epoch {epoch + 1}: mean training loss={epoch_loss:.5f}")
        if args.save_every_epoch:
            torch.save(model.state_dict(), args.output_dir / f"checkpoint_epoch_{epoch + 1}.pth")
        if iteration >= args.max_iter:
            break

    checkpoint_path = args.output_dir / f"{MODEL_NAME}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    run_summary = {
        "dataframe": str(args.dataframe.resolve()),
        "path_root": str(args.path_root.resolve()),
        "pretrained_checkpoint": str(args.pretrained_checkpoint.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "samples": len(dataset),
        "epochs_completed": len(history),
        "iterations_completed": iteration,
        "final_mean_training_loss": history[-1]["mean_training_loss"],
        "training_seconds": time.time() - start_time,
        "device_used": str(device),
        "finetune": args.finetune,
        "augmentation": not args.no_augmentation,
    }
    with (args.output_dir / "training_summary.json").open("w") as handle:
        json.dump(run_summary, handle, indent=2)
    print(f"Saved trained model to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
