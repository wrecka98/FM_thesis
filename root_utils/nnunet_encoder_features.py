"""Utilities for extracting pooled bottleneck features from a trained nnU-Net v2."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Union

import numpy as np
import torch
import torch.nn.functional as F


PathLike = Union[str, Path]


def load_nnunet_encoder(
    model_folder: PathLike,
    device: Union[str, torch.device],
    fold: int = 0,
    checkpoint_name: str = "checkpoint_final.pth",
):
    """Load a trained nnU-Net v2 and retain only its encoder.

    nnU-Net records the exact architecture in the checkpoint and plans. Using its
    predictor is therefore safer than rebuilding a particular UNet class here.
    The decoder is detached immediately after loading and is never executed.
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder = Path(model_folder)
    checkpoint = model_folder / f"fold_{fold}" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"nnU-Net checkpoint not found: {checkpoint}. Finish training or set "
            "NNUNET_MODEL_FOLDER/FOLD/CHECKPOINT_NAME to an existing checkpoint."
        )

    device = torch.device(device)
    predictor = nnUNetPredictor(
        device=device,
        perform_everything_on_device=device.type == "cuda",
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=(fold,), checkpoint_name=checkpoint_name
    )

    encoder = predictor.network.encoder
    predictor.network.decoder = None  # release decoder parameters; it is not used
    encoder = encoder.to(device).eval()
    return encoder, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json


def _as_nnunet_array(image) -> np.ndarray:
    """Convert a grayscale/RGB image to nnU-Net NaturalImage2DIO layout (C, 1, H, W)."""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        if image.shape[-1] in (1, 3, 4):
            image = image[..., :3].mean(axis=-1)[None, None]
        elif image.shape[0] in (1, 3, 4):
            image = image[:3].mean(axis=0)[None, None]
        else:
            raise ValueError(f"Cannot identify the channel axis in image shape {image.shape}")
    else:
        raise ValueError(f"Expected a 2D image (optionally with channels), got {image.shape}")
    return image.astype(np.float32, copy=False)


def preprocess_nnunet_image(image, plans_manager, configuration_manager, dataset_json) -> torch.Tensor:
    """Apply the preprocessing recorded in the trained nnU-Net plans."""
    preprocessor = configuration_manager.preprocessor_class(verbose=False)
    preprocessor.show_progress_bar = False
    data, _, _ = preprocessor.run_case_npy(
        _as_nnunet_array(image),
        None,
        {"spacing": (999, 1, 1)},  # NaturalImage2DIO's spacing for PNG-like 2D data
        plans_manager,
        configuration_manager,
        dataset_json,
    )
    # 2D natural-image preprocessing retains a singleton pseudo-depth dimension.
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 3:
        raise ValueError(f"Expected preprocessed (C, H, W) data, got {data.shape}")
    return torch.from_numpy(np.ascontiguousarray(data)).float()


def _sliding_window_starts(image_size, patch_size, step_fraction: float):
    starts = []
    for image_dim, patch_dim in zip(image_size, patch_size):
        if image_dim <= patch_dim:
            starts.append([0])
            continue
        target_step = patch_dim * step_fraction
        num_steps = int(np.ceil((image_dim - patch_dim) / target_step)) + 1
        actual_step = (image_dim - patch_dim) / (num_steps - 1)
        starts.append([int(round(actual_step * i)) for i in range(num_steps)])
    return starts


@torch.inference_mode()
def extract_nnunet_bottleneck(
    image,
    encoder,
    plans_manager,
    configuration_manager,
    dataset_json,
    device: Union[str, torch.device],
    tile_step_size: float = 0.5,
) -> torch.Tensor:
    """Return one global-average-pooled deepest-encoder vector for an image.

    Images larger than the training patch are tiled exactly at the nnU-Net patch
    size. The pooled vectors of all tiles are averaged, yielding one fixed-size
    feature vector per image without invoking the decoder.
    """
    if not 0 < tile_step_size <= 1:
        raise ValueError("tile_step_size must be in (0, 1]")

    data = preprocess_nnunet_image(image, plans_manager, configuration_manager, dataset_json)
    patch_size = tuple(int(i) for i in configuration_manager.patch_size)
    pad_h = max(0, patch_size[0] - data.shape[-2])
    pad_w = max(0, patch_size[1] - data.shape[-1])
    data = F.pad(data, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

    starts = _sliding_window_starts(data.shape[-2:], patch_size, tile_step_size)
    pooled_tiles = []
    for row, col in product(*starts):
        tile = data[:, row : row + patch_size[0], col : col + patch_size[1]]
        skips = encoder(tile.unsqueeze(0).to(device))
        bottleneck = skips[-1] if isinstance(skips, (tuple, list)) else skips
        pooled_tiles.append(bottleneck.mean(dim=tuple(range(2, bottleneck.ndim))).cpu())
    return torch.stack(pooled_tiles).mean(dim=0)


def extract_nnunet_dataset_features(
    dataset: Iterable[Mapping],
    encoder,
    plans_manager,
    configuration_manager,
    dataset_json,
    device: Union[str, torch.device],
    tile_step_size: float = 0.5,
) -> torch.Tensor:
    """Extract an ``(N, C)`` bottleneck feature matrix from a BaseDataset-like iterable."""
    features = [
        extract_nnunet_bottleneck(
            sample["image"], encoder, plans_manager, configuration_manager,
            dataset_json, device, tile_step_size
        )
        for sample in dataset
    ]
    if not features:
        raise ValueError("Cannot extract nnU-Net features from an empty dataset")
    return torch.cat(features, dim=0)
