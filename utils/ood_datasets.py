
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode

PathLike = Union[str, Path]


class BaseIndexedDataset(Dataset):
    def __init__(
        self,
        root: Optional[PathLike] = None,
        file_paths: Optional[Sequence[PathLike]] = None,
        *,
        label: int = 0,
        split: Literal["train", "test", "all"] = "all",
        train_ratio: float = 0.7,
        seed: int = 42,
        limit: Optional[int] = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()

        if split not in {"train", "test", "all"}:
            raise ValueError(f"Unknown split: {split}")

        self.label = int(label)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.seed = int(seed)
        self.return_metadata = bool(return_metadata)

        files = self._collect_files(root=root, file_paths=file_paths)
        if len(files) == 0:
            raise ValueError("No files found.")

        rng = random.Random(self.seed)
        files = list(files)
        rng.shuffle(files)

        if limit is not None:
            files = files[:limit]

        n_train = int(round(len(files) * self.train_ratio))

        if self.split == "train":
            self.files = files[:n_train]
        elif self.split == "test":
            self.files = files[n_train:]
        else:
            self.files = files

        if len(self.files) == 0:
            raise ValueError(f"No files left after applying split='{self.split}'.")

        self.samples = self._build_samples(self.files)

        if len(self.samples) == 0:
            raise ValueError("No samples available after sample index construction.")

    def _collect_files(
        self,
        root: Optional[PathLike],
        file_paths: Optional[Sequence[PathLike]],
    ) -> list[Path]:
        if file_paths is not None:
            return [Path(p) for p in file_paths]

        if root is None:
            raise ValueError("Provide either `root` or `file_paths`.")

        root = Path(root)
        return sorted([p for p in root.rglob("*") if p.is_file()])

    def _build_samples(self, files: Sequence[Path]) -> list[dict[str, Any]]:
        return [{"path": p, "label": self.label} for p in files]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError


class VolumetricDataset(BaseIndexedDataset):
    VALID_EXTS = (".nii", ".nii.gz")

    def __init__(
        self,
        root: Optional[PathLike] = None,
        file_paths: Optional[Sequence[PathLike]] = None,
        *,
        label: int = 0,
        split: Literal["train", "test", "all"] = "all",
        train_ratio: float = 0.7,
        seed: int = 42,
        limit: Optional[int] = None,
        return_metadata: bool = False,
        output_mode: Literal["2d", "3d"] = "2d",
        slice_strategy: Literal["middle", "middle_3"] = "middle_3",
        slice_axis: int = 2,
        frame_selector: Union[str, int] = "middle",
        resize_to_2d: Optional[tuple[int, int]] = None,
        resize_to_3d: Optional[tuple[int, int, int]] = None,
        channels: int = 1,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        model_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        exclude_gt_suffixes: Sequence[str] = ("_gt.nii", "_gt.nii.gz"),
    ) -> None:
        self.output_mode = output_mode
        self.slice_strategy = slice_strategy
        self.slice_axis = int(slice_axis)
        self.frame_selector = frame_selector
        self.resize_to_2d = resize_to_2d
        self.resize_to_3d = resize_to_3d
        self.channels = int(channels)
        self.transform = transform
        self.model_transform = model_transform
        self.exclude_gt_suffixes = tuple(exclude_gt_suffixes)

        if self.output_mode not in {"2d", "3d"}:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        if self.slice_strategy not in {"middle", "middle_3"}:
            raise ValueError(f"Unsupported slice_strategy: {self.slice_strategy}")
        if self.channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")
        if self.output_mode == "3d" and self.channels != 1:
            raise ValueError("For 3D output, only channels=1 is supported.")

        super().__init__(
            root=root,
            file_paths=file_paths,
            label=label,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            limit=limit,
            return_metadata=return_metadata,
        )

    def _collect_files(
        self,
        root: Optional[PathLike],
        file_paths: Optional[Sequence[PathLike]],
    ) -> list[Path]:
        files = super()._collect_files(root=root, file_paths=file_paths)
        out: list[Path] = []
        for p in files:
            path_str = str(p)
            if not path_str.endswith(self.VALID_EXTS):
                continue
            if any(p.name.endswith(sfx) for sfx in self.exclude_gt_suffixes):
                continue
            out.append(p)
        return sorted(out)

    def _build_samples(self, files: Sequence[Path]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        if self.output_mode == "3d":
            for path in files:
                samples.append({"path": path, "label": self.label, "sample_type": "volume"})
            return samples

        offsets = [0] if self.slice_strategy == "middle" else [-1, 0, 1]
        for path in files:
            for offset in offsets:
                samples.append(
                    {
                        "path": path,
                        "label": self.label,
                        "sample_type": "slice",
                        "slice_offset": offset,
                    }
                )
        return samples

    def _load_nifti(self, path: Path) -> np.ndarray:
        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

    def _select_3d_volume(self, vol: np.ndarray) -> tuple[np.ndarray, Optional[int]]:
        if vol.ndim == 3:
            return vol, None
        if vol.ndim != 4:
            raise ValueError(f"Expected 3D or 4D volume, got shape {vol.shape}")
        n_frames = vol.shape[-1]
        if self.frame_selector == "middle":
            frame_idx = n_frames // 2
        elif isinstance(self.frame_selector, int):
            frame_idx = int(np.clip(self.frame_selector, 0, n_frames - 1))
        else:
            raise ValueError(f"Unsupported frame_selector: {self.frame_selector}")
        return np.asarray(vol[..., frame_idx], dtype=np.float32), frame_idx

    def _preprocess_volume(self, vol3d: np.ndarray) -> np.ndarray:
        return np.asarray(vol3d, dtype=np.float32)

    def _preprocess_slice(self, img2d: np.ndarray) -> np.ndarray:
        return np.asarray(img2d, dtype=np.float32)

    def _extract_2d_slice(self, vol3d: np.ndarray, slice_idx: int) -> np.ndarray:
        return np.asarray(np.take(vol3d, indices=slice_idx, axis=self.slice_axis), dtype=np.float32)

    def _volume_to_tensor_3d(self, vol3d: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(vol3d).float()
        if self.slice_axis != 0:
            x = torch.movedim(x, self.slice_axis, 0)
        x = x.unsqueeze(0)
        if self.resize_to_3d is not None:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.resize_to_3d,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return x

    def _slice_to_tensor_2d(self, img2d: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img2d).float().unsqueeze(0)
        if self.resize_to_2d is not None:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.resize_to_2d,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.channels == 3:
            x = x.repeat(3, 1, 1)
        return x

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        path = sample["path"]

        vol = self._load_nifti(path)
        vol3d, frame_idx = self._select_3d_volume(vol)
        vol3d = self._preprocess_volume(vol3d)

        if self.output_mode == "3d":
            image = self._volume_to_tensor_3d(vol3d)
            out = {
                "image": image,
                "label": torch.tensor(sample["label"], dtype=torch.long),
                "path": str(path),
            }
            if self.return_metadata:
                out["metadata"] = {
                    "path": str(path),
                    "file_name": path.name,
                    "original_shape": tuple(vol.shape),
                    "used_shape": tuple(vol3d.shape),
                    "frame_index": frame_idx,
                    "sample_type": "volume",
                    "modality": self.__class__.__name__,
                }
        else:
            n_slices = vol3d.shape[self.slice_axis]
            mid = n_slices // 2
            slice_offset = int(sample["slice_offset"])
            slice_idx = int(np.clip(mid + slice_offset, 0, n_slices - 1))
            img2d = self._extract_2d_slice(vol3d, slice_idx)
            img2d = self._preprocess_slice(img2d)
            image = self._slice_to_tensor_2d(img2d)
            out = {
                "image": image,
                "label": torch.tensor(sample["label"], dtype=torch.long),
                "path": str(path),
            }
            if self.return_metadata:
                out["metadata"] = {
                    "path": str(path),
                    "file_name": path.name,
                    "original_shape": tuple(vol.shape),
                    "used_shape": tuple(vol3d.shape),
                    "frame_index": frame_idx,
                    "sample_type": "slice",
                    "slice_axis": self.slice_axis,
                    "slice_index": slice_idx,
                    "slice_offset": slice_offset,
                    "modality": self.__class__.__name__,
                }

        if self.transform is not None:
            out = self.transform(out)
        if self.model_transform is not None:
            out["image"] = self.model_transform(out["image"])
        return out


class MRIDataset(VolumetricDataset):
    def _normalize_mri(self, arr: np.ndarray) -> np.ndarray:
        x = arr.astype(np.float32)
        mask = x != 0
        if mask.sum() == 0:
            return np.zeros_like(x, dtype=np.float32)
        vals = x[mask]
        lo = np.percentile(vals, 0.5)
        hi = np.percentile(vals, 99.5)
        if hi <= lo:
            out = np.zeros_like(x, dtype=np.float32)
            out[mask] = 1.0
            return out
        x = np.clip(x, lo, hi)
        x = (x - lo) / (hi - lo)
        x[~mask] = 0.0
        return x.astype(np.float32)

    def _preprocess_volume(self, vol3d: np.ndarray) -> np.ndarray:
        return self._normalize_mri(vol3d)

    def _preprocess_slice(self, img2d: np.ndarray) -> np.ndarray:
        return np.asarray(img2d, dtype=np.float32)


class CTDataset(VolumetricDataset):
    def __init__(
        self,
        *args,
        window_center: float = 40.0,
        window_width: float = 400.0,
        **kwargs,
    ) -> None:
        self.window_center = float(window_center)
        self.window_width = float(window_width)
        super().__init__(*args, **kwargs)

    def _window_ct(self, arr: np.ndarray) -> np.ndarray:
        x = arr.astype(np.float32)
        lo = self.window_center - self.window_width / 2.0
        hi = self.window_center + self.window_width / 2.0
        x = np.clip(x, lo, hi)
        x = (x - lo) / max(hi - lo, 1e-8)
        return x.astype(np.float32)

    def _preprocess_volume(self, vol3d: np.ndarray) -> np.ndarray:
        return self._window_ct(vol3d)

    def _preprocess_slice(self, img2d: np.ndarray) -> np.ndarray:
        return self._window_ct(img2d)


class EndoscopyDataset(BaseIndexedDataset):
    def __init__(
        self,
        root: Optional[PathLike] = None,
        file_paths: Optional[Sequence[PathLike]] = None,
        *,
        domains: Optional[Sequence[str]] = None,
        label: int = 0,
        split: Literal["train", "test", "all"] = "all",
        train_ratio: float = 0.7,
        seed: int = 42,
        limit: Optional[int] = None,
        return_metadata: bool = False,
        valid_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        image_mode: Literal["unchanged", "gray", "rgb"] = "rgb",
        resize_to: Optional[tuple[int, int]] = None,
        channels: int = 3,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        model_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.domains = list(domains) if domains is not None else None
        self.valid_exts = {e.lower() for e in valid_exts}
        self.image_mode = image_mode
        self.resize_to = resize_to
        self.channels = int(channels)
        self.transform = transform
        self.model_transform = model_transform

        if self.channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")

        super().__init__(
            root=root,
            file_paths=file_paths,
            label=label,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            limit=limit,
            return_metadata=return_metadata,
        )

    def _collect_files(self, root, file_paths) -> list[Path]:
        if file_paths is not None:
            files = [Path(p) for p in file_paths]
        else:
            if root is None:
                raise ValueError("Provide either `root` or `file_paths`.")
            root = Path(root)
            search_roots = [root] if self.domains is None else [root / d for d in self.domains]
            files = []
            for sroot in search_roots:
                if not sroot.exists():
                    raise FileNotFoundError(f"Directory not found: {sroot}")
                files.extend(
                    [
                        p for p in sroot.rglob("*")
                        if p.is_file()
                        and p.suffix.lower() in self.valid_exts
                        and not any(part.startswith(".") for part in p.parts)
                    ]
                )
        return sorted(files)

    def _build_samples(self, files: Sequence[Path]) -> list[dict[str, Any]]:
        return [
            {
                "path": path,
                "label": self.label,
                "image_id": path.stem,
                "domain": path.parent.name,
            }
            for path in files
        ]

    def _read_mode(self) -> ImageReadMode:
        if self.image_mode == "gray":
            return ImageReadMode.GRAY
        if self.image_mode == "rgb":
            return ImageReadMode.RGB
        return ImageReadMode.UNCHANGED

    def _postprocess_channels(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image tensor, got {tuple(image.shape)}")
        c = image.shape[0]
        if c == 4:
            image = image[:3]
            c = 3
        if self.channels == 1:
            if c == 3:
                image = image.mean(dim=0, keepdim=True)
            elif c != 1:
                raise ValueError(f"Cannot convert image with {c} channels to 1")
        else:
            if c == 1:
                image = image.repeat(3, 1, 1)
            elif c != 3:
                raise ValueError(f"Cannot convert image with {c} channels to 3")
        return image

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        path = sample["path"]
        image = decode_image(str(path), mode=self._read_mode()).float() / 255.0
        image = self._postprocess_channels(image)
        if self.resize_to is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.resize_to,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        out = {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "path": str(path),
        }
        if self.return_metadata:
            out["metadata"] = {
                "path": str(path),
                "file_name": path.name,
                "image_id": sample["image_id"],
                "domain": sample["domain"],
                "modality": self.__class__.__name__,
            }
        if self.transform is not None:
            out = self.transform(out)
        if self.model_transform is not None:
            out["image"] = self.model_transform(out["image"])
        return out


class ComposeImageTransforms:
    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class EnsureChannels:
    def __init__(self, out_channels: int) -> None:
        if out_channels not in {1, 3}:
            raise ValueError("out_channels must be 1 or 3")
        self.out_channels = out_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {3, 4}:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")
        c = x.shape[0]
        if self.out_channels == 1:
            if c == 1:
                return x
            if c == 3:
                return x.mean(dim=0, keepdim=True)
            raise ValueError(f"Cannot reduce {c} channels to 1.")
        if c == 3:
            return x
        if c == 1:
            reps = [1] * x.ndim
            reps[0] = 3
            return x.repeat(*reps)
        raise ValueError(f"Cannot expand {c} channels to 3.")


class ResizeImageOrVolume:
    def __init__(self, size: Union[tuple[int, int], tuple[int, int, int]]) -> None:
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if len(self.size) != 2:
                raise ValueError("2D tensor requires size=(H, W)")
            return F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if x.ndim == 4:
            if len(self.size) != 3:
                raise ValueError("3D tensor requires size=(D, H, W)")
            return F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")


class NormalizePerChannel:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {3, 4}:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")
        c = x.shape[0]
        if len(self.mean) != c or len(self.std) != c:
            raise ValueError(f"Stats length mismatch: got {len(self.mean)} for {c} channels")
        shape = [c] + [1] * (x.ndim - 1)
        mean = self.mean.view(*shape).to(device=x.device, dtype=x.dtype)
        std = self.std.view(*shape).to(device=x.device, dtype=x.dtype)
        return (x - mean) / std.clamp_min(1e-8)
