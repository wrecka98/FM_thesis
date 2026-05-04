from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union
import numpy as np
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
# Make the project root importable so `utils.ood_datasets` can be found.
current = Path.cwd().parent.parent.parent
for parent in [current] + list(current.parents):
    if (parent / "utils").exists():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("Could not find project root containing 'utils' folder")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.ood_datasets import (
    MRIDataset,
    ComposeImageTransforms,
    EnsureChannels,
    ResizeImageOrVolume,
)

PathLike = Union[str, Path]


class MedSAMMRIDataset(MRIDataset):
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
        output_mode: Literal["2d"] = "2d",
        slice_strategy: Literal["middle", "middle_3"] = "middle_3",
        slice_axis: int = 2,
        frame_selector: Union[str, int] = "middle",
        channels: int = 3,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        model_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        exclude_gt_suffixes: Sequence[str] = ("_gt.nii", "_gt.nii.gz"),
        gt_suffixes: Sequence[str] = ("_gt.nii", "_gt.nii.gz"),
        return_gt_mask: bool = True,
        return_bbox: bool = True,
        bbox_padding: int = 0,
        allow_empty_mask: bool = False,
        skip_empty_mask: bool = False,
    ) -> None:
        if output_mode != "2d":
            raise ValueError("MedSAMMRIDataset currently supports output_mode='2d' only.")

        self.gt_suffixes = tuple(gt_suffixes)
        self.return_gt_mask = bool(return_gt_mask)
        self.return_bbox = bool(return_bbox)
        self.bbox_padding = int(bbox_padding)
        self.allow_empty_mask = bool(allow_empty_mask)
        self.skip_empty_mask = bool(skip_empty_mask)

        super().__init__(
            root=root,
            file_paths=file_paths,
            label=label,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            limit=limit,
            return_metadata=return_metadata,
            output_mode=output_mode,
            slice_strategy=slice_strategy,
            slice_axis=slice_axis,
            frame_selector=frame_selector,
            resize_to_2d=None,   # important: no resizing in dataset
            resize_to_3d=None,   # important: no resizing in dataset
            channels=channels,
            transform=transform,
            model_transform=model_transform,
            exclude_gt_suffixes=exclude_gt_suffixes,
        )

    def _image_to_gt_path(self, image_path: Path) -> Path:
        s = str(image_path)
        if s.endswith(".nii.gz"):
            return Path(s[:-7] + "_gt.nii.gz")
        if s.endswith(".nii"):
            return Path(s[:-4] + "_gt.nii")
        raise ValueError(f"Unsupported image path: {image_path}")

    def _mask_slice_to_tensor_2d(self, mask2d: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(mask2d.astype(np.float32)).unsqueeze(0)

    def _compute_bbox_2d(self, mask2d: np.ndarray) -> Optional[torch.Tensor]:
        ys, xs = np.where(mask2d > 0)

        if len(xs) == 0 or len(ys) == 0:
            if self.allow_empty_mask:
                return None
            raise ValueError("Empty GT mask; cannot compute bounding box.")

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        if self.bbox_padding > 0:
            h, w = mask2d.shape
            x_min = max(0, x_min - self.bbox_padding)
            y_min = max(0, y_min - self.bbox_padding)
            x_max = min(w - 1, x_max + self.bbox_padding)
            y_max = min(h - 1, y_max + self.bbox_padding)

        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

    def _build_samples(self, files: Sequence[Path]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        offsets = [0] if self.slice_strategy == "middle" else [-1, 0, 1]

        for path in files:
            gt_path = self._image_to_gt_path(path)
            if not gt_path.exists():
                continue

            if self.skip_empty_mask:
                # Load once at index-building time and keep only informative slices
                vol = self._load_nifti(path)
                gt = self._load_nifti(gt_path)

                _, _ = self._select_3d_volume(vol)  # not needed further, just consistency
                gt3d, _ = self._select_3d_volume(gt)
                gt3d = (gt3d > 0).astype(np.uint8)

                n_slices = gt3d.shape[self.slice_axis]
                mid = n_slices // 2

                for offset in offsets:
                    slice_idx = int(np.clip(mid + offset, 0, n_slices - 1))
                    gt2d = self._extract_2d_slice(gt3d, slice_idx)
                    if np.any(gt2d > 0):
                        samples.append(
                            {
                                "path": path,
                                "gt_path": gt_path,
                                "label": self.label,
                                "sample_type": "slice",
                                "slice_offset": offset,
                            }
                        )
            else:
                for offset in offsets:
                    samples.append(
                        {
                            "path": path,
                            "gt_path": gt_path,
                            "label": self.label,
                            "sample_type": "slice",
                            "slice_offset": offset,
                        }
                    )

        return samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        path = sample["path"]
        gt_path = sample["gt_path"]

        vol = self._load_nifti(path)
        gt = self._load_nifti(gt_path)

        vol3d, frame_idx = self._select_3d_volume(vol)
        gt3d, gt_frame_idx = self._select_3d_volume(gt)

        # image path: MRI preprocessing
        vol3d = self._preprocess_volume(vol3d)

        # mask path: binarize only, no MRI normalization
        gt3d = (gt3d > 0).astype(np.float32)

        if vol3d.shape != gt3d.shape:
            raise ValueError(
                f"Image/GT shape mismatch for {path.name}: "
                f"{vol3d.shape} vs {gt3d.shape}"
            )

        n_slices = vol3d.shape[self.slice_axis]
        mid = n_slices // 2
        slice_offset = int(sample["slice_offset"])
        slice_idx = int(np.clip(mid + slice_offset, 0, n_slices - 1))

        img2d = self._extract_2d_slice(vol3d, slice_idx)
        gt2d = self._extract_2d_slice(gt3d, slice_idx)

        # MRI normalization only for image slice
        img2d = self._preprocess_slice(img2d)

        image = torch.from_numpy(img2d).float().unsqueeze(0)  # [1, H, W]
        if self.channels == 3:
            image = image.repeat(3, 1, 1)

        out = {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "path": str(path),
            "gt_path": str(gt_path),
        }

        if self.return_gt_mask:
            out["mask"] = self._mask_slice_to_tensor_2d(gt2d)

        if self.return_bbox:
            out["bbox"] = self._compute_bbox_2d(gt2d)

        if self.return_metadata:
            out["metadata"] = {
                "path": str(path),
                "gt_path": str(gt_path),
                "file_name": path.name,
                "original_shape": tuple(vol.shape),
                "used_shape": tuple(vol3d.shape),
                "frame_index": frame_idx,
                "gt_frame_index": gt_frame_idx,
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



class ComposeSampleTransforms:
    def __init__(self, transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    
    
class EnsureImageChannels:
    def __init__(self, out_channels: int) -> None:
        if out_channels not in {1, 3}:
            raise ValueError("out_channels must be 1 or 3")
        self.out_channels = out_channels

    def _convert(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {3, 4}:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

        c = x.shape[0]

        if self.out_channels == 1:
            if c == 1:
                return x
            if c == 3:
                return x.mean(dim=0, keepdim=True)
            raise ValueError(f"Cannot reduce {c} channels to 1.")

        if self.out_channels == 3:
            if c == 3:
                return x
            if c == 1:
                reps = [1] * x.ndim
                reps[0] = 3
                return x.repeat(*reps)
            raise ValueError(f"Cannot expand {c} channels to 3.")

        raise ValueError(f"Unsupported out_channels={self.out_channels}")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "image" not in sample:
            raise KeyError("Sample must contain 'image'")

        sample["image"] = self._convert(sample["image"])
        return sample    
    
       
    
class ResizeSample:
    def __init__(
        self,
        size: Union[tuple[int, int], tuple[int, int, int]],
        *,
        resize_image: bool = True,
        resize_mask: bool = True,
        resize_bbox: bool = True,
    ) -> None:
        self.size = size
        self.resize_image = resize_image
        self.resize_mask = resize_mask
        self.resize_bbox = resize_bbox

    def _resize_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if len(self.size) != 2:
                raise ValueError("2D image tensor requires size=(H, W)")
            return F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if x.ndim == 4:
            if len(self.size) != 3:
                raise ValueError("3D image tensor requires size=(D, H, W)")
            return F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        raise ValueError(f"Expected 3D or 4D image tensor, got shape {tuple(x.shape)}")

    def _resize_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if len(self.size) != 2:
                raise ValueError("2D mask tensor requires size=(H, W)")
            return F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="nearest",
            ).squeeze(0)

        if x.ndim == 4:
            if len(self.size) != 3:
                raise ValueError("3D mask tensor requires size=(D, H, W)")
            return F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="nearest",
            ).squeeze(0)

        raise ValueError(f"Expected 3D or 4D mask tensor, got shape {tuple(x.shape)}")

    def _resize_bbox_2d(
        self,
        bbox: torch.Tensor,
        old_hw: tuple[int, int],
        new_hw: tuple[int, int],
    ) -> torch.Tensor:
        old_h, old_w = old_hw
        new_h, new_w = new_hw

        scale_x = new_w / old_w
        scale_y = new_h / old_h

        x0, y0, x1, y1 = bbox.float()
        return torch.tensor(
            [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
            dtype=torch.float32,
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "image" not in sample:
            raise KeyError("Sample must contain 'image'")

        image = sample["image"]

        if image.ndim == 3:
            _, old_h, old_w = image.shape
            if len(self.size) != 2:
                raise ValueError("2D sample requires size=(H, W)")
            new_h, new_w = self.size
        elif image.ndim == 4:
            _, old_d, old_h, old_w = image.shape
            if len(self.size) != 3:
                raise ValueError("3D sample requires size=(D, H, W)")
            new_d, new_h, new_w = self.size
        else:
            raise ValueError(f"Expected 3D or 4D image tensor, got shape {tuple(image.shape)}")

        if self.resize_image:
            sample["image"] = self._resize_image(image)

        if self.resize_mask and "mask" in sample and sample["mask"] is not None:
            sample["mask"] = self._resize_mask(sample["mask"])

        if (
            self.resize_bbox
            and "bbox" in sample
            and sample["bbox"] is not None
            and image.ndim == 3
        ):
            sample["bbox"] = self._resize_bbox_2d(
                sample["bbox"],
                old_hw=(old_h, old_w),
                new_hw=(new_h, new_w),
            )

        sample["original_size"] = (old_h, old_w) if image.ndim == 3 else (old_d, old_h, old_w)
        sample["resized_size"] = self.size
        return sample
    
    
    
    
    