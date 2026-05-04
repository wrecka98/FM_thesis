from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib


class MRIMiddleSlicesDataset(Dataset):
    """
    Load MRI .nii / .nii.gz volumes and return only the middle slice
    and its 2 adjacent slices as 2D images.

    For each volume, this dataset yields 3 samples:
        middle-1, middle, middle+1

    Supports:
    - 3D volumes: (H, W, D) or similar
    - 4D volumes: selects one time/frame first, then extracts slices

    Parameters
    ----------
    root : str or Path
        Root folder containing .nii or .nii.gz files.
    file_paths : optional sequence of paths
        Explicit list of files. If provided, `root` is ignored for scanning.
    slice_axis : int
        Axis along which to slice after reducing to 3D.
        Usually 2 for volumes shaped (H, W, D).
    time_index : str or int
        How to choose the frame if the input is 4D.
        - "middle": use middle time/frame
        - int: use that frame index
    resize_to : optional tuple[int, int]
        If given, resize each 2D slice to (H, W).
    repeat_channels : int
        Number of channels in output.
        - 1 => shape [1, H, W]
        - 3 => shape [3, H, W]
    transform : callable, optional
        Applied after tensor conversion.
    return_metadata : bool
        If True, returns a dict with file path, slice index, etc.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        file_paths: Optional[Sequence[Union[str, Path]]] = None,
        slice_axis: int = 2,
        time_index: Union[str, int] = "middle",
        resize_to: Optional[tuple[int, int]] = None,
        repeat_channels: int = 1,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        label=0
    ):
        super().__init__()

        if file_paths is not None:
            self.files = [Path(p) for p in file_paths]
        else:
            if root is None:
                raise ValueError("Provide either `root` or `file_paths`.")
            root = Path(root)
            self.files = sorted(list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))

        if len(self.files) == 0:
            raise ValueError("No .nii or .nii.gz files found.")

        self.slice_axis = slice_axis
        self.time_index = time_index
        self.resize_to = resize_to
        self.repeat_channels = repeat_channels
        self.transform = transform
        self.return_metadata = return_metadata
        self.label=label

        if repeat_channels not in (1, 3):
            raise ValueError("repeat_channels must be 1 or 3.")

        # Each volume contributes exactly 3 slices
        self.index_map = []
        for file_idx, path in enumerate(self.files):
            # We do not read the whole image here; just remember 3 logical slice slots.
            # slice_offset in {-1, 0, +1}
            for slice_offset in (-1, 0, 1):
                self.index_map.append((file_idx, slice_offset))

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_nifti(self, path: Path) -> np.ndarray:
        vol = nib.load(str(path)).get_fdata()
        return np.asarray(vol, dtype=np.float32)

    def _reduce_to_3d(self, vol: np.ndarray) -> np.ndarray:
        """
        Reduce volume to 3D.
        - If 3D: return as is
        - If 4D: select one time/frame
        """
        if vol.ndim == 3:
            return vol

        if vol.ndim == 4:
            if self.time_index == "middle":
                t = vol.shape[-1] // 2
            elif isinstance(self.time_index, int):
                t = self.time_index
                if not (0 <= t < vol.shape[-1]):
                    raise IndexError(f"time_index={t} out of range for shape {vol.shape}")
            else:
                raise ValueError("time_index must be 'middle' or an integer")

            vol3d = vol[..., t]
            return np.asarray(vol3d, dtype=np.float32)

        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {vol.shape}")

    def _normalize_mri_slice(self, img2d: np.ndarray) -> np.ndarray:
        """
        MRI-style normalization:
        - compute percentiles on nonzero voxels
        - clip to [0.5, 99.5] percentiles
        - scale to [0, 1]
        - keep background at 0
        """
        img = img2d.astype(np.float32)
        mask = img != 0

        if mask.sum() == 0:
            return np.zeros_like(img, dtype=np.float32)

        vals = img[mask]
        lo = np.percentile(vals, 0.5)
        hi = np.percentile(vals, 99.5)

        if hi <= lo:
            out = np.zeros_like(img, dtype=np.float32)
            out[mask] = 1.0
            return out

        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo)
        img[~mask] = 0.0
        return img.astype(np.float32)

    def _extract_slice(self, vol3d: np.ndarray, slice_offset: int) -> tuple[np.ndarray, int]:
        n_slices = vol3d.shape[self.slice_axis]
        mid = n_slices // 2
        idx = np.clip(mid + slice_offset, 0, n_slices - 1)

        img2d = np.take(vol3d, indices=idx, axis=self.slice_axis)
        return np.asarray(img2d, dtype=np.float32), int(idx)

    def _to_tensor(self, img2d: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img2d).float().unsqueeze(0)  # [1, H, W]

        if self.resize_to is not None:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.resize_to,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.repeat_channels == 3:
            x = x.repeat(3, 1, 1)

        return x

    def __getitem__(self, idx: int):
        file_idx, slice_offset = self.index_map[idx]
        path = self.files[file_idx]

        vol = self._load_nifti(path)
        vol3d = self._reduce_to_3d(vol)

        img2d, slice_idx = self._extract_slice(vol3d, slice_offset)
        img2d = self._normalize_mri_slice(img2d)
        x = self._to_tensor(img2d)

        if self.transform is not None:
            x = self.transform(x)

        if self.return_metadata:
            meta = {
                "path": str(path),
                "file_name": path.name,
                "label":label,
                "slice_offset": slice_offset,
                "slice_index": slice_idx,
                "original_shape": tuple(vol.shape),
                "used_shape": tuple(vol3d.shape),
            }
            return x, self.label, meta

        return x,self.label