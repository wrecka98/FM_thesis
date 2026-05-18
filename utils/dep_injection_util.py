from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union, List, Dict
import random
from PIL import Image

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from monai.transforms import Transform as T
from scipy.ndimage import binary_dilation, label as connected_components

PathLike = Union[str, Path]

SampleDict = Dict[str, Any]
FileCollector = Callable[..., List[Dict[str, Any]]]



class BaseDataset(Dataset):
    def __init__(
        self,
        root: Optional[PathLike] = None,
        file_paths: Optional[Sequence[PathLike]] = None,
        *,
        file_collector: Optional[FileCollector] = None,
        collector_kwargs: Optional[dict[str, Any]] = None,
        label: int = 0,
        split: Literal["train", "test", "all"] = "all",
        train_ratio: float = 0.7,
        seed: int = 42,
        limit: Optional[int] = None,
        format_loader = None,            # a list could be passed as valid formats and if-then statements would handle the rest. Or just pass standardized format such as .npz
        transforms= None,                # monai transforms composer which would apply the transforms to images and masks
        bbox_generator=None,             # a bbox generator class
        return_metadata: bool = False,
    ) -> None:
        super().__init__()

        
        

        if split not in {"train", "test", "all"}:
            raise ValueError(f"Unknown split: {split}")

        self.label = int(label)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.seed = int(seed)
        self.format_loader=format_loader
        self.transforms=transforms
        self.bbox_generator=bbox_generator
        self.return_metadata = bool(return_metadata)


        self.file_collector = file_collector or self._collect_files
        self.collector_kwargs = collector_kwargs or {}

        files = self.file_collector(
            root=root,
            **self.collector_kwargs,
        )


        if len(files) == 0:
            raise ValueError("No files found.")

        rng = random.Random(self.seed)
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

        for sample in self.files:
            sample["label"] = self.label

            
    def _collect_files(
    self,
    root: PathLike,
    ) -> List[Dict[str, Path]]:
        root = Path(root)

        if not root.exists():
            raise ValueError(f"Root path does not exist: {root}")

        patients = []

        for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            image_file = None
            mask_file = None

            for file in patient_dir.rglob(f"*.npz"):
                name = file.name.lower()
                if "image" in name:
                    image_file = file
                elif "mask" in name:
                    mask_file = file

            if image_file is None or mask_file is None:
                raise ValueError(
                    f"Missing image or mask in {patient_dir}"
                )

            patients.append({
                "image_path": image_file,
                "mask_path": mask_file
            })

        return patients


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        
        
        sample = self.files[idx]
        img_path: Path = sample["image_path"]
        annotation_path: Optional[Path] = sample["mask_path"]
        
        image = self.format_loader(img_path)
        mask = self.format_loader(annotation_path) if annotation_path is not None else None

        
        #choose bbox generator

        bbox= self.bbox_generator(mask)

        
    

        out: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "bbox": bbox,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "original_size": image.shape
        }
        
        if self.transforms is not None:
            out = self.transforms(out)
        
        
        if getattr(self, "return_metadata", False):
            out["metadata"] = {
                "path": str(path),
                "annotation_path": str(annotation_path) if annotation_path is not None else None,
                "file_name": path.name,
                "modality": self.__class__.__name__,
                "annotation_format": self.annotation_format,
                "prefix_match_annotations": self.prefix_match_annotations,
            }

        return out
        
        
        


##### Data loaders##########

class DicomLoader:
    def __call__(self, path):
        dcm = pydicom.dcmread(str(path), stop_before_pixels=False)
        img = dcm.pixel_array.astype(np.float32)

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = img.max() - img

        return img
    
   

class NpzLoader:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, path):
        with np.load(str(path)) as npz:
            arr = npz[npz.files[0]]

        return arr.astype(self.dtype) if self.dtype is not None else arr

    
class PngLoader:
    def __init__(self, dtype=None, mode=None):
        """
        Args:
            dtype: target numpy dtype (e.g. np.uint8)
            mode: optional PIL mode conversion ('L', 'RGB', 'RGBA', etc.)
        """
        self.dtype = dtype
        self.mode = mode

    def __call__(self, path):
        img = Image.open(str(path))

        if self.mode is not None:
            img = img.convert(self.mode)

        arr = np.array(img)

        return arr.astype(self.dtype) if self.dtype is not None else arr


#nifti loader for 3d volumes. This is just a mock-up. Logic about slice selection etc. must be addded
class NiftiLoader:
    def __call__(self, path):
        import nibabel as nib
        
        volume= np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        
        ###perform some transforms related to the modality###
        
        return volume


######## image and mask formatters to tensor,np etc.########

##receive !paired images and masks and perform several transforms on them: resize, normalize, binarize mask... Output the desired format.



######### box prompt generators ###########


    
class BBoxFromMask:   #receives a mask and outputs a bbox as 4 coords
    """Load an annotation mask and add mask + bbox prompts to a sample."""
    
    def __init__(
        self,
        roi_class_name: Optional[str] = None,
        annotation_threshold: float = 0.5,
        min_mask_area: Optional[int] = None,
        allow_empty_mask: bool = False,
        dilate_tiny_masks: bool = False,
        tiny_mask_threshold: int = 5,
        tiny_mask_dilation_iters: int = 3,
        bbox_padding: int = 0,
    ) -> None:
        self.roi_class_name = roi_class_name
        self.annotation_threshold = float(annotation_threshold)
        self.min_mask_area = min_mask_area
        self.allow_empty_mask = bool(allow_empty_mask)
        self.dilate_tiny_masks = bool(dilate_tiny_masks)
        self.tiny_mask_threshold = int(tiny_mask_threshold)
        self.tiny_mask_dilation_iters = int(tiny_mask_dilation_iters)
        self.bbox_padding = int(bbox_padding)


    def __call__(self, mask: np.ndarray) -> Optional[np.ndarray]:
        
        if mask is None:
            if self.allow_empty_mask:
                return None
            raise ValueError("Mask is None, but allow_empty_mask=False")
        
        mask = self._postprocess_mask(mask)
        bbox = self._compute_bbox_2d(mask)
        return bbox    


    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        area = int(mask.sum())
        if self.min_mask_area is not None and area < self.min_mask_area:
            if self.allow_empty_mask:
                return np.zeros_like(mask)
            raise ValueError(f"Mask too small: area={area} < {self.min_mask_area}")
        if self.dilate_tiny_masks and 0 < area <= self.tiny_mask_threshold:
            mask = binary_dilation(mask, iterations=self.tiny_mask_dilation_iters)
        return mask.astype(np.uint8)

    def _compute_bbox_2d(self, mask: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            if self.allow_empty_mask:
                return None
            raise ValueError("Empty mask")
        H, W = mask.shape
        x_min = max(0, int(xs.min()) - self.bbox_padding)
        x_max = min(W, int(xs.max()) + self.bbox_padding)
        y_min = max(0, int(ys.min()) - self.bbox_padding)
        y_max = min(H, int(ys.max()) + self.bbox_padding)
        return [np.array([x_min, y_min, x_max, y_max], dtype=np.float32)]


class ConnectedComponentsBBoxFromMask(BBoxFromMask):
    """Return one bounding box per connected component in a binary mask."""

    def __init__(
        self,
        roi_class_name: Optional[str] = None,
        annotation_threshold: float = 0.5,
        min_mask_area: Optional[int] = None,
        allow_empty_mask: bool = False,
        dilate_tiny_masks: bool = False,
        tiny_mask_threshold: int = 5,
        tiny_mask_dilation_iters: int = 3,
        bbox_padding: int = 0,
        connectivity: int = 2,
    ) -> None:
        super().__init__(
            roi_class_name=roi_class_name,
            annotation_threshold=annotation_threshold,
            min_mask_area=min_mask_area,
            allow_empty_mask=allow_empty_mask,
            dilate_tiny_masks=dilate_tiny_masks,
            tiny_mask_threshold=tiny_mask_threshold,
            tiny_mask_dilation_iters=tiny_mask_dilation_iters,
            bbox_padding=bbox_padding,
        )
        if connectivity not in {1, 2}:
            raise ValueError("connectivity must be 1 or 2")
        self.connectivity = int(connectivity)

    def _compute_bbox_2d(self, mask: np.ndarray) -> Optional[List[np.ndarray]]:
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() == 0:
            if self.allow_empty_mask:
                return []
            raise ValueError("Empty mask")

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        structure = np.ones((3, 3), dtype=np.uint8) if self.connectivity == 2 else None
        labeled_mask, num_components = connected_components(mask, structure=structure)
        H, W = mask.shape
        boxes = []

        for component_idx in range(1, num_components + 1):
            component = labeled_mask == component_idx
            area = int(component.sum())
            if self.min_mask_area is not None and area < self.min_mask_area:
                continue

            ys, xs = np.where(component)
            if len(xs) == 0:
                continue

            x_min = max(0, int(xs.min()) - self.bbox_padding)
            x_max = min(W, int(xs.max()) + self.bbox_padding)
            y_min = max(0, int(ys.min()) - self.bbox_padding)
            y_max = min(H, int(ys.max()) + self.bbox_padding)
            boxes.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))

        if len(boxes) == 0 and not self.allow_empty_mask:
            raise ValueError("No connected components left after filtering")

        return boxes
    



