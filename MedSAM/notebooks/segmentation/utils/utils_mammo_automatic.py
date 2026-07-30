from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union
import plistlib
import sys

import numpy as np
#import pydicom
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from skimage.draw import polygon

base_dataset_path="/home/jovyan/thesis_project/FM_thesis"
sys.path.append(base_dataset_path)

from root_utils.ood_datasets import BaseIndexedDataset



PathLike = Union[str, Path]
AnnotationFormat = Literal["xml", "mask_image", "none", "auto"]
ImageSuffix = Literal[".dcm", ".dicom"]


class MedSAMMammoDataset(BaseIndexedDataset):
    """
    2D mammography dataset for MedSAM-style training/evaluation.

    Expected sample contract:
        {
            "image": Tensor[C, H, W],
            "label": LongTensor[],
            "path": str,
            "annotation_path": str | None,
            "mask": Tensor[1, H, W] | None,
            "bbox": Tensor[4] | None,
            "metadata": dict[str, Any]  # optional
        }

    Supports:
    - DICOM mammograms
    - INbreast-style XML ROI annotations
    - mask image annotations (png/jpg/tif/...) if desired
    - optional filtering by ROI class (e.g. "Mass")
    - optional skipping of empty masks
    - optional dilation of very small masks
    - optional prefix-based image/annotation matching
    """

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
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        model_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_suffixes: Sequence[str] = (".dcm", ".dicom"),
        annotation_dir_name: Optional[str] = None,
        annotation_suffixes: Sequence[str] = (".xml", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"),
        annotation_format: AnnotationFormat = "auto",
        prefix_match_annotations: bool = True,
        skip_empty_mask: bool = False,
        invert_monochrome1: bool = True,
        percentile_clip: Optional[tuple[float, float]] = (1.0, 99.0),
        normalize: bool = True,
        annotation_threshold: float = 0.0,
    ) -> None:


        self.transform = transform
        self.model_transform = model_transform
        self.image_suffixes = tuple(s.lower() for s in image_suffixes)
        self.annotation_dir_name = annotation_dir_name
        self.annotation_suffixes = tuple(s.lower() for s in annotation_suffixes)
        self.annotation_format = annotation_format
        self.prefix_match_annotations = bool(prefix_match_annotations)
        self.skip_empty_mask = bool(skip_empty_mask)
        self.invert_monochrome1 = bool(invert_monochrome1)
        self.percentile_clip = percentile_clip
        self.normalize = bool(normalize)
        self.annotation_threshold = float(annotation_threshold)

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
        if file_paths is not None:
            files = [Path(p) for p in file_paths]
            return [p for p in files if p.is_file() and p.suffix.lower() in self.image_suffixes]

        if root is None:
            raise ValueError("Provide either `root` or `file_paths`.")

        root = Path(root)
        files = sorted(
            [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self.image_suffixes]
        )
        return files

    def _build_samples(self, files: Sequence[Path]) -> list[dict[str, Any]]:    #only builds samples aka images from _collect_files paths and attaches masks if available
        samples: list[dict[str, Any]] = []

        for path in files:
            annotation_path = self._image_to_annotation_path(path)


            if self.skip_empty_mask and annotation_path is None:    #this entirely skips the sample that does not contain a  mask. So be careful about skip_empty_mask=True
                continue

            samples.append(
                {
                    "path": path,
                    "annotation_path": annotation_path,
                    "label": self.label,
                    "sample_type": "image",
                }
            )
        
        print(f"found {len(samples)} samples")

        return samples

    
    
    def _image_to_annotation_path(self, image_path: Path) -> Optional[Path]:               # this logic is quite specific to INbreast only. Maybe standardize it more
        if self.annotation_format == "none":
            return None

        search_root = image_path.parent
        if self.annotation_dir_name is not None:
            candidate_dir = image_path.parent.parent / self.annotation_dir_name
            if candidate_dir.exists():
                search_root = candidate_dir

        prefix = image_path.name.split("_")[0] if self.prefix_match_annotations else image_path.stem

        candidates: list[Path] = []
        for suffix in self.annotation_suffixes:
            if self.prefix_match_annotations:
                candidates.extend(sorted(search_root.glob(f"{prefix}*{suffix}")))
            else:
                exact = search_root / f"{image_path.stem}{suffix}"
                if exact.exists():
                    candidates.append(exact)

        if not candidates:
            return None

        if self.annotation_format == "xml":
            xmls = [p for p in candidates if p.suffix.lower() == ".xml"]
            return xmls[0] if xmls else None

        if self.annotation_format == "mask_image":
            imgs = [p for p in candidates if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}]
            return imgs[0] if imgs else None

        # auto: prefer XML for INbreast-style ROI annotations, then mask images.
        xmls = [p for p in candidates if p.suffix.lower() == ".xml"]
        if xmls:
            return xmls[0]

        return candidates[0]
    
    


    def _peek_image_shape(self, path: Path) -> tuple[int, int]:
        dcm = pydicom.dcmread(str(path), stop_before_pixels=False)
        img = dcm.pixel_array
        if img.ndim < 2:
            raise ValueError(f"Expected at least 2D image, got shape {img.shape} for {path}")
        return int(img.shape[0]), int(img.shape[1])

    def _load_dicom_image(self, path: Path) -> np.ndarray:
        dcm = pydicom.dcmread(str(path), stop_before_pixels=False)
        img = dcm.pixel_array.astype(np.float32)

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        if self.invert_monochrome1 and getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = img.max() - img

        if self.percentile_clip is not None:
            lo_pct, hi_pct = self.percentile_clip
            lo, hi = np.percentile(img, [lo_pct, hi_pct])
            img = np.clip(img, lo, hi)
        else:
            lo, hi = float(img.min()), float(img.max())

        if self.normalize:
            denom = float(hi - lo)
            if denom > 0:
                img = (img - lo) / (denom + 1e-8)
            else:
                img = np.zeros_like(img, dtype=np.float32)

        return img.astype(np.float32)




    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        path: Path = sample["path"]
        annotation_path: Optional[Path] = sample["annotation_path"]

        img2d = self._load_dicom_image(path)
        image = torch.from_numpy(img2d).float().unsqueeze(0)  # [1, H, W]


        out: dict[str, Any] = {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "path": str(path),
            "annotation_path": str(annotation_path) if annotation_path is not None else None,
        }



        if self.return_metadata:
            out["metadata"] = {
                "path": str(path),
                "annotation_path": str(annotation_path) if annotation_path is not None else None,
                "file_name": path.name,
                "sample_type": "image",
                "original_shape": tuple(img2d.shape),
                "modality": self.__class__.__name__,
                "annotation_format": self.annotation_format,
                "prefix_match_annotations": self.prefix_match_annotations,
            }
            
        print(f"annotation_path:{out['annotation_path']}")

        if self.transform is not None:
            out = self.transform(out)

        if self.model_transform is not None:
            out["image"] = self.model_transform(out["image"])

        return out


   

class BBoxFromMask:
    def __init__(
        self,
        roi_class_name=None,
        annotation_threshold=0.5,
        min_mask_area=None,
        allow_empty_mask=False,
        dilate_tiny_masks=False,
        tiny_mask_threshold=5,
        tiny_mask_dilation_iters=3,
        bbox_padding=0,
    ):
        self.roi_class_name = roi_class_name
        self.annotation_threshold = annotation_threshold
        self.min_mask_area = min_mask_area
        self.allow_empty_mask = allow_empty_mask
        self.dilate_tiny_masks = dilate_tiny_masks
        self.tiny_mask_threshold = tiny_mask_threshold
        self.tiny_mask_dilation_iters = tiny_mask_dilation_iters
        self.bbox_padding = bbox_padding

    def __call__(self, sample):
        annotation_path = sample.get("annotation_path")
        image = sample["image"]
        _, H, W = image.shape

        if annotation_path is None:
            sample["mask"] = None
            sample["bbox"] = None
            print("annot_path is none!")
            return sample

        mask = self._load_annotation_mask(annotation_path, (H, W))
        mask = self._postprocess_mask(mask)

        sample["mask"] = torch.from_numpy(mask).unsqueeze(0).to(torch.uint8)
        sample["bbox"] = self._compute_bbox_2d(mask)

        return sample

    def _load_annotation_mask(self, annotation_path, image_shape):
        suffix = Path(annotation_path).suffix.lower()
        
        print(f"suffix:{suffix}")

        if suffix == ".xml":
            return self._load_inbreast_xml_mask(annotation_path, image_shape)

        if suffix == ".npy":
            arr = np.load(annotation_path)
            if arr.shape != image_shape:
                raise ValueError("Shape mismatch")
            return (arr > self.annotation_threshold).astype(np.uint8)

        from PIL import Image
        arr = np.array(Image.open(annotation_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > self.annotation_threshold).astype(np.uint8)
    

    def _load_inbreast_xml_mask(
        self,
        xml_path: Path,
        image_shape: tuple[int, int],
        roi_class_name: Optional[str] = None,
    ) -> np.ndarray:
        def load_point(point_string: str) -> tuple[float, float]:
            x, y = [float(num) for num in point_string.strip("()").split(",")]
            return y, x  # col, row

        mask = np.zeros(image_shape, dtype=np.uint8)

        with open(xml_path, "rb") as f:
            plist_dict = plistlib.load(f, fmt=plistlib.FMT_XML)["Images"][0]

        rois = plist_dict.get("ROIs", [])
        for roi in rois:
            if roi_class_name is not None:
                name = roi.get("Name", "")
                if roi_class_name == "Calcification":
                    if name not in {"Calcification", "Cluster"}:
                        continue
                elif name != roi_class_name:
                    continue

            points = roi.get("Point_px", [])
            if not points:
                continue

            coords = [load_point(p) for p in points]
            if len(coords) <= 2:
                for r, c in coords:
                    rr = int(round(r))
                    cc = int(round(c))
                    if 0 <= rr < image_shape[0] and 0 <= cc < image_shape[1]:
                        mask[rr, cc] = 1
                continue

            rows = np.array([p[0] for p in coords], dtype=np.float32)
            cols = np.array([p[1] for p in coords], dtype=np.float32)
            rr, cc = polygon(rows, cols, shape=image_shape)
            mask[rr, cc] = 1

        return mask
    
    
    def _postprocess_mask(self, mask):
        mask = (mask > 0).astype(np.uint8)
        area = int(mask.sum())

        if self.min_mask_area is not None and area < self.min_mask_area:
            if self.allow_empty_mask:
                return np.zeros_like(mask)
            raise ValueError("Mask too small")

        if self.dilate_tiny_masks and 0 < area <= self.tiny_mask_threshold:
            mask = binary_dilation(mask, iterations=self.tiny_mask_dilation_iters)

        return mask.astype(np.uint8)

    def _compute_bbox_2d(self, mask):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            if self.allow_empty_mask:
                return None
            raise ValueError("Empty mask")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


    
    
    
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
        size: tuple[int, int],
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
        if x.ndim != 3:
            raise ValueError(f"Expected 3D image tensor [C, H, W], got shape {tuple(x.shape)}")
        return F.interpolate(
            x.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _resize_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D mask tensor [1, H, W], got shape {tuple(x.shape)}")
        return F.interpolate(
            x.unsqueeze(0),
            size=self.size,
            mode="nearest",
        ).squeeze(0)

    @staticmethod
    def _resize_bbox_2d(
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
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image tensor [C, H, W], got shape {tuple(image.shape)}")

        _, old_h, old_w = image.shape
        new_h, new_w = self.size

        if self.resize_image:
            sample["image"] = self._resize_image(image)

        if self.resize_mask and "mask" in sample and sample["mask"] is not None:
            sample["mask"] = self._resize_mask(sample["mask"])

        if self.resize_bbox and "bbox" in sample and sample["bbox"] is not None:
            sample["bbox"] = self._resize_bbox_2d(
                sample["bbox"],
                old_hw=(old_h, old_w),
                new_hw=(new_h, new_w),
            )

        sample["original_size"] = (old_h, old_w)
        sample["resized_size"] = self.size
        return sample

    




@torch.no_grad()
def medsam_inference_single(
    medsam_model,
    img_embed: torch.Tensor,
    box_1024: torch.Tensor,
    H: int,
    W: int,
    threshold: float = 0.5,
    return_prob: bool = False,
):
    """
    img_embed: [1, 256, 64, 64]
    box_1024: [4] or [1, 4]
    returns:
        pred_mask: [1, 1, H, W] uint8
        optionally probs: [1, 1, H, W] float
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=img_embed.device)

    if box_torch.ndim == 1:
        box_torch = box_torch[None, :]      # [1, 4]
    if box_torch.ndim == 2:
        box_torch = box_torch[:, None, :]   # [1, 1, 4]

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # [1, 256, 64, 64]
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    probs = torch.sigmoid(low_res_logits)  # [1, 1, 256, 256]
    probs = F.interpolate(
        probs,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # [1, 1, H, W]

    pred_mask = (probs > threshold).to(torch.uint8)

    if return_prob:
        return pred_mask, probs
    return pred_mask



@torch.no_grad()
def medsam_inference_batch(
    medsam_model,
    img_embed: torch.Tensor,
    box_1024: torch.Tensor,
    H: int,
    W: int,
    threshold: float = 0.5,
):
    """
    img_embed: [B, 256, 64, 64]
    box_1024: [B, 4]
    returns:
        pred_masks: [B, 1, H, W]
    """
    if img_embed.ndim != 4:
        raise ValueError(f"Expected img_embed [B, C, H, W], got {tuple(img_embed.shape)}")
    if box_1024.ndim != 2 or box_1024.shape[1] != 4:
        raise ValueError(f"Expected box_1024 [B, 4], got {tuple(box_1024.shape)}")

    B = img_embed.shape[0]
    if box_1024.shape[0] != B:
        raise ValueError(
            f"Batch mismatch: img_embed batch={B}, boxes batch={box_1024.shape[0]}"
        )

    preds = []
    for i in range(B):
        pred_i = medsam_inference_single(
            medsam_model=medsam_model,
            img_embed=img_embed[i:i+1],   # [1, 256, 64, 64]
            box_1024=box_1024[i],         # [4]
            H=H,
            W=W,
            threshold=threshold,
        )
        preds.append(pred_i)

    return torch.cat(preds, dim=0)  # [B, 1, H, W]
