import torch
import torch.nn.functional as F
import numpy as np


class MedSAM(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks
    


class MedSAMPreprocess:
    """Convert BaseDataset numpy sample into tensors and scale boxes to MedSAM size."""

    def __init__(self, size=(1024, 1024)):
        self.size = tuple(size)

    def __call__(self, sample):
        image = sample["image"]
        mask = sample.get("mask", None)
        boxes = sample.get("bbox", None)

        # -------------------------
        # Image handling
        # -------------------------
        if image.ndim == 3:
            image = image[..., 0]

        old_h, old_w = image.shape[:2]
        new_h, new_w = self.size

        image = image.astype(np.float32)
        image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)

        image_t = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)
        image_t = F.interpolate(
            image_t.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # -------------------------
        # Mask handling
        # -------------------------
        if mask is None:
            mask_t = None
        else:
            if mask.ndim == 3:
                mask = mask[..., 0]

            mask_t = torch.from_numpy((mask > 0).astype(np.uint8)).float().unsqueeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0),
                size=self.size,
                mode="nearest",
            ).squeeze(0).to(torch.uint8)

        # -------------------------
        # BBox handling
        # -------------------------
        if boxes is None:
            box_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

            box_arr[:, [0, 2]] *= new_w / old_w
            box_arr[:, [1, 3]] *= new_h / old_h

            box_t = torch.from_numpy(box_arr).float()

        sample["image"] = image_t
        sample["mask"] = mask_t
        sample["bbox"] = box_t
        sample["original_size"] = (old_h, old_w)
        sample["resized_size"] = self.size

        return sample