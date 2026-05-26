import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor 
import albumentations as A
from pathlib import Path

class BinaryLoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'datasets'
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]

            image_path = os.path.join(self.path,'image_1024/',image_id)
            mask_path = os.path.join(self.path,'mask_1024/',image_id)

            modality_name = list(image_id.split('_'))[0]

            mcls = None

            if modality_name == 'ISIC':
                mcls = 0
            elif modality_name == 'CHNCXR':
                mcls = 1
            elif modality_name == 'ultra':
                mcls = 2

            if mcls is None and len(list(image_id.split('_'))) > 1:
                mcls = 3

            if mcls is None and len(modality_name) > 10:
                mcls = 4

            if mcls is None and len(modality_name) <= 10:
                mcls = 5
 
    
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32')
            mask = io.imread(mask_path+'.png', as_gray=True)

            mask[mask>0]=255
            


            data_group = self.transforms(image=img, mask=mask)
            img_resized = data_group['image']
            mask = data_group['mask']

            img = self.img_tesnor(img)
            img = self.preprocess(img)

            # print(mask.shape)
            # mask_64 = F.interpolate(mask, scale_factor=2)
            # print(mask_64.shape)

   
            return (img_resized, img, mask, image_id, mcls)
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x


class CSVMammoBinaryLoader(Dataset):
        def __init__(
            self,
            dataframe,
            transforms,
            data_root=".",
            image_col="image_path",
            mask_col="mask_path",
            id_col="unique_id",
            domain_class=1,
        ):
            self.df = dataframe.reset_index(drop=True)
            self.transforms = transforms
            self.data_root = Path(data_root)
            self.image_col = image_col
            self.mask_col = mask_col
            self.id_col = id_col
            self.domain_class = int(domain_class)

        def __len__(self):
            return len(self.df)

        def _resolve_path(self, value):
            path = Path(str(value))
            if path.is_absolute():
                return path
            return self.data_root / path

        def _read_image(self, path):
            img = io.imread(str(path))
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            return img.astype("float32")

        def _read_mask(self, path):
            mask = io.imread(str(path), as_gray=True)
            mask = (mask > 0).astype("float32")
            return mask

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_path = self._resolve_path(row[self.image_col])
            mask_path = self._resolve_path(row[self.mask_col])

            img = self._read_image(image_path)
            mask = self._read_mask(mask_path)

            data_group = self.transforms(image=img, mask=mask)
            img_tensor = data_group["image"].float()
            mask_tensor = data_group["mask"].float()
            if mask_tensor.max() > 1:
                mask_tensor = mask_tensor / 255.0
            mask_tensor = (mask_tensor > 0.5).float()

            if self.id_col in row.index:
                image_id = str(row[self.id_col])
            else:
                image_id = image_path.stem

            return img_tensor, mask_tensor, image_id, self.domain_class
