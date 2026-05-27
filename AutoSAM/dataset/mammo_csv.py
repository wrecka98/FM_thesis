import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


def _parse_values(value):
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def _resolve_path(data_root, value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(data_root) / path


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image.astype(np.float32)


def _read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.float32)


class MammographyCSVVolumeDataset(torch.utils.data.Dataset):
    """
    Adapts 2D mammography PNG image/mask pairs to the current AutoSAM 3D
    training loop by repeating each 2D slice along a synthetic depth axis.

    Returned tensors match LungData.py:
      image: H x W x D
      mask:  H x W x D
      original_size: [H, W]
      image_size:    [H, W]
    """

    def __init__(
        self,
        dataframe,
        data_root,
        image_col="image_path",
        mask_col="mask_path",
        id_col="unique_id",
        depth=32,
        loops=1,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.image_col = image_col
        self.mask_col = mask_col
        self.id_col = id_col
        self.depth = int(depth)
        self.loops = int(loops)
        print(f"num of data:{len(self.df)}")

    def __len__(self):
        return len(self.df) * self.loops

    def __getitem__(self, index):
        row = self.df.iloc[index % len(self.df)]
        image_path = _resolve_path(self.data_root, row[self.image_col])
        mask_path = _resolve_path(self.data_root, row[self.mask_col])

        image = _read_image(image_path)
        mask = _read_mask(mask_path)

        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)

        image_volume = np.repeat(image[:, :, None], self.depth, axis=2)
        mask_volume = np.repeat(mask[:, :, None], self.depth, axis=2)

        image_tensor = torch.tensor(image_volume, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_volume, dtype=torch.float32)
        original_size = torch.tensor(image.shape, dtype=torch.float32)
        image_size = torch.tensor(image.shape, dtype=torch.float32)

        return image_tensor, mask_tensor, original_size, image_size


def _load_fold_dataframe(args):
    csv_template = args.get("csv_template", "")
    csv_file = args.get("csv_file", "")
    fold = str(args.get("fold", 0))

    if csv_file:
        csv_path = Path(csv_file)
    elif csv_template:
        csv_path = Path(str(csv_template).format(fold=fold))
    else:
        raise ValueError("For task='mammo', provide --csv_file or --csv_template.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}.")
    return df.reset_index(drop=True)


def get_mammo_csv_dataset(args, sam_trans=None):
    df = _load_fold_dataframe(args)
    split_col = args.get("split_col", "split")
    fold_col = args.get("fold_col", "fold")
    fold = str(args.get("fold", 0))
    train_values = _parse_values(args.get("train_split_values", "training,train,trainval"))
    test_values = _parse_values(args.get("test_split_values", "test"))

    if split_col not in df.columns:
        raise ValueError(f"CSV must contain split column '{split_col}'.")

    split = df[split_col].astype(str).str.lower()
    train_mask = split.isin(train_values)
    test_mask = split.isin(test_values)

    if fold_col in df.columns:
        fold_values = df[fold_col].astype(str)
        fold_train_mask = train_mask & (fold_values == fold)
        if fold_train_mask.any():
            train_mask = fold_train_mask

        fold_test_mask = test_mask & (fold_values == fold)
        held_out_test_mask = test_mask & fold_values.isin(["-1", "heldout", "held-out"])
        if fold_test_mask.any():
            test_mask = fold_test_mask
        elif held_out_test_mask.any():
            test_mask = held_out_test_mask

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found for split values: {train_values}")
    if test_df.empty:
        raise ValueError(f"No testing rows found for split values: {test_values}")

    depth = int(args.get("NumSliceDim", 32))
    loops = int(args.get("train_loops", 1))
    data_root = args.get("data_root", ".")
    image_col = args.get("image_col", "image_path")
    mask_col = args.get("mask_col", "mask_path")
    id_col = args.get("id_col", "unique_id")

    trainset = MammographyCSVVolumeDataset(
        train_df,
        data_root=data_root,
        image_col=image_col,
        mask_col=mask_col,
        id_col=id_col,
        depth=depth,
        loops=loops,
    )
    testset = MammographyCSVVolumeDataset(
        test_df,
        data_root=data_root,
        image_col=image_col,
        mask_col=mask_col,
        id_col=id_col,
        depth=depth,
        loops=1,
    )
    return trainset, testset
