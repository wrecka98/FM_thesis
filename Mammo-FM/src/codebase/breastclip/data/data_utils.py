import os
from typing import Dict

from albumentations import *
from transformers import AutoTokenizer


def load_tokenizer(source, pretrained_model_name_or_path, cache_dir, **kwargs):
    if source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
            **kwargs,
        )
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.cls_token_id
    elif source == "ft":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="answerdotai/ModernBERT-base",
            padding_side="right",
            trust_remote_code=True,
            **kwargs,
        )
        # Set special token IDs from the config
        tokenizer.bos_token_id = 50281  # [CLS]
        tokenizer.eos_token_id = 50282  # [SEP]
        tokenizer.pad_token_id = 50283  # [PAD]
        tokenizer.unk_token_id = 50280  # [UNK]
        tokenizer.cls_token_id = 50281  # [CLS]
        tokenizer.sep_token_id = 50282  # [SEP]
    else:
        raise KeyError(f"Not supported tokenizer source: {source}")

    return tokenizer


def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}
    transforms = transform_config[split]
    if split == "train":
        if (transforms["Resize"]["size_h"] == 512 or transforms["Resize"]["size_h"] == 224) and (
                transforms["Resize"]["size_w"] == 512 or transforms["Resize"]["size_w"] == 224):
            return Compose([
                Resize(width=transforms["Resize"]["size_h"], height=transforms["Resize"]["size_w"]),
                HorizontalFlip(),
                VerticalFlip(),
                Affine(
                    rotate=transforms["transform"]["affine_transform_degree"],
                    translate_percent=transforms["transform"]["affine_translate_percent"],
                    scale=transforms["transform"]["affine_scale"],
                    shear=transforms["transform"]["affine_shear"]
                ),
                ElasticTransform(
                    alpha=transforms["transform"]["elastic_transform_alpha"],
                    sigma=transforms["transform"]["elastic_transform_sigma"]
                )
            ], p=transforms["transform"]["p"]
            )
        else:
            return Compose([
                HorizontalFlip(),
                VerticalFlip(),
                Affine(
                    rotate=transforms["transform"]["affine_transform_degree"],
                    translate_percent=transforms["transform"]["affine_translate_percent"],
                    scale=transforms["transform"]["affine_scale"],
                    shear=transforms["transform"]["affine_shear"]
                ),
                ElasticTransform(
                    alpha=transforms["transform"]["elastic_transform_alpha"],
                    sigma=transforms["transform"]["elastic_transform_sigma"]
                )
            ], p=transforms["transform"]["p"]
            )
    elif split == "valid":
        if transforms["Resize"]["size_h"] == 512 and transforms["Resize"]["size_w"] == 512:
            return Compose([
                Resize(width=transforms["Resize"]["size_h"], height=transforms["Resize"]["size_w"])
            ])