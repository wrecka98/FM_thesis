import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from asymmetry_metrics import hybrid_asymmetry
from mirai_localized_dif_head import LocalizedDifModel
from mammo_clip_localized_diff_head import MammoCLIPLocalizedDifModel
from mirai_metadataset import MiraiMetadataset
from mammo_clip_metadataset import Mammo_CLIPMetadataset
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
import torch.nn as nn
import sys


def to_list_of_str(x):
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    if isinstance(x, torch.Tensor):
        # PyTorch doesn’t have string tensors; handle numeric/bytes just in case
        return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v.item() if v.ndim == 0 else v)
                for v in x]
    return [str(x)]


def main(device_ids=[0],
         save_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out",
         num_epochs=50,
         model_path="",
         KD_label="logit",
         risk_yr=1,
         loss_type="CE",
         label_col=None,
         dataset=None,
         data_file=None,
         use_stretch=True,
         train_backbone=False,
         flexible_asymmetry=True,
         use_stretch_matrix=False,
         batch_size=40,
         lr=1,
         max_workers=20,
         initial_asym_mean=4000,
         initial_asym_std=200,
         latent_h=5,
         latent_w=5,
         use_addon_layers=False,
         save_file_suffix="full",
         use_all_training_data=False,
         oversample_cancer_rate=13,
         topk_for_heatmap=None,
         align_images=False,
         multiple_pairs_per_exam=False,
         verbose=False,
         model=None,
         chk_pt=None,
         arch=None,
         batch_acc=5,
         use_bias=False,
         lr_step_size=None,
         weight_decay=1e-3,
         use_bn=False,
         save_interval=5,
         linear_only_epochs=[]):
    # Setting the primary cuda device to be the first listed device
    torch.cuda.set_device(0)
    # torch.cuda.set_device(device_ids[0])
    label_col = label_col.format(risk_yr)
    save_dir = save_dir.format(risk_yr)
    save_dir = Path(save_dir)

    output_file = f"validation_{risk_yr}yr_predictions_epoch38_{dataset}.csv"

    print(f"Using label column: {label_col}")
    print(f"Results will be saved to: {save_dir}")
    print(f"output_file: {output_file}")
    print(" ")
    val_accs = []
    avg_loss_vals = []

    output_dir = save_dir

    augmentations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=(-20, 20)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    def crop(img):
        nonzero_inds = torch.nonzero(img - torch.min(img))
        top = torch.min(nonzero_inds[:, 0])
        left = torch.min(nonzero_inds[:, 1])
        bottom = torch.max(nonzero_inds[:, 0])
        right = torch.max(nonzero_inds[:, 1])

        return img[top:bottom, left:right]

    def resize_and_normalize(img, use_crop=False, augment=True):
        img_mean = 7699.5
        img_std = 11765.06
        target_size = (1664, 2048)
        dummy_batch_dim = False

        if np.sum(img) == 0:
            img = torch.tensor(img).expand(1, 3, *img.shape) \
                .type(torch.FloatTensor)
            return F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')[0]

        # Adding a dummy batch dimension if necessary
        if len(img.shape) == 3:
            img = torch.unsqueeze(img, 0)
            dummy_batch_dim = True

        with torch.no_grad():
            if use_crop:
                img = crop(torch.tensor((img - img_mean) / img_std))
            else:
                img = torch.tensor((img - img_mean) / img_std)
            img = img.expand(1, 3, *img.shape) \
                .type(torch.FloatTensor)
            img_resized = F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')
            if augment:
                img_resized = augmentations(img_resized[0])
                return img_resized
        # img_resized = img

        if dummy_batch_dim:
            return img_resized[0]
        else:
            return img_resized[0]

    df = pd.read_csv(data_file)
    val_tfm = None
    train_tfm = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
        A.ElasticTransform(alpha=10, sigma=15)
    ], p=1.0)
    if dataset.lower() == "bu":
        val_df = df[df["split"] == "val"]
    elif dataset.lower() == "rsna":
        val_df = df
    # val_df = val_df.head(4)
    val_dataset = MiraiMetadataset(val_df, resizer=resize_and_normalize, mode="val",
                                   align_images=align_images,
                                   oversample_cancer_rate=oversample_cancer_rate,
                                   multiple_pairs_per_exam=multiple_pairs_per_exam, label_col=label_col)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=min(max_workers, batch_size))

    train_df = df[df["split"] == "train"]
    train_dataset = MiraiMetadataset(train_df, resizer=resize_and_normalize, mode="training",
                                     align_images=align_images,
                                     oversample_cancer_rate=oversample_cancer_rate,
                                     multiple_pairs_per_exam=multiple_pairs_per_exam, label_col=label_col)
    # print(oversample_cancer_rate)
    if oversample_cancer_rate is not None:
        pos_count = train_df[train_df[label_col] == 1].shape[0] * oversample_cancer_rate
        neg_count = train_df[train_df[label_col] == 0].shape[0]
    else:
        pos_count = train_df[train_df[label_col] == 1].shape[0]
        neg_count = train_df[train_df[label_col] == 0].shape[0]

    print(f"train_df.shape: {train_df.shape}, val_df.shape: {val_df.shape}")
    print(f"train data stats, pos_count: {pos_count}, neg_count: {neg_count}")
    print(
        f"val data stats, pos_count: {val_df[val_df[label_col] == 1].shape[0]}, neg_count: {val_df[val_df[label_col] == 0].shape[0]}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=min(max_workers, batch_size))

    # print(len(train_dataloader), len(val_dataloader))
    # print(xxx)

    print("Have not yet entered model")
    print(f"model: {model}, chk_pt: {chk_pt}, arch: {arch}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading model from scratch")
    model = LocalizedDifModel(
        asymmetry_metric=hybrid_asymmetry,
        chk_pt=chk_pt,
        embedding_channel=512,
        latent_h=latent_h,
        latent_w=latent_w,
        embedding_model=None,
        initial_asym_mean=initial_asym_mean,
        initial_asym_std=initial_asym_std,
        use_stretch=use_stretch,
        train_backbone=train_backbone,
        flexible_asymmetry=flexible_asymmetry,
        use_stretch_matrix=use_stretch_matrix,
        device_ids=device_ids,
        use_addon_layers=use_addon_layers,
        topk_for_heatmap=topk_for_heatmap,
        use_bias=use_bias,
        use_bn=use_bn)

    print(f"Loading model from: {output_dir / model_path}")
    try:
        model = torch.load(output_dir / model_path, map_location=device)
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file not found at {output_dir / model_path}")
        sys.exit(1)

    if oversample_cancer_rate is not None:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([1 - neg_count / (pos_count + neg_count),
                                                                   1 - pos_count / (pos_count + neg_count)]).cuda())

    param_list = [{'params': model.mlo_stretch_params, 'lr': lr, 'weight_decay': weight_decay},
                  {'params': model.cc_stretch_params, 'lr': lr, 'weight_decay': weight_decay}]
    if train_backbone:
        param_list = param_list + [{'params': model.backbone.parameters(), 'lr': lr / 10, 'weight_decay': weight_decay}]
    if use_addon_layers:
        param_list = param_list + [{'params': model.conv1.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
    if not (topk_for_heatmap is None):
        param_list = param_list + [{'params': model.topk_weights, 'lr': lr}]
    if use_bias:
        param_list = param_list + [{'params': model.learned_asym_mean, 'lr': 1e-1},
                                   {'params': model.learned_asym_std, 'lr': 1e-2}]
    if model.use_bn:
        param_list = param_list + [{'params': model.bn.parameters(), 'lr': lr}]

    print(
        f"train_backbone: {train_backbone}, use_addon_layers: {use_addon_layers}, topk_weights: {topk_for_heatmap},"
        f"use_bias: {use_bias}, use_bn: {use_bn}, flexible_asymmetry: {flexible_asymmetry}")
    # print(param_list)
    # for p in model.backbone.parameters():
    #     print(p)  # full tensor
    #     break
    # print(xxx)
    optimizer = torch.optim.Adam(param_list)  # , lr=lr)
    if lr_step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.2)

    if verbose:
        print("About to enter train loop")

    results = []
    print("Starting inference...")

    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataloader, desc="Processing Batches")):
            eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
            l_cc_img, r_cc_img, l_mlo_img, r_mlo_img = l_cc_img.cuda(), r_cc_img.cuda(), l_mlo_img.cuda(), r_mlo_img.cuda()
            label = label.cuda()

            output, proba_cancer, logit_cancer, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

            cc_meta, mlo_meta = other[0], other[1]
            hm_cc, hm_mlo = cc_meta['heatmap'], mlo_meta['heatmap']

            exam_id_str = to_list_of_str(eid)[0]

            exam_save_dir = output_dir / exam_id_str
            exam_save_dir.mkdir(parents=True, exist_ok=True)

            # Save the heatmap tensors. Move to CPU first as a best practice.
            torch.save(cc_meta, exam_save_dir / 'cc_heatmap.pt')
            torch.save(mlo_meta, exam_save_dir / 'mlo_heatmap.pt')

            sample_result = {
                'exam_id': to_list_of_str(eid)[0],
                'prediction_neg': output[0, 0].item(),
                'prediction_pos': output[0, 1].item(),
                'y_argmin_cc': cc_meta['y_argmin'][0],
                'x_argmin_cc': cc_meta['x_argmin'][0],
                'y_argmin_mlo': mlo_meta['y_argmin'][0],
                'x_argmin_mlo': mlo_meta['x_argmin'][0],
                'l_cc_path': to_list_of_str(l_cc_path)[0],
                'r_cc_path': to_list_of_str(r_cc_path)[0],
                'l_mlo_path': to_list_of_str(l_mlo_path)[0],
                'r_mlo_path': to_list_of_str(r_mlo_path)[0],
                'proba_cancer': proba_cancer[0].item(),
                'target': label[0].item(),
            }
            results.append(sample_result)

            if (i + 1) % save_interval == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)

    # 5. Save Final Results
    if results:
        pd.DataFrame(results).to_csv(output_dir / output_file, index=False)
        print(f"\nInference complete. All results saved to {output_dir / output_file}")
        print(f"Save file: {output_dir}")
    else:
        print("No samples were processed.")


if __name__ == '__main__':
    main(device_ids=[2],
         num_epochs=50,
         use_stretch=True,
         train_backbone=False,
         flexible_asymmetry=True,
         batch_size=40,
         lr=1,
         max_workers=20)
