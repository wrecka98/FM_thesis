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
         linear_only_epochs=[]):
    # Setting the primary cuda device to be the first listed device
    torch.cuda.set_device(0)
    # torch.cuda.set_device(device_ids[0])
    save_dir = Path(save_dir)
    save_dir = save_dir / dataset / arch
    train_preds_dir = save_dir / "training_preds"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_preds_dir, exist_ok=True)
    print("Saving to directory:", train_preds_dir)
    val_accs = []
    avg_loss_vals = []

    df = pd.read_csv(data_file)
    val_tfm = None
    val_df = df[df["split"] == "val"]
    val_dataset = Mammo_CLIPMetadataset(val_df, dataset=dataset, transforms=val_tfm, align_images=align_images,
                                        multiple_pairs_per_exam=False, label_col=label_col)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=min(max_workers, batch_size))

    print(
        f"val data stats, pos_count: {val_df[val_df[label_col] == 1].shape[0]}, neg_count: {val_df[val_df[label_col] == 0].shape[0]}")

    print("Have not yet entered model")
    print(f"model: {model}, chk_pt: {chk_pt}, arch: {arch}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MammoCLIPLocalizedDifModel(
            asymmetry_metric=hybrid_asymmetry,
            chk_pt=chk_pt,
            arch=arch,
            embedding_channel=2048,
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
            use_bn=use_bn,
            device=device)

    model.eval()
    print("Starting calculation...")
    all_raw_scores = []
    all_raw_scores = []

    # 2. Wrap your dataloader with tqdm() and add an optional description
    # The loop will now display a progress bar.
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataloader, desc="Calculating Raw Asymmetry Scores")):
            # --- Your data loading logic ---
            (
                eid, label,
                logit_yr1, logit_yr2, logit_yr3, logit_yr4, logit_yr5,
                year1_risk, year2_risk, year3_risk, year4_risk, year5_risk,
                l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path
            ) = sample
            l_cc_img = l_cc_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)
            r_cc_img = r_cc_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)
            l_mlo_img = l_mlo_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)
            r_mlo_img = r_mlo_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)

            # --- Call the model with the new flag ---
            raw_scores_batch, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img, get_raw_score=True)

            # Append the batch scores to our master list
            all_raw_scores.append(raw_scores_batch.cpu())

            # 3. The manual print statement is no longer needed
            # if (i + 1) % 100 == 0:
            #     print(f"Processed {i + 1} batches...")

    # Concatenate all collected tensors into a single large tensor
    all_scores_tensor = torch.cat(all_raw_scores)

    # --- Final calculation and printing ---
    new_asym_mean = torch.mean(all_scores_tensor)
    new_asym_std = torch.std(all_scores_tensor)

    print("\n" + "=" * 40)
    print("Calculation Complete")
    print(f"Total samples processed: {len(all_scores_tensor)}")
    print(f"Computed initial_asym_mean: {new_asym_mean.item()}")
    print(f"Computed initial_asym_std: {new_asym_std.item()}")
    print("=" * 40)

    return val_accs, avg_loss_vals


if __name__ == '__main__':
    main(device_ids=[2],
         num_epochs=50,
         use_stretch=True,
         train_backbone=False,
         flexible_asymmetry=True,
         batch_size=40,
         lr=1,
         max_workers=20)
