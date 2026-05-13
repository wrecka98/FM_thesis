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


class KnowledgeDistillationLoss(nn.Module):
    """
    Binary KD: KL( Bern(p_teacher_T) || Bern(p_student_T) ) at temperature T,
    plus supervised BCEWithLogits on the student logit (no temperature).
    """

    def __init__(self, temperature=1.0, alpha=1.0, pos_weight=None, eps=1e-7):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)

    def forward(self, student_logit, teacher_logit, target):
        s = student_logit.float().view(-1)
        t = teacher_logit.float().view(-1)
        y = target.float().view(-1)

        sT = s / self.T
        tT = t / self.T

        p_t = torch.sigmoid(tT)

        log_p_s = -F.softplus(-sT)  # log σ(sT)
        log1m_p_s = -F.softplus(sT)  # log (1-σ(sT))

        kl = p_t * (torch.log(p_t.clamp_min(self.eps)) - log_p_s) + \
             (1 - p_t) * (torch.log((1 - p_t).clamp_min(self.eps)) - log1m_p_s)
        kd_loss = kl.mean() * (self.T ** 2)

        task_loss = self.bce(s, y)

        return self.alpha * kd_loss + (1.0 - self.alpha) * task_loss


class MSELoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=1.0, pos_weight=None, eps=1e-7, device=None):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.eps = float(eps)
        if pos_weight is not None:
            pos_weight = pos_weight.float()
            if device is not None:
                pos_weight = pos_weight.to(device)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)

    def forward(self, student_logit, teacher_logit, target):
        # flatten + cast everything to float32
        s = student_logit.float().view(-1)
        t = teacher_logit.float().view(-1)
        y = target.float().view(-1)

        # use the casted s/t (not the originals) to avoid float64 sneaking in
        mse = F.mse_loss(s / self.T, t / self.T)
        task_loss = self.bce(s, y)
        return self.alpha * mse + (1.0 - self.alpha) * task_loss


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
         save_dir=None,
         training_stage="stage1",
         stage2_model_path=None,
         num_epochs=50,
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
         use_mean_risk=False,
         linear_only_epochs=[]):
    # Setting the primary cuda device to be the first listed device
    torch.cuda.set_device(0)
    # torch.cuda.set_device(device_ids[0])
    save_dir = Path(save_dir)
    train_preds_dir = save_dir / dataset / arch / training_stage / f"loss_{loss_type}_unweighted" / f"risk_yr_{risk_yr}_KD_col_{KD_label}_use_mean_risk_{use_mean_risk}" / "training_preds"
    os.makedirs(train_preds_dir, exist_ok=True)
    print("Saving to directory:", train_preds_dir)
    label_col = f"cancer{risk_yr}yr_updated" if dataset == "bu" else label_col
    val_accs = []
    avg_loss_vals = []

    df = pd.read_csv(data_file)
    val_tfm = None
    train_tfm = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
        A.ElasticTransform(alpha=10, sigma=15)
    ], p=1.0)
    val_df = df[df["split"] == "val"]
    val_dataset = Mammo_CLIPMetadataset(val_df, dataset=dataset, transforms=val_tfm, align_images=align_images,
                                        multiple_pairs_per_exam=False, label_col=label_col, use_mean_risk=use_mean_risk)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=min(max_workers, batch_size))

    train_df = df[df["split"] == "train"]
    train_dataset = Mammo_CLIPMetadataset(train_df, dataset=dataset, transforms=train_tfm, align_images=align_images,
                                          multiple_pairs_per_exam=False, label_col=label_col,
                                          use_mean_risk=use_mean_risk)
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

    if training_stage.lower() == "stage2":
        model = torch.load(stage2_model_path, map_location=device)
        print("Stage 1 model is loaded, starting stage 2 training...")

    if oversample_cancer_rate is not None:
        loss_func = torch.nn.CrossEntropyLoss()
    elif loss_type == "KD":
        eps = 1e-8
        pos_weight_value = neg_count / (pos_count + eps)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        loss_func = KnowledgeDistillationLoss(pos_weight=pos_weight)
        print("KD loss is initialized")
    elif loss_type == "MSE":
        eps = 1e-8
        pos_weight_value = neg_count / (pos_count + eps)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        loss_func = MSELoss(pos_weight=pos_weight)
        print("MSE loss is initialized")
    else:
        loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([1 - neg_count / (pos_count + neg_count),
                                                                   1 - pos_count / (pos_count + neg_count)]).cuda())
        print("CrossEntropyLoss loss is initialized")

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
    for epoch in range(num_epochs):
        if epoch in linear_only_epochs:
            print("Setting no grad")
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
        total_loss = 0
        total_loss_for_subbatch = 0
        num_samples = 0
        subbatch_samples = 0
        start = time.time()
        optimizer.zero_grad()
        teacher_logits = None
        predictions_for_epoch_pos = []
        predictions_for_epoch_neg = []
        eids_for_epoch = []

        for sample_ind, sample in enumerate(tqdm(train_dataloader, desc="Running train set")):
            if verbose:
                print(f"Loaded next sample in {time.time() - start} seconds")
            if multiple_pairs_per_exam:
                eid, label, exam_list = sample
                label = label.cuda()
                output, _ = model(None, None, None, None, exam_list=exam_list)
            else:
                (
                    eid, label,
                    logit_yr1, logit_yr2, logit_yr3, logit_yr4, logit_yr5,
                    year1_risk, year2_risk, year3_risk, year4_risk, year5_risk,
                    l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path
                ) = sample
                label = label.cuda()
                l_cc_img = l_cc_img.squeeze(1).permute(0, 3, 1, 2)
                r_cc_img = r_cc_img.squeeze(1).permute(0, 3, 1, 2)
                l_mlo_img = l_mlo_img.squeeze(1).permute(0, 3, 1, 2)
                r_mlo_img = r_mlo_img.squeeze(1).permute(0, 3, 1, 2)
                l_cc_img = l_cc_img.to(device, non_blocking=True)
                r_cc_img = r_cc_img.to(device, non_blocking=True)
                l_mlo_img = l_mlo_img.to(device, non_blocking=True)
                r_mlo_img = r_mlo_img.to(device, non_blocking=True)

                if risk_yr == 1 and KD_label == "logit":
                    teacher_logits = logit_yr1.to(device, non_blocking=True)
                elif risk_yr == 2 and KD_label == "logit":
                    teacher_logits = logit_yr2.to(device, non_blocking=True)
                elif risk_yr == 3 and KD_label == "logit":
                    teacher_logits = logit_yr3.to(device, non_blocking=True)
                elif risk_yr == 4 and KD_label == "logit":
                    teacher_logits = logit_yr4.to(device, non_blocking=True)
                elif risk_yr == 5 and KD_label == "logit":
                    teacher_logits = logit_yr5.to(device, non_blocking=True)
                elif risk_yr == 1 and KD_label == "risk":
                    teacher_logits = year1_risk.to(device, non_blocking=True)
                elif risk_yr == 2 and KD_label == "risk":
                    teacher_logits = year2_risk.to(device, non_blocking=True)
                elif risk_yr == 3 and KD_label == "risk":
                    teacher_logits = year3_risk.to(device, non_blocking=True)
                elif risk_yr == 4 and KD_label == "risk":
                    teacher_logits = year4_risk.to(device, non_blocking=True)
                elif risk_yr == 5 and KD_label == "risk":
                    teacher_logits = year5_risk.to(device, non_blocking=True)

                output, proba_cancer, logit_cancer, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

            with torch.no_grad():
                predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
                predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
                # eids_for_epoch = eids_for_epoch + list(eid.numpy())
                eids_for_epoch.extend(to_list_of_str(eid))

            if loss_type == "KD":
                loss = loss_func(student_logit=logit_cancer, teacher_logit=teacher_logits, target=label) / batch_acc
            elif loss_type == "MSE":
                loss = loss_func(student_logit=logit_cancer, teacher_logit=teacher_logits, target=label) / batch_acc
            else:
                loss = loss_func(output, label) / batch_acc
            # print(loss)
            avg_loss_vals.append(loss.item())
            loss.backward()
            if sample_ind % batch_acc == 0 and sample_ind > 0:

                optimizer.step()
                optimizer.zero_grad()
                if verbose:
                    print(f"loss: {total_loss_for_subbatch}", f"asym mean: {model.learned_asym_mean}",
                          f"asym std: {model.learned_asym_std}")

            total_loss += loss.item()
            total_loss_for_subbatch += loss.item()
            num_samples += 1
            subbatch_samples += 1
            start = time.time()
            if sample_ind % 100 == 0:
                subbatch_samples = 0
                total_loss_for_subbatch = 0
                torch.save(model, train_preds_dir / f'full_model_partial_epoch_{epoch}_{save_file_suffix}.pt')

        if lr_step_size is not None:
            print("Stepping")
            scheduler.step()
        print(f"Epoch {epoch} had average loss value {total_loss / num_samples}")

        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds.to_csv(train_preds_dir / f'train_preds_epoch_{epoch}_{save_file_suffix}.csv',
                         index=False)

        correct_count = 0
        num_samples = 0

        predictions_for_epoch_pos = []
        predictions_for_epoch_neg = []
        eids_for_epoch = []
        preds_for_epoch = []
        labels_for_epoch = []
        proba_cancer_for_epoch = []
        logit_cancer_for_epoch = []
        proba_mirai_1yr_for_epoch = []
        logit_mirai_1yr_for_epoch = []
        proba_mirai_2yr_for_epoch = []
        logit_mirai_2yr_for_epoch = []
        proba_mirai_3yr_for_epoch = []
        logit_mirai_3yr_for_epoch = []
        proba_mirai_4yr_for_epoch = []
        logit_mirai_4yr_for_epoch = []
        proba_mirai_5yr_for_epoch = []
        logit_mirai_5yr_for_epoch = []

        with torch.no_grad():
            start = time.time()
            if not topk_for_heatmap is None:
                print("topk weights: ", model.topk_weights)
            for sample_ind, sample in enumerate(tqdm(val_dataloader, desc="Running val set")):
                (
                    eid, label,
                    logit_yr1, logit_yr2, logit_yr3, logit_yr4, logit_yr5,
                    year1_risk, year2_risk, year3_risk, year4_risk, year5_risk,
                    l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path
                ) = sample

                l_cc_img = l_cc_img.squeeze(1).permute(0, 3, 1, 2)
                r_cc_img = r_cc_img.squeeze(1).permute(0, 3, 1, 2)
                l_mlo_img = l_mlo_img.squeeze(1).permute(0, 3, 1, 2)
                r_mlo_img = r_mlo_img.squeeze(1).permute(0, 3, 1, 2)
                l_cc_img = l_cc_img.to(device, non_blocking=True)
                r_cc_img = r_cc_img.to(device, non_blocking=True)
                l_mlo_img = l_mlo_img.to(device, non_blocking=True)
                r_mlo_img = r_mlo_img.to(device, non_blocking=True)

                label = label.cuda()

                output, proba_cancer, logit_cancer, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

                # store student predictions
                proba_cancer_for_epoch.extend(proba_cancer.detach().float().view(-1).cpu().numpy().tolist())
                logit_cancer_for_epoch.extend(logit_cancer.detach().float().view(-1).cpu().numpy().tolist())

                logit_mirai_1yr_for_epoch.extend(logit_yr1.detach().float().view(-1).cpu().numpy().tolist())
                proba_mirai_1yr_for_epoch.extend(year1_risk.detach().float().view(-1).cpu().numpy().tolist())
                logit_mirai_2yr_for_epoch.extend(logit_yr2.detach().float().view(-1).cpu().numpy().tolist())
                proba_mirai_2yr_for_epoch.extend(year2_risk.detach().float().view(-1).cpu().numpy().tolist())
                logit_mirai_3yr_for_epoch.extend(logit_yr3.detach().float().view(-1).cpu().numpy().tolist())
                proba_mirai_3yr_for_epoch.extend(year3_risk.detach().float().view(-1).cpu().numpy().tolist())
                logit_mirai_4yr_for_epoch.extend(logit_yr4.detach().float().view(-1).cpu().numpy().tolist())
                proba_mirai_4yr_for_epoch.extend(year4_risk.detach().float().view(-1).cpu().numpy().tolist())
                logit_mirai_5yr_for_epoch.extend(logit_yr5.detach().float().view(-1).cpu().numpy().tolist())
                proba_mirai_5yr_for_epoch.extend(year5_risk.detach().float().view(-1).cpu().numpy().tolist())

                predictions_for_epoch_neg += list(output[:, 0].detach().cpu().numpy())
                predictions_for_epoch_pos += list(output[:, 1].detach().cpu().numpy())
                eids_for_epoch.extend(to_list_of_str(eid))

                # (Optional) keep the simple acc/sensitivity you already print
                preds = torch.argmax(output, dim=1)
                correct_count += (preds == label).sum().item()
                num_samples += label.shape[0]
                preds_for_epoch.extend(preds.detach().cpu().numpy().tolist())
                labels_for_epoch.extend(label.detach().cpu().numpy().tolist())

                start = time.time()

        # Existing prints
        print(f"Epoch {epoch} had average val accuracy {correct_count / num_samples:.4f}")

        # ----- NEW: AUROC per horizon (1–5 yr) using the SAME student probability -----
        # Build a table of predictions for this epoch
        preds_df = pd.DataFrame({
            "exam_id": eids_for_epoch,
            "proba_student": proba_cancer_for_epoch
        })

        # Pull horizon labels from the validation dataframe (must contain these columns)
        needed_cols = ["exam_id",
                       "cancer1yr_updated", "cancer2yr_updated", "cancer3yr_updated",
                       "cancer4yr_updated", "cancer5yr_updated"]
        missing = [c for c in needed_cols if c not in val_df.columns]
        if missing:
            print(f"[WARN] Missing columns in val_df for AUROC computation: {missing}")
        else:
            labels_df = val_df[needed_cols].copy()
            merged = preds_df.merge(labels_df, on="exam_id", how="left")

            aurocs_by_horizon = {}
            for n in range(1, 6):
                col = f"cancer{n}yr_updated"
                # Keep rows that have valid binary labels (0/1)
                mask = merged[col].isin([0, 1])
                y_true_n = merged.loc[mask, col].astype(int).values
                y_score_n = merged.loc[mask, "proba_student"].astype(float).values
                try:
                    # Need at least one positive and one negative
                    if (y_true_n == 1).any() and (y_true_n == 0).any():
                        au_n = roc_auc_score(y_true_n, y_score_n)
                        aurocs_by_horizon[n] = au_n
                    else:
                        aurocs_by_horizon[n] = float("nan")
                except ValueError:
                    aurocs_by_horizon[n] = float("nan")

            # Pretty print AUROCs
            msg = " | ".join([f"{k}y AUC: {v:.4f}" if np.isfinite(v) else f"{k}y AUC: n/a"
                              for k, v in aurocs_by_horizon.items()])
            print(f"Epoch {epoch} AUROCs — {msg}")

        # Save per-exam predictions (kept your existing fields + student scores)
        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds[f'proba_pred_cancer_{risk_yr}'] = proba_cancer_for_epoch
        cur_preds[f'logit_pred_cancer_{risk_yr}'] = logit_cancer_for_epoch
        # These MIRAI fields are left here if you still want them in the CSV; remove if not used
        cur_preds[f'logit_MIRAI_1yr'] = logit_mirai_1yr_for_epoch
        cur_preds[f'proba_MIRAI_1yr'] = proba_mirai_1yr_for_epoch
        cur_preds[f'logit_MIRAI_2yr'] = logit_mirai_2yr_for_epoch
        cur_preds[f'proba_MIRAI_2yr'] = proba_mirai_2yr_for_epoch
        cur_preds[f'logit_MIRAI_3yr'] = logit_mirai_3yr_for_epoch
        cur_preds[f'proba_MIRAI_3yr'] = proba_mirai_3yr_for_epoch
        cur_preds[f'logit_MIRAI_4yr'] = logit_mirai_4yr_for_epoch
        cur_preds[f'proba_MIRAI_4yr'] = proba_mirai_4yr_for_epoch
        cur_preds[f'logit_MIRAI_5yr'] = logit_mirai_5yr_for_epoch
        cur_preds[f'proba_MIRAI_5yr'] = proba_mirai_5yr_for_epoch
        cur_preds['labels_for_epoch'] = labels_for_epoch
        cur_preds.to_csv(train_preds_dir / f'validation_preds_epoch_{epoch}_{save_file_suffix}_risk_yr_{risk_yr}.csv',
                         index=False)

        torch.save(model, train_preds_dir / f'full_model_epoch_{epoch}_{save_file_suffix}_risk_yr_{risk_yr}.pt')
        val_accs.append(correct_count / num_samples)


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
