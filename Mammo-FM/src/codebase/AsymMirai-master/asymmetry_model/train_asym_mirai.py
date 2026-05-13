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
         save_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out",
         num_epochs=30,
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
         linear_only_epochs=[]):
    # Setting the primary cuda device to be the first listed device
    torch.cuda.set_device(0)
    # torch.cuda.set_device(device_ids[0])
    save_dir = Path(save_dir)
    train_preds_dir = save_dir / dataset / arch / f"loss_{loss_type}" / f"risk_yr_{risk_yr}_KD_col_{KD_label}" / "training_preds"
    os.makedirs(train_preds_dir, exist_ok=True)
    print("Saving to directory:", train_preds_dir)
    val_accs = []
    avg_loss_vals = []

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
    val_df = df[df["split"] == "val"]
    val_dataset = MiraiMetadataset(val_df, resizer=resize_and_normalize, mode="val",
                                   align_images=align_images,
                                   oversample_cancer_rate=oversample_cancer_rate,
                                   multiple_pairs_per_exam=multiple_pairs_per_exam)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=min(max_workers, batch_size))

    train_df = df[df["split"] == "train"]
    train_dataset = MiraiMetadataset(train_df, resizer=resize_and_normalize, mode="training",
                                     align_images=align_images,
                                     oversample_cancer_rate=oversample_cancer_rate,
                                     multiple_pairs_per_exam=multiple_pairs_per_exam)
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
    if model.lower() == "mirai":
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

        # print(model)
        # print(xxx)
    elif model.lower() == "mammo_clip":
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

    if oversample_cancer_rate is not None:
        loss_func = torch.nn.CrossEntropyLoss()
    elif loss_type == "KD":
        eps = 1e-8
        pos_weight_value = neg_count / (pos_count + eps)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        loss_func = KnowledgeDistillationLoss(pos_weight=pos_weight)
    elif loss_type == "MSE":
        eps = 1e-8
        pos_weight_value = neg_count / (pos_count + eps)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        loss_func = MSELoss(pos_weight=pos_weight)
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
                eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
                label = label.cuda()

                l_cc_img = l_cc_img.to(device, non_blocking=True)
                r_cc_img = r_cc_img.to(device, non_blocking=True)
                l_mlo_img = l_mlo_img.to(device, non_blocking=True)
                r_mlo_img = r_mlo_img.to(device, non_blocking=True)

                output, proba_cancer, logit_cancer, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

            with torch.no_grad():
                predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
                predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
                # eids_for_epoch = eids_for_epoch + list(eid.numpy())
                eids_for_epoch.extend(to_list_of_str(eid))

            loss = loss_func(output, label) / batch_acc
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
            # print(num_samples)
            # print(total_loss)
            # print(xxx)
            if sample_ind % 100 == 0:
                # print(f"Saving for sample index {sample_ind}")
                # print(f"Last 10 samples had average loss value {total_loss_for_subbatch / subbatch_samples}")
                # print("cc_stretch_params", model.cc_stretch_params)
                # print("mlo_stretch_params", model.mlo_stretch_params)
                subbatch_samples = 0
                total_loss_for_subbatch = 0
                torch.save(model, train_preds_dir / f'full_model_partial_epoch_{epoch}_{save_file_suffix}.pt')

        if lr_step_size is not None:
            print("Stepping")
            scheduler.step()
        print(f"Epoch {epoch} had average loss value {total_loss / num_samples}")
        # print("cc_stretch_params", model.cc_stretch_params)
        # print("mlo_stretch_params", model.mlo_stretch_params)

        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds.to_csv(train_preds_dir / f'train_preds_epoch_{epoch}_{save_file_suffix}.csv',
                         index=False)

        correct_count = 0
        num_samples = 0
        teacher_logits = None

        predictions_for_epoch_pos = []
        predictions_for_epoch_neg = []
        eids_for_epoch = []
        preds_for_epoch = []
        labels_for_epoch = []
        proba_cancer_for_epoch = []
        logit_cancer_for_epoch = []
        proba_mirai_for_epoch = []
        logit_mirai_for_epoch = []

        with torch.no_grad():
            start = time.time()
            if not topk_for_heatmap is None:
                print("topk weights: ", model.topk_weights)
            for sample in val_dataloader:
                if verbose:
                    print(f"Loaded next sample in {time.time() - start} seconds")
                eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
                l_cc_img, r_cc_img, l_mlo_img, r_mlo_img = l_cc_img.cuda(), r_cc_img.cuda(), l_mlo_img.cuda(), r_mlo_img.cuda()
                label = label.cuda()
                l_cc_img = l_cc_img.to(device, non_blocking=True)
                r_cc_img = r_cc_img.to(device, non_blocking=True)
                l_mlo_img = l_mlo_img.to(device, non_blocking=True)
                r_mlo_img = r_mlo_img.to(device, non_blocking=True)

                output, proba_cancer, logit_cancer, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)
                preds = torch.argmax(output, dim=1)
                pc = proba_cancer
                lc = logit_cancer
                proba_cancer_for_epoch.extend(pc.detach().float().view(-1).cpu().numpy().tolist())
                logit_cancer_for_epoch.extend(lc.detach().float().view(-1).cpu().numpy().tolist())


                predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
                predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
                # eids_for_epoch = eids_for_epoch + list(eid.numpy())
                eids_for_epoch.extend(to_list_of_str(eid))
                if verbose:
                    print(preds)

                correct_count += label[preds == label].shape[0]
                num_samples += label.shape[0]
                preds_for_epoch.extend(preds.detach().cpu().numpy().tolist())
                labels_for_epoch.extend(label.detach().cpu().numpy().tolist())

                start = time.time()

        print(f"Epoch {epoch} had average val accuracy {correct_count / num_samples}")
        y_true = np.array(labels_for_epoch, dtype=np.int32)
        y_score = np.array(proba_cancer_for_epoch, dtype=np.float32)  # P(pos)
        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = 0.5
        print(f"Epoch {epoch} val AUROC (pos class): {auroc:.4f}")

        y_pred = np.array(preds_for_epoch, dtype=np.int32)  # argmax labels
        pos_mask = (y_true == 1)
        if pos_mask.any():
            pos_acc = (y_pred[pos_mask] == 1).mean()  # TP / (TP+FN)
        else:
            pos_acc = float('nan')
        print(f"Epoch {epoch} val positive-sample accuracy (sensitivity): {pos_acc:.4f}  (n_pos={pos_mask.sum()})")

        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds[f'proba_cancer_{risk_yr}'] = proba_cancer_for_epoch
        cur_preds[f'logit_cancer_{risk_yr}'] = logit_cancer_for_epoch
        # cur_preds[f'logit_MIRAI_{risk_yr}'] = logit_mirai_for_epoch
        # cur_preds[f'proba_MIRAI_{risk_yr}'] = proba_mirai_for_epoch
        cur_preds[f'labels_for_epoch'] = labels_for_epoch
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
