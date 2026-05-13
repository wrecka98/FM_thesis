import pandas as pd
import torch
from torch.utils.data import DataLoader
from mirai_metadataset import MiraiMetadataset
from mammo_clip_metadataset import Mammo_CLIPMetadataset
from embed_explore import resize_and_normalize
import numpy as np

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


def connected_cluster_bbox(hm2d: torch.Tensor, threshold: float = 0.02):
    """
    hm2d: [Hh, Wh] heatmap tensor (CPU or CUDA)
    threshold: absolute tolerance from the global max (same as official code)
    returns:
      ymin, xmin, ymax, xmax  (integers in heatmap coordinates, inclusive)
      cy, cx                  (centroid in heatmap coords, floats)
    """
    device = hm2d.device
    Hh, Wh = hm2d.shape

    # argmax
    am = torch.argmax(hm2d)
    am_h = (am // Wh).item()
    am_w = (am %  Wh).item()

    # mask of candidates within threshold of the max
    max_val = hm2d[am_h, am_w]
    mask = (max_val - hm2d) <= threshold  # True where within threshold

    # BFS/stack to collect 8-connected component containing the max
    visited = torch.zeros_like(mask, dtype=torch.bool, device=device)
    stack = [(am_h, am_w)]
    ys, xs = [], []

    while stack:
        y, x = stack.pop()
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        ys.append(y); xs.append(x)
        # 8-neighborhood
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < Hh and 0 <= nx < Wh and not visited[ny, nx]:
                    if mask[ny, nx]:
                        stack.append((ny, nx))

    # Fallback: if something odd happens, at least return the max cell
    if len(ys) == 0:
        ys, xs = [am_h], [am_w]

    ys_t = torch.tensor(ys, dtype=torch.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32)

    ymin = int(torch.min(ys_t).item())
    ymax = int(torch.max(ys_t).item())
    xmin = int(torch.min(xs_t).item())
    xmax = int(torch.max(xs_t).item())

    cy = float(torch.mean(ys_t).item())
    cx = float(torch.mean(xs_t).item())

    return ymin, xmin, ymax, xmax, cy, cx

def heatmap_box_to_image(ymin, xmin, ymax, xmax, Hh, Wh, Hi, Wi):
    """
    Map a heatmap-space box [ymin..ymax, xmin..xmax] to image pixels.
    Returns (x0, y0, x1, y1) in image coords (int), where (x1,y1) is exclusive bound.
    """
    cell_h = Hi / float(Hh)
    cell_w = Wi / float(Wh)

    # left/top corner of top-left cell
    y0 = int((ymin * cell_h) // 1)  # floor
    x0 = int((xmin * cell_w) // 1)

    # right/bottom edge of bottom-right cell (exclusive)
    y1 = int(((ymax + 1) * cell_h + 0.9999) // 1)  # ceil
    x1 = int(((xmax + 1) * cell_w + 0.9999) // 1)

    # clamp
    y0 = max(0, min(y0, Hi - 1))
    x0 = max(0, min(x0, Wi - 1))
    y1 = max(1, min(y1, Hi))
    x1 = max(1, min(x1, Wi))

    return x0, y0, x1, y1


def mirror_bbox_to_right(x0, y0, x1, y1, Wi):
    """
    Mirror a left-image bbox horizontally to the right image coordinates,
    consistent with AsymMirai (right aligned to left).
    """
    rx0 = max(0, min(Wi - x1, Wi - 1))
    rx1 = max(1, min(Wi - x0, Wi))
    return rx0, y0, rx1, y1

def get_centroid_activation(array, threshold=0.02):
    print(array.shape)
    h, w = array.shape
    am = torch.argmax(array)
    am_h, am_w = am // w, am % w

    # Grab all the locations at which activation is within threshold of the max
    candidate_locations = list(((array[am_h, am_w] - array) <= threshold).nonzero())

    # First, we're going to grab all the locations that are contiguous with the max
    added_new = True
    contiguous_w_max = [torch.tensor([am_h, am_w])]

    while added_new:
        added_new = False
        to_move = []
        for cl_ind, cl in enumerate(candidate_locations):
            for contig_ind, contig in enumerate(contiguous_w_max):
                if abs(cl[0] - contig[0]) <= 1 and abs(cl[1] - contig[1]) <= 1:
                    if abs(cl[0] - contig[0]) == 0 and abs(cl[1] - contig[1]) == 0:
                        continue
                    if cl_ind not in to_move:
                        to_move.append(cl_ind)
                    added_new = True

        for index in sorted(to_move, reverse=True):
            contiguous_w_max.append(candidate_locations[index])
            del candidate_locations[index]

    # This is a bit of a hack, but the true max gets double counted,
    # so this removes the first time we counted it
    if len(contiguous_w_max) > 1:
        del contiguous_w_max[0]

    h_mean, w_mean = 0.0, 0.0
    for cm in contiguous_w_max:
        h_mean += cm[0].item()
        w_mean += cm[1].item()
    print(h_mean, w_mean, contiguous_w_max)
    h_mean /= len(contiguous_w_max)
    w_mean /= len(contiguous_w_max)

    return (h_mean, w_mean)


use_crop = False
multiple_pairs_per_exam = False
device = 0
torch.cuda.set_device(device)

# model_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1/training_preds/full_model_epoch_0_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt"
# model_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/training_preds/full_model_epoch_1_4_26_alt_ablation_flex_width_5_matrix_learned_dist.pt"
# model_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds/full_model_epoch_4_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt"
model_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds/full_model_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt"

# model_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_MSE/risk_yr_1_KD_col_logit/training_preds/full_model_epoch_1_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt"

data_file = '/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv'
dataset = "bu"
align_images = False
label_col = "cancer1yr_updated"
model = torch.load(model_path, map_location=torch.device(f'cuda:{device}'))
risk_yr = 1
KD_label = "logit"
batch_size = 1
val_tfm = None

df = pd.read_csv(data_file)
val_df = df[df["split"] == "val"]
val_df = val_df[val_df[label_col] == 1]
val_df = val_df[val_df["exam_id"]=="E135880"]


# val_dataset = Mammo_CLIPMetadataset(val_df, resizer=resize_and_normalize,
#                                mode='val', align_images=False,
#                                multiple_pairs_per_exam=multiple_pairs_per_exam)
val_dataset = Mammo_CLIPMetadataset(val_df, dataset=dataset, transforms=val_tfm, align_images=align_images,
                                    multiple_pairs_per_exam=False, label_col=label_col)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=min(10, batch_size))

with torch.no_grad():
    predictions_for_epoch_pos = []
    predictions_for_epoch_neg = []
    eids_for_epoch = []
    y_argmins_mlo_for_epoch = []
    x_argmins_mlo_for_epoch = []
    y_argmins_cc_for_epoch = []
    x_argmins_cc_for_epoch = []
    correct_count = 0
    num_samples = 0
    left_cc_x_min, left_cc_x_max, left_cc_y_min, left_cc_y_max = [], [], [], []
    right_cc_x_min, right_cc_x_max, right_cc_y_min, right_cc_y_max = [], [], [], []

    left_mlo_x_min, left_mlo_x_max, left_mlo_y_min, left_mlo_y_max = [], [], [], []
    right_mlo_x_min, right_mlo_x_max, right_mlo_y_min, right_mlo_y_max = [], [], [], []

    for index, sample in enumerate(val_dataloader):
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

            output, proba_cancer, logit_cancer, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

        preds = torch.argmax(output, dim=1)

        predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
        predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
        eids_for_epoch.extend(to_list_of_str(eid))
        print(f"CC heatmap")
        print(other[0])
        print(f"======"*20)
        print(f"MLO heatmap")
        print(other[1])
        print(proba_cancer)
        print(logit_cancer)

        cc_meta, mlo_meta = other[0], other[1]

        # CC
        hm_cc = cc_meta['heatmap']  # [B, Hh_cc, Wh_cc]
        Hh_cc, Wh_cc = hm_cc.shape[-2], hm_cc.shape[-1]
        Hi_cc, Wi_cc = l_cc_img.shape[-2], l_cc_img.shape[-1]

        # per-sample (assuming batch_size == 1 here for clarity; if >1, loop over i)
        i = 0
        ymin_cc, xmin_cc, ymax_cc, xmax_cc, cy_cc, cx_cc = connected_cluster_bbox(hm_cc[i], threshold=0.02)
        LCC_x0, LCC_y0, LCC_x1, LCC_y1 = heatmap_box_to_image(ymin_cc, xmin_cc, ymax_cc, xmax_cc, Hh_cc, Wh_cc, Hi_cc,
                                                              Wi_cc)
        RCC_x0, RCC_y0, RCC_x1, RCC_y1 = mirror_bbox_to_right(LCC_x0, LCC_y0, LCC_x1, LCC_y1, Wi_cc)

        # MLO
        hm_m = mlo_meta['heatmap']  # [B, Hh_m, Wh_m]
        Hh_m, Wh_m = hm_m.shape[-2], hm_m.shape[-1]
        Hi_m, Wi_m = l_mlo_img.shape[-2], l_mlo_img.shape[-1]

        ymin_m, xmin_m, ymax_m, xmax_m, cy_m, cx_m = connected_cluster_bbox(hm_m[i], threshold=0.02)
        LMLO_x0, LMLO_y0, LMLO_x1, LMLO_y1 = heatmap_box_to_image(ymin_m, xmin_m, ymax_m, xmax_m, Hh_m, Wh_m, Hi_m,
                                                                  Wi_m)
        RMLO_x0, RMLO_y0, RMLO_x1, RMLO_y1 = mirror_bbox_to_right(LMLO_x0, LMLO_y0, LMLO_x1, LMLO_y1, Wi_m)

        # now you can save all 16 columns:
        left_cc_x_min.append(LCC_x0)
        left_cc_x_max.append(LCC_x1)
        left_cc_y_min.append(LCC_y0)
        left_cc_y_max.append(LCC_y1)

        right_cc_x_min.append(RCC_x0)
        right_cc_x_max.append(RCC_x1)
        right_cc_y_min.append(RCC_y0)
        right_cc_y_max.append(RCC_y1)

        left_mlo_x_min.append(LMLO_x0)
        left_mlo_x_max.append(LMLO_x1)
        left_mlo_y_min.append(LMLO_y0)
        left_mlo_y_max.append(LMLO_y1)

        right_mlo_x_min.append(RMLO_x0)
        right_mlo_x_max.append(RMLO_x1)
        right_mlo_y_min.append(RMLO_y0)
        right_mlo_y_max.append(RMLO_y1)

        print("LCC")
        print(f"x0: {LCC_x0}, y0: {LCC_y0}, x1: {LCC_x1}, y1: {LCC_y1}")
        x = cc_meta['x_argmin'][0]
        y = cc_meta['y_argmin'][0]

        print(f"Centroid: x: {x}, y: {y}")

        print("RCC")
        print(f"x0: {RCC_x0}, y0: {RCC_y0}, x1: {RCC_x1}, y1: {RCC_y1}")

        print("LMLO")
        print(f"x0: {LMLO_x0}, y0: {LMLO_y0}, x1: {LMLO_x1}, y1: {LMLO_y1}")
        print(f"Centroid: x: {mlo_meta['x_argmin'][0]}, y: {mlo_meta['y_argmin'][0]}")

        print("RMLO")
        print(f"x0: {RMLO_x0}, y0: {RMLO_y0}, x1: {RMLO_x1}, y1: {RMLO_y1}")



        # stash in your dataframe
        # cur_preds['cc_x0'], cur_preds['cc_y0'], cur_preds['cc_x1'], cur_preds['cc_y1'] = zip(*cc_bboxes)
        # cur_preds['mlo_x0'], cur_preds['mlo_y0'], cur_preds['mlo_x1'], cur_preds['mlo_y1'] = zip(*mlo_bboxes)

        print(l_cc_path)
        print(r_cc_path)
        print(l_mlo_path)
        print(r_mlo_path)
        # print("cc_bboxes")
        # print(cc_bboxes)
        # print("mlo_bboxes")
        # print(mlo_bboxes)
        print(f"cc x_argmin: {other[0]['x_argmin']}")
        print(f"cc y_argmin: {other[0]['y_argmin']}")


        print(xxxx)
        y_argmins_cc_for_epoch = y_argmins_cc_for_epoch + list(other[0]['y_argmin'])  # .cpu().detach().numpy())
        x_argmins_cc_for_epoch = x_argmins_cc_for_epoch + list(other[0]['x_argmin'])  # .cpu().detach().numpy())
        y_argmins_mlo_for_epoch = y_argmins_mlo_for_epoch + list(other[1]['y_argmin'])  # .cpu().detach().numpy())
        x_argmins_mlo_for_epoch = x_argmins_mlo_for_epoch + list(other[1]['x_argmin'])  # .cpu().detach().numpy())

        correct_count += label[preds == label].shape[0]
        num_samples += label.shape[0]

        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos

        # (existing argmin debug if you want)
        cur_preds['y_argmin_cc'] = y_argmins_cc_for_epoch
        cur_preds['x_argmin_cc'] = x_argmins_cc_for_epoch
        cur_preds['y_argmin_mlo'] = y_argmins_mlo_for_epoch
        cur_preds['x_argmin_mlo'] = x_argmins_mlo_for_epoch

        # --- NEW: 16 bbox columns
        cur_preds['left_cc_x_min'] = left_cc_x_min
        cur_preds['left_cc_x_max'] = left_cc_x_max
        cur_preds['left_cc_y_min'] = left_cc_y_min
        cur_preds['left_cc_y_max'] = left_cc_y_max

        cur_preds['right_cc_x_min'] = right_cc_x_min
        cur_preds['right_cc_x_max'] = right_cc_x_max
        cur_preds['right_cc_y_min'] = right_cc_y_min
        cur_preds['right_cc_y_max'] = right_cc_y_max

        cur_preds['left_mlo_x_min'] = left_mlo_x_min
        cur_preds['left_mlo_x_max'] = left_mlo_x_max
        cur_preds['left_mlo_y_min'] = left_mlo_y_min
        cur_preds['left_mlo_y_max'] = left_mlo_y_max

        cur_preds['right_mlo_x_min'] = right_mlo_x_min
        cur_preds['right_mlo_x_max'] = right_mlo_x_max
        cur_preds['right_mlo_y_min'] = right_mlo_y_min
        cur_preds['right_mlo_y_max'] = right_mlo_y_max

        if index % 5 == 0:
            cur_preds.to_csv(f'./training_preds/validation_predictions.csv', index=False)

    cur_preds = pd.DataFrame()
    cur_preds['exam_id'] = eids_for_epoch
    cur_preds['prediction_neg'] = predictions_for_epoch_neg
    cur_preds['prediction_pos'] = predictions_for_epoch_pos
    cur_preds['y_argmin_cc'] = y_argmins_cc_for_epoch
    cur_preds['x_argmin_cc'] = x_argmins_cc_for_epoch
    cur_preds['y_argmin_mlo'] = y_argmins_mlo_for_epoch
    cur_preds['x_argmin_mlo'] = x_argmins_mlo_for_epoch
    cur_preds.to_csv(f'./training_preds/validation_predictions.csv', index=False)
