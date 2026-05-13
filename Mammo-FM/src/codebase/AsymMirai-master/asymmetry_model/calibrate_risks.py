import pickle
import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# import sklearn.svm.classes
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.metrics import roc_curve

callibrator_snapshot = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/snapshots/callibrators/MIRAI_FULL_PRED_RF.callibrator.p"
callibrator = pickle.load(open(callibrator_snapshot, 'rb'))
print(callibrator)

csv = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv"
df = pd.read_csv(csv)
df = df[df["split"] == "val"]
risk_yr = 1
print(df.columns)
print(df.shape)
exam_ids_1 = df[df["cancer1yr_updated"] == 1]["exam_id"].tolist()

group_cols = ['patient_id', 'exam_id']
agg_cols = [
    'cancer_registry', '1_year_risk', '2_year_risk', '3_year_risk', '4_year_risk', '5_year_risk',
    'logit_yr1', 'logit_yr2', 'logit_yr3', 'logit_yr4', 'logit_yr5',
    'cancer1yr_updated', 'cancer2yr_updated', 'cancer3yr_updated', 'cancer4yr_updated', 'cancer5yr_updated'
]

df_grouped = df.groupby(group_cols, as_index=False)[agg_cols].first()
print(df_grouped.shape)

result_dir = Path(
    "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds")
# csv = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds/validation_preds_epoch_1_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.csv"
csv = result_dir / "validation_preds_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.csv"
# csv = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/training_preds/validation_preds_epoch_1_4_26_alt_ablation_flex_width_5_matrix_learned_dist.csv"
df = pd.read_csv(csv)
print(df.columns)
print(df.shape)

df['exam_id'] = df['exam_id'].astype(str)
df_grouped['exam_id'] = df_grouped['exam_id'].astype(str)
df_grouped['patient_id'] = df_grouped['patient_id'].astype(str)

pid_map = df_grouped[['exam_id', 'patient_id']].drop_duplicates()
df_with_pid = df.merge(pid_map, on='exam_id', how='left', validate='m:1')  # many rows in df → one in pid_map

# 3) Final merge on (patient_id, exam_id)
out = df_with_pid.merge(
    df_grouped,
    on=['patient_id', 'exam_id'],
    how='left',  # use 'inner' if you only want rows present in both
    suffixes=('', '_grp')  # avoids accidental column name collisions
)

print(out.columns)
print(out[f"proba_cancer_{risk_yr}"].values.shape)
# scores = out[f"proba_cancer_{risk_yr}"].values.astype(float).reshape(-1, 1)
scores = out[f"proba_cancer_{risk_yr}"].tolist()
risk_yr1 = np.array([])

for pred in scores:
    print(pred)
    new_risk_value = callibrator[risk_yr - 1].predict_proba([[pred]])[0, 1]
    risk_yr1 = np.append(risk_yr1, new_risk_value)


print(risk_yr1)

out["proba_cancer_1_calibrated"] = risk_yr1
print(out.columns)
out.to_csv(
    result_dir / "validation_preds_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1_calibrated.csv",
    index=False
)
