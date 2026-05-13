#!/bin/bash -l

#$ -N bu        # Give job a name
#$ -o /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/scc_logs/bu_$JOB_ID_$JOB_NAME.out       # File name for the stdout output of the job.

#$ -P batmanlab     # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00  # Specify the hard time limit for the job
#$ -pe omp 8        # Number of cores
#$ -l gpus=2        # Number of GPUs
#$ -l gpu_c=8.0     # GPU Compute capacity

#$ -j y            # Merge the error and output streams into a single file
#$ -m bea          # The batch system sends an email to you. The possible values are – when the job begins (b), ends (e), is aborted (a), is suspended (s),
                   # or never (n) – default.


pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/scc_logs/bu_$CURRENT.out

echo "breast-clip"

module load miniconda/23.1.0
module load miniconda
module load python3/3.8
conda activate /restricted/projectnb/batmanlab/shawn24/breast_clip_rtx_6000

######################################################## Frozen Encoder ########################################################
# baseline asymirai (Vedant - Don't run)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_asym.py


python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_cal_mean.py

# mammo-clip 1 yr
# stage 1
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 1 \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage1 \
    --loss_type "KD" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

# stage 2
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 1 \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage2 \
    --loss_type "CE" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

# IID Eval BU (single institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_1yr_predictions_epoch38.csv" \
    --label_col "cancer1yr_updated"

# OOD Eval RSNA  (single institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_RSNA_image_level_mammo-clip_risk.csv" \
    --dataset_name "RSNA" \
    --output_file "validation_1yr_predictions_epoch38_RSNA.csv" \
    --label_col "cancer"


# IID Eval BU  (multi institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_6_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_1yr_predictions_epoch38.csv" \
    --label_col "cancer1yr_updated"

# OOD Eval RSNA  (multi institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_6_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_RSNA_image_level_mammo-clip_risk.csv" \
    --dataset_name "RSNA" \
    --output_file "validation_1yr_predictions_epoch38_RSNA.csv" \
    --label_col "cancer"


# mammo-clip 2 yr
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 2  \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage1 \
    --loss_type "KD" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 2 \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage2 \
    --loss_type "CE" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

# mammo-clip 3 yr
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 3  \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage1 \
    --loss_type "KD" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 3 \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage2 \
    --loss_type "CE" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

# mammo-clip 4 yr
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 4  \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage1 \
    --loss_type "KD" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 4 \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage2 \
    --loss_type "CE" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"

# mammo-clip 5 yr
# target col: logit_yr5
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 5  \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage1 \
    --loss_type "KD" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"


# target col: logit_yr5_mean
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 5  \
    --data_file  "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk_5yr_mean.csv" \
    --arch "breast_clip_det_b5_period_n_lp" \
    --use_mean_risk "y" \
    --batch_size 16 \
    --training_stage stage1 \
    --loss_type "KD" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"


# IID Eval BU (single institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/stage1/loss_KD_unweighted/risk_yr_5_KD_col_logit_use_mean_risk_False/training_preds" \
    --model_path "full_model_epoch_20_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_5.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_1yr_predictions_epoch20.csv" \
    --label_col "cancer1yr_updated"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/stage1/loss_KD_unweighted/risk_yr_5_KD_col_logit_use_mean_risk_False/training_preds" \
    --model_path "full_model_epoch_20_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_5.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_2yr_predictions_epoch20.csv" \
    --label_col "cancer2yr_updated"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/stage1/loss_KD_unweighted/risk_yr_5_KD_col_logit_use_mean_risk_False/training_preds" \
    --model_path "full_model_epoch_20_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_5.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_3yr_predictions_epoch20.csv" \
    --label_col "cancer3yr_updated"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/stage1/loss_KD_unweighted/risk_yr_5_KD_col_logit_use_mean_risk_False/training_preds" \
    --model_path "full_model_epoch_20_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_5.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_4yr_predictions_epoch20.csv" \
    --label_col "cancer4yr_updated"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/stage1/loss_KD_unweighted/risk_yr_5_KD_col_logit_use_mean_risk_False/training_preds" \
    --model_path "full_model_epoch_20_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_5.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_5yr_predictions_epoch20.csv" \
    --label_col "cancer5yr_updated"

# OOD Eval RSNA  (single institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_RSNA_image_level_mammo-clip_risk.csv" \
    --dataset_name "RSNA" \
    --output_file "validation_1yr_predictions_epoch38_RSNA.csv" \
    --label_col "cancer"



#/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk_5yr_mean.csv

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 5 \
    --arch "breast_clip_det_b5_period_n_lp" \
    --training_stage stage2 \
    --loss_type "CE" \
    --train_backbone "n" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"


# IID Eval BU  (multi institution)
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --dataset_name "bu" \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_5_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_6_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv" \
    --output_file "validation_1yr_predictions_epoch38.csv" \
    --label_col "cancer1yr_updated"

######################################################## Frozen Encoder ########################################################


######################################################## Finetune Encoder ########################################################
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_bu_KD.py --risk_yr 1 \
    --arch "breast_clip_det_b5_period_n_ft" \
    --batch_size 1 \
    --train_backbone "y" \
    --use_addon_layers "n" \
    --use_stretch_matrix "y" \
    --flexible_asymmetry "y" \
    --align_images "n" \
    --use_all_training_data "n" \
    --multiple_pairs_per_exam "n"
######################################################## Finetune Encoder ########################################################





