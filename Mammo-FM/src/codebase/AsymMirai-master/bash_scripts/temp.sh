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
    --output_file "validation_5yr_predictions_epoch20.csv" \
    --label_col "cancer5yr_updated"