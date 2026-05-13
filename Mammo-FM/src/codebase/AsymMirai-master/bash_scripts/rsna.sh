#!/bin/bash -l

#$ -N rsna        # Give job a name
#$ -o /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/scc_logs/rsna_$JOB_ID_$JOB_NAME.out       # File name for the stdout output of the job.

#$ -P batmanlab     # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00  # Specify the hard time limit for the job
#$ -pe omp 8        # Number of cores
#$ -l gpus=1        # Number of GPUs
#$ -l gpu_c=8.0     # GPU Compute capacity

#$ -j y            # Merge the error and output streams into a single file
#$ -m bea          # The batch system sends an email to you. The possible values are – when the job begins (b), ends (e), is aborted (a), is suspended (s),
                   # or never (n) – default.


pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/scc_logs/rsna_$CURRENT.out

echo "breast-clip"

module load miniconda/23.1.0
module load miniconda
module load python3/3.8
conda activate /restricted/projectnb/batmanlab/shawn24/breast_clip_rtx_6000

# cuda test
python /restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/cuda.py

#python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_rsna.py >$slurm_output_train1


python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_train_rsna_KD.py >$slurm_output_train1


python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/run_eval_bu_KD.py \
    --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds" \
    --model_path "full_model_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt" \
    --data_file "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_RSNA_image_level_mammo-clip_risk.csv" \
    --dataset_name "bu" \
    --output_file "validation_1yr_predictions_epoch38.csv" \
    --label_col "cancer"