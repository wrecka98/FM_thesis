#!/bin/bash

### USER INPUT SECTION — set your checkpoint here:


CKPT_KEY="Mammo-FM"
CLIP_CKPT="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/Mammo-FM/b5_detector_n_modernbert_2048/checkpoints/fold_0/Mammo-FM_BatmanlabTrained_CLIP.tar"

### Define global log base path
LOG_BASE="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scc_logs/breast"
SUB_DIR="epoch4/batmanlab_trained_weightedBCE_n"
source "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scripts/configs/vindr_configs.sh"

#VINDR_LABELS=("Mass" "Suspicious_Calcification" "density" "Architectural_Distortion" "Focal_Asymmetry" "Asymmetry")
VINDR_LABELS=("Mass" "Suspicious_Calcification" "Asymmetry")
#VINDR_LABELS=("Mass" "Suspicious_Calcification" "density")
VINDR_DATA_FRACS=("1.0" "0.8" "0.5" "0.25" "0.1")
#VINDR_DATA_FRACS=("1.0")
VINDR_ARCHS=("breast_clip_det_b5_period_n_lp" "breast_clip_det_b5_period_n_ft")
#VINDR_ARCHS=("breast_clip_det_b5_period_n_lp")

for LABEL in "${VINDR_LABELS[@]}"; do
  for ARCH in "${VINDR_ARCHS[@]}"; do
    for FRAC in "${VINDR_DATA_FRACS[@]}"; do
#      if [[ "$ARCH" == *"ft"* && "$FRAC" != "1.0" ]]; then
#        continue
#      fi
      export CLIP_CKPT="$CLIP_CKPT"
      export ARCH="$ARCH"
      export DATA_FRAC="$FRAC"
      export DATASET_NAME="$DATASET_NAME"
      export DATA_DIR="$DATA_DIR"
      export IMG_DIR="$IMG_DIR"
      export CSV_FILE="$CSV_FILE"
      export LABEL="$LABEL"
      export WEIGHTED_BCE="$WEIGHTED_BCE"
      export EXTRA_ARGS="--label '$LABEL' --n_folds 1"

      echo $ARCH
      echo $DATASET_NAME
      echo $DATA_DIR
      echo $IMG_DIR
      echo $CSV_FILE
      echo $DATA_FRAC
      echo $LABEL
      echo $CLIP_CKPT
      echo $WEIGHTED_BCE

      timestamp=$(date +"%Y-%m-%d-%H-%M-%S-%N")
      jobname="${ARCH//[^a-zA-Z0-9]/_}_${DATASET_NAME}_${LABEL}_${FRAC}_${CKPT_KEY}_${timestamp}"
      logdir="${LOG_BASE}/classification_${DATASET_NAME,,}/${CKPT_KEY}/${SUB_DIR}"
      mkdir -p "$logdir"

      echo "Submitting VinDr: $jobname"
      qsub -j y -N "$jobname" \
      -o "${logdir}/${jobname}.qlog" \
      -v ARCH,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,CLIP_CKPT,LABEL,WEIGHTED_BCE,EXTRA_ARGS \
      "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scripts/qsub_templates/base_classifier.qsub"

    done
  done
done
