#!/bin/bash

DATASET_NAME="RSNA"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="RSNA_Cancer_Detection/train_images_png"
CSV_FILE="RSNA_Cancer_Detection/train_folds.csv"
TASK_TYPE="classification"
#LABELS=("cancer" "density")
LABEL="cancer"
EXTRA_ARGS="--label 'cancer' --n_folds 1"
WEIGHTED_BCE="n"



