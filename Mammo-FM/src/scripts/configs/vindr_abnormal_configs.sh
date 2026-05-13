#!/bin/bash

DATASET_NAME="VINDR"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png"
CSV_FILE="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds_abnormal.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="abnormal"
EXTRA_ARGS="--label 'abnormal' --n_folds 1"
