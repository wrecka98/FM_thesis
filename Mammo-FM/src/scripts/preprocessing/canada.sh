#!/bin/bash -l

#$ -P batmanlab
#$ -j y
#$ -N preprocess_NLBS
#$ -o preprocess.qlog

#$ -l h_rt=48:00:00
#$ -pe omp 8

module load miniconda
conda activate /restricted/projectnb/batmanlab/shawn24/python_3_9
python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/preprocessing/preprocess_canada.py --metadata_csv "/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data/nlbs_image_labels.csv"
