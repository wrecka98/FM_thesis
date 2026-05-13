# Mammo-FM: Breast-specific foundational model for Integrated Mammographic Diagnosis, Prognosis, and Reporting

[![Paper](https://img.shields.io/badge/Paper-9cf)](https://arxiv.org/pdf/2512.00198)
[![Hugging Face](https://img.shields.io/badge/Checkpoints-Hugging%20Face-yellow)](https://huggingface.co/batmanLab/Mammo-FM)
[![VinDr PNG](https://img.shields.io/badge/VinDr%20Mammogram%20png%20images-lightblue)](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)
[![RSNA PNG](https://img.shields.io/badge/RSNA%20Mammogram%20png%20images-lightblue)](https://huggingface.co/datasets/shawn24/RSNA-Breast-Cancer-detection-from-mammograms-PNG)
![Visitor Badge](https://visitor-badge.laobi.icu/badge?page_id=batmanlab.FM&right_color=%23FFA500)

#### ⚠️ WARNING: We are updating this codebase, so if anyone finds any error, please contact us. We will try to resolve it ASAP.

#### ⚠️ WARNING: There is a plethora of pre-processing settings available for RSNA and VinDr Mammo datasets. We recommend using the pre-processing discussed in the following sections. We are not responsible for any discrepancies in the results due to different pre-processing settings. If you use the VinDr png dataset uploaded in kaggle, it is fully pre-processed. Else you can use the pre-processing scripts provided in the following sections.

#### ⚠️ WARNING: If you find the `punkt_tab` error, run the following command in the python environment:

```python
import nltk

nltk.download('punkt_tab')
```

## FAQ

We follow our previous code [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP) strictly. We will update the
pretraining setup shortly. This code is for validating Mammo-FM checkpoints for diagnostic (e.g., downstream
classification on linear probe and full-finetuning) performance only. Zero-shot, prognostic and report generation will
be uploaded in coming weeks.

## Table of Contents

## Environment Setup

Use [environment.yml](https://github.com/batmanlab/Mammo-FM/blob/main/environment.yml) to setup the environment.

```bash
git clone git@github.com:batmanlab/Mammo-FM.git
cd Mammo-FM
conda env create --name Mammo-FM -f environment.yml
conda activate Mammo-FM
```

Mammo-FM is implemented with following specification:

* Python version: 3.9+
* PyTorch version: 2.2.2
* CUDA version: 11.8

## Data Download

Download the original versions VinDr and RSNA from the links for downstream evaluations:

- [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr](https://vindr.ai/datasets/mammo)

For the PNG images converted from the original Dicom images, as mentioned in the preprocessing steps in the paper, refer
to the following links:

- [VinDr](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)

To preprocess the dicom images directly, follow the instructions in the next section. If you downloaded the PNG images,
skip the preprocessing steps.

## Pre-processing images

### Convert to png: RSNA

```bash
python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="test" \
  --base_folder="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection"
```

### convert to png: VinDr

```bash
python ./src/preprocessing/preprocess_image_to_png_vindr.py \
  --phase="test" \
  --base_folder="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
```

## Final dataset directory structures

### Image+Text pretraining dataset

```bash
.
├── list_tree_files.sh
├── img_text_dicom_consolidated_final_folds_BIRADS_num_1_report.csv
├── clip_pretrain_100.csv
└── DICOM/images_png_CC_MLO/
    ├── Patient_100/
    │   ├── 1.png
    │   ├── 2.png
    └── Patient_200/
        ├── 3.png
        ├── 4.png
        ├── 53.png
        ├── 6.png
        └── 7.png
        

```

### VinDr

```bash
.
├── breast-level_annotations.csv
├── finding_annotations.csv
├── vindr_detection_v1_folds.csv 
├── clip_vindr_final.csv
└── images_png/
    ├── c7811f4575c1229ad4a7606de49ea68f/
    │   ├── 9eb4650a2b630e44074c403f6127c5a1.png
    │   ├── cc3fdc5d733a671f3000e20838e192d9.png
    │   ├── 181fd193d3b785dc9faafdaa8e1695fc.png
    │   └── 55eb5ea616abacd225e584ffc8be57da.png
    └── a1dd219b28806fc295fac20ceb147870/
        ├── 887cdcc99ebed66bd062ada6c8210152.png
        ├── 36f2921a2ac19eba7420c591c4c07ae4.png
        ├── 12dc17dfd9d30ea7c0c1ccb33a505085.png
        └── e22e4f297b4c82279e7b78a98417a6cd.png
```

### RSNA

```bash
.
├── train_folds.csv
├── train_images_png/
    ├── 59549/
    │   ├── 1154694388.png
    │   ├── 1192817932.png
    │   ├── 1979035704.png
    │   ├── 2022274082.png
    │   ├── 431013616.png
    │   ├── 457600713.png
    │   ├── 78005871.png
    │   └── 856162422.png
    └── 28242/
        ├── 1966298736.png
        ├── 233201459.png
        ├── 349787619.png
        └── 98615814.png


```

## Mammo-FM checkpoints

Following are the pre-training checkpoints of Mammo-FM:

| Model architecture               | Checkpoints (Hugging Face)                                                                                                   |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Last epoch on BU, UPMC and EMBED | [Mammo-FM_BatmanlabTrained_CLIP.tar](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_BatmanlabTrained_CLIP.tar) |
| Last epoch on Mayo               | [Mammo-FM_ASU_Trained_CLIP.tar](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_ASU_Trained_CLIP.tar)           |

## Evaluation

### Zero-shot evaluation of Mammo-FM

[To be uploaded]

## Downstream run scripts (bash)

For downstream experiments we keep bash entrypoints
in [src/scripts](https://github.com/batmanlab/Mammo-FM/tree/main/src/scripts) named
`<dataset>_<task>_run.sh`. These are intended for running the same training commands from a shell or job scheduler with
dataset-specific defaults. Current examples include `cmmd_classification_run.sh`, `nl_breast_classification_run.sh`,
`vindr_abnormal_classification_run.sh`, `vindr_classification_run.sh`, and `vindr_detection_run.sh`.

The blocks below are single-run examples that mirror those scripts. Update paths for your environment and pick the
checkpoint you want to evaluate. `_lp` and `_ft` denotes linear probing and full finetuning respectively for `--arch`
parameter.

### RSNA classification

```bash
DATASET_NAME="RSNA"
CLIP_CKPT="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/Mammo-FM/b5_detector_n_modernbert_2048/checkpoints/fold_0/Mammo-FM_BatmanlabTrained.tar"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="RSNA_Cancer_Detection/train_images_png"
CSV_FILE="RSNA_Cancer_Detection/train_folds.csv"
EXTRA_ARGS="--label 'cancer' --n_folds 1"
LABEL="density"

CLIP_CKPT=""
# Note: if you set CLIP_CKPT multiple times, the last assignment wins.

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/train_classifier.py \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --arch "breast_clip_det_b5_period_n_lp" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 2 \
  --batch-size 2 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'y' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

### VinDr classification (abnormal)

```bash
DATASET_NAME="VINDR"
CLIP_CKPT=""
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png"
CSV_FILE="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds_abnormal.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="abnormal"
EXTRA_ARGS="--label 'abnormal' --n_folds 1"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/train_classifier.py \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --arch "breast_clip_det_b5_period_n_lp" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 2 \
  --batch-size 2 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'n' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

### CMMD classification (cancer)

```bash
DATASET_NAME="CMMD"
CLIP_CKPT=""
DATA_DIR="/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset"
IMG_DIR="cmmd_png"
CSV_FILE="cmmd_png/merged_final_cmmd.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="cancer"
EXTRA_ARGS="--label 'cancer' --n_folds 1"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/train_classifier.py \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --inference_mode "y" \
  --arch "breast_clip_det_b5_period_n_ft" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 2 \
  --batch-size 8 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'n' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

### NL-Breast classification (cancer)

```bash
DATASET_NAME="NLBreast"
CLIP_CKPT=""
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data"
IMG_DIR="NLBS_Data_png_v1"
CSV_FILE="nlbs_image_labels.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="cancer"
EXTRA_ARGS="--label 'cancer' --n_folds 1"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/train_classifier.py \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --arch "breast_clip_det_b5_period_n_ft" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 2 \
  --batch-size 8 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'y' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

### VinDr detection (Mass)

```bash
DATASET_NAME="ViNDr"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png"
CSV_FILE="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv"
TASK_TYPE="Detection"


CLIP_CKPT=""
CLIP_CKPT="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/Mammo-FM/b5_detector_n_modernbert_2048/checkpoints/fold_0/Mammo-FM_BatmanlabTrained.tar"
LABEL="Mass"
ARCH="breast_clip_det_b5"

python /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/train_detector.py \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --dataset "$DATASET_NAME" \
  --arch "$ARCH" \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone "n" \
  --data_frac 1.0 \
  --concepts "$LABEL" \
  --clip_chk_pt "$CLIP_CKPT" \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive 'n' \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2
```

## Citation

**Mammo-FM**

```bibtex
@article{ghosh2025mammo,
  title={Mammo-FM: Breast-specific foundational model for Integrated Mammographic Diagnosis, Prognosis, and Reporting},
  author={Ghosh, Shantanu and Joshi, Vedant Parthesh and Syed, Rayan and Kassem, Aya and Varshney, Abhishek and Basak, Payel and Dai, Weicheng and Gichoya, Judy Wawira and Trivedi, Hari M and Banerjee, Imon and others},
  journal={arXiv preprint arXiv:2512.00198},
  year={2025}
}
```

**Mammo-CLIP (MICCAI 2024)**

```bibtex
@InProceedings{10.1007/978-3-031-72390-2_59,
author="Ghosh, Shantanu
and Poynton, Clare B.
and Visweswaran, Shyam
and Batmanghelich, Kayhan",
editor="Linguraru, Marius George
and Dou, Qi
and Feragen, Aasa
and Giannarou, Stamatia
and Glocker, Ben
and Lekadir, Karim
and Schnabel, Julia A.",
title="Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="632--642",
abstract="The lack of large and diverse training data on Computer-Aided Diagnosis (CAD) in breast cancer detection has been one of the concerns that impedes the adoption of the system. Recently, pre-training with large-scale image text datasets via Vision-Language models (VLM) (e.g., CLIP) partially addresses the issue of robustness and data efficiency in computer vision (CV). This paper proposes Mammo-CLIP, the first VLM pre-trained on a substantial amount of screening mammogram-report pairs, addressing the challenges of dataset diversity and size. Our experiments on two public datasets demonstrate strong performance in classifying and localizing various mammographic attributes crucial for breast cancer detection, showcasing data efficiency and robustness similar to CLIP in CV. We also propose Mammo-FActOR, a novel feature attribution method, to provide spatial interpretation of representation with sentence-level granularity within mammography reports. Code is available publicly: https://github.com/batmanlab/Mammo-CLIP.",
isbn="978-3-031-72390-2"
}
```

## License and copyright

The source code is Licensed under [Apache License 2.0](https://github.com/batmanlab/Mammo-FM/blob/main/LICENSE.txt).
Model weights is Licensed
under [Custom Academic License for Model Weights](https://github.com/batmanlab/Mammo-FM/blob/main/LICENSE_WEIGHTS.txt).

Copyright © [Batman Lab](https://www.batman-lab.com/), 2026

## Contact

For any queries, contact [Shantanu Ghosh](https://shantanu-ai.github.io/) (email: **shawn24@bu.edu**)

## Contributing

Did you try Mammo-FM on other datasets containing 2D-Mammograms and want to report the results? Feel free to send
a [pull request](https://github.com/shantanu-ai/deep-learning-resources/pulls).
