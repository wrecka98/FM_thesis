DATA_DIR="/export/home/rstanciu/ZGT_Mammo-FM_format"
IMG_DIR="images_png"
CLIP_CKPT="/export/home/rstanciu/Downloads/Mammo-FM_BatmanlabTrained_CLIP.tar"
RUN_ROOT="/export/home/rstanciu/FM_thesis_Razvan/Mammo-FM/Mammo-FM_runs"

for FOLD in 0 1 2 3 4
do
  CSV_FILE="detection_mammofm_fold${FOLD}.csv"

  python3 ./src/codebase/train_detector.py \
    --data-dir "$DATA_DIR" \
    --img-dir "$IMG_DIR" \
    --csv-file "$CSV_FILE" \
    --cur-fold "$FOLD" \
    --dataset "custom" \
    --arch "breast_clip_det_b5" \
    --epochs 40 \
    --batch-size 2 \
    --freeze_backbone "y" \
    --data_frac 1.0 \
    --concepts "Mass" \
    --clip_chk_pt_path "$CLIP_CKPT" \
    --print-freq 500 \
    --log-freq 100 \
    --running-interactive "n" \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --score-threshold 0.1 \
    --n_folds 5 \
    --tensorboard-path "$RUN_ROOT/fold${FOLD}/tb" \
    --checkpoints "$RUN_ROOT/fold${FOLD}/checkpoints" \
    --output_path "$RUN_ROOT/fold${FOLD}/out"
done