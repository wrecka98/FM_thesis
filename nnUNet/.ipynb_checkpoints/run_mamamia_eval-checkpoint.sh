#!/bin/bash

set -e

# ============================================================
# Configuration
# ============================================================

DATASET="Dataset001_Mammography"
CONFIG="2d"

INPUT="/home/jovyan/thesis_project/datasets/INbreast/INbreast Release 1.0/INbreast_nnUNet/imagesTs"

GT="/home/jovyan/thesis_project/datasets/INbreast/INbreast Release 1.0/INbreast_nnUNet/labelsTs"

OUTPUT_ROOT="/home/jovyan/thesis_project/FM_thesis/nnUNet/test_predictions/OOD_nnUNet_preds/INbreast"

MODEL_DIR="$nnUNet_results/Dataset001_Mammography/nnUNetTrainer__nnUNetPlans__2d"

DATASET_JSON="$MODEL_DIR/dataset.json"
PLANS_JSON="$MODEL_DIR/plans.json"


# ============================================================
# Sanity checks
# ============================================================

echo "============================================================"
echo "INbreast OOD evaluation"
echo "============================================================"

echo "Input:        $INPUT"
echo "Ground truth: $GT"
echo "Output:       $OUTPUT_ROOT"
echo "Model:        $MODEL_DIR"
echo


if [ ! -d "$INPUT" ]; then
    echo "ERROR: Input directory does not exist:"
    echo "$INPUT"
    exit 1
fi

if [ ! -d "$GT" ]; then
    echo "ERROR: Ground-truth directory does not exist:"
    echo "$GT"
    exit 1
fi

if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: dataset.json not found:"
    echo "$DATASET_JSON"
    exit 1
fi

if [ ! -f "$PLANS_JSON" ]; then
    echo "ERROR: plans.json not found:"
    echo "$PLANS_JSON"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"


# ============================================================
# Run folds
# ============================================================

for FOLD in 0 1 2 3 4
do

    echo
    echo "============================================================"
    echo "FOLD $FOLD"
    echo "============================================================"

    OUTPUT="$OUTPUT_ROOT/fold_$FOLD"
    SUMMARY="$OUTPUT/summary.json"

    mkdir -p "$OUTPUT"


    # --------------------------------------------------------
    # Skip fold completely if evaluation already exists
    # --------------------------------------------------------

    if [ -f "$SUMMARY" ]; then
        echo "Fold $FOLD already has evaluation results:"
        echo "$SUMMARY"
        echo
        echo "Skipping fold $FOLD."
        continue
    fi


    # --------------------------------------------------------
    # Check whether predictions already exist
    # --------------------------------------------------------

    NUM_PREDICTIONS=$(find "$OUTPUT" -maxdepth 1 -type f -name "*.png" | wc -l)
    NUM_INPUTS=$(find "$INPUT" -maxdepth 1 -type f -name "*_0000.png" | wc -l)

    echo "Input cases:          $NUM_INPUTS"
    echo "Existing predictions: $NUM_PREDICTIONS"


    if [ "$NUM_PREDICTIONS" -eq "$NUM_INPUTS" ] && [ "$NUM_INPUTS" -gt 0 ]; then

        echo
        echo "All predictions already exist for fold $FOLD."
        echo "Skipping prediction and proceeding to evaluation."

    else

        echo
        echo "Running prediction for fold $FOLD..."
        echo

        nnUNetv2_predict \
            -i "$INPUT" \
            -o "$OUTPUT" \
            -d "$DATASET" \
            -c "$CONFIG" \
            -f "$FOLD"

    fi


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    echo
    echo "Evaluating fold $FOLD..."
    echo

    nnUNetv2_evaluate_folder \
        "$GT" \
        "$OUTPUT" \
        -djfile "$DATASET_JSON" \
        -pfile "$PLANS_JSON" \
        -o "$SUMMARY"


    echo
    echo "Fold $FOLD complete."
    echo "Results:"
    echo "$SUMMARY"

done


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "ALL FOLDS COMPLETE"
echo "============================================================"
echo

echo "Results are located in:"
echo "$OUTPUT_ROOT"
echo