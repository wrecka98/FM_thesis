import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import sys
from tqdm import tqdm
from typing import List, Tuple, Any
from pathlib import Path

# Assuming these custom dataset classes are in your environment
# and accessible in the same directory or Python path.
from mammo_clip_metadataset import Mammo_CLIPMetadataset


def to_list_of_str(x: Any) -> List[str]:
    """Converts various data types to a list of strings."""
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    if isinstance(x, torch.Tensor):
        return [str(v.item()) for v in x]
    return [str(x)]


def main(args):
    """
    Main function to run the model inference, process heatmaps,
    and save the results to a CSV file.
    """
    # 1. Setup Device
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    # 2. Load Model
    print(f"Loading model from: {output_dir / args.model_path}")
    try:
        model = torch.load(output_dir / args.model_path, map_location=device)
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    # 3. Prepare Dataloader
    print("Loading and preparing data...")
    df = pd.read_csv(args.data_file)
    print(df.shape)
    print(df.columns)
    # if args.dataset_name.lower() == "bu":
    #     val_df = df[df["split"] == "val"]
    # elif args.dataset_name.lower() == "rsna":
    #     val_df = df

    val_df = df
    print(val_df.shape)
    # print(xxx)

    if args.target_exam_id:
        val_df = val_df[val_df["exam_id"] == args.target_exam_id]
        print(f"Filtering for single exam_id: {args.target_exam_id}")

    val_dataset = Mammo_CLIPMetadataset(
        val_df, dataset=args.dataset_name, transforms=None,
        align_images=args.align_images, multiple_pairs_per_exam=False,
        label_col=args.label_col
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=args.shuffle_data,
        num_workers=min(args.num_workers, args.batch_size)
    )
    print(f"Data ready. Found {len(val_dataset)} samples.")

    if args.dataset_name.lower() == "rsna":
        output_dir = output_dir / "OOD_RSNA"
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.dataset_name.lower() == "bu":
        output_dir = output_dir / "ID_BU_Clare_discordant"
        output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Run Inference Loop
    results = []
    print("Starting inference...")
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataloader, desc="Processing Batches")):
            (
                eid, label,
                logit_yr1, logit_yr2, logit_yr3, logit_yr4, logit_yr5,
                year1_risk, year2_risk, year3_risk, year4_risk, year5_risk,
                l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path
            ) = sample

            l_cc_img = l_cc_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)
            r_cc_img = r_cc_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)
            l_mlo_img = l_mlo_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)
            r_mlo_img = r_mlo_img.squeeze(1).permute(0, 3, 1, 2).to(device, non_blocking=True)

            output, proba_cancer, _, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

            cc_meta, mlo_meta = other[0], other[1]
            hm_cc, hm_mlo = cc_meta['heatmap'], mlo_meta['heatmap']
            exam_id_str = to_list_of_str(eid)[0]

            exam_save_dir = output_dir / args.label_col / exam_id_str
            exam_save_dir.mkdir(parents=True, exist_ok=True)

            # Save the heatmap tensors. Move to CPU first as a best practice.
            torch.save(cc_meta, exam_save_dir / 'cc_heatmap.pt')
            torch.save(mlo_meta, exam_save_dir / 'mlo_heatmap.pt')

            sample_result = {
                'exam_id': to_list_of_str(eid)[0],
                'prediction_neg': output[0, 0].item(),
                'prediction_pos': output[0, 1].item(),
                'y_argmin_cc': cc_meta['y_argmin'][0],
                'x_argmin_cc': cc_meta['x_argmin'][0],
                'y_argmin_mlo': mlo_meta['y_argmin'][0],
                'x_argmin_mlo': mlo_meta['x_argmin'][0],
                'l_cc_path': to_list_of_str(l_cc_path)[0],
                'r_cc_path': to_list_of_str(r_cc_path)[0],
                'l_mlo_path': to_list_of_str(l_mlo_path)[0],
                'r_mlo_path': to_list_of_str(r_mlo_path)[0],
                'proba_cancer': proba_cancer[0].item(),
                'mirai_logit_yr1': logit_yr1[0].item(),
                'mirai_logit_yr2': logit_yr2[0].item(),
                'mirai_logit_yr3': logit_yr3[0].item(),
                'mirai_logit_yr4': logit_yr4[0].item(),
                'mirai_logit_yr5': logit_yr5[0].item(),
                'mirai_year1_risk': year1_risk[0].item(),
                'mirai_year2_risk': year2_risk[0].item(),
                'mirai_year3_risk': year3_risk[0].item(),
                'mirai_year4_risk': year4_risk[0].item(),
                'mirai_year5_risk': year5_risk[0].item(),
                'target': label[0].item(),
            }
            results.append(sample_result)

            if (i + 1) % args.save_interval == 0:
                pd.DataFrame(results).to_csv(args.output_file, index=False)

    # 5. Save Final Results
    if results:
        pd.DataFrame(results).to_csv(output_dir / args.output_file, index=False)
        print(f"\nInference complete. All results saved to {output_dir / args.output_file}")
        print(f"Save file: {output_dir}")
    else:
        print("No samples were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on mammography images, process attention heatmaps to find bounding boxes, and save the results."
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='dir'
    )
    # --- Path Arguments ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pt file.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the input CSV data file.')
    parser.add_argument('--output_file', type=str, default='./training_preds/validation_predictions.csv',
                        help='Path to save the output CSV file.')

    # --- Dataset and Model Arguments ---
    parser.add_argument('--dataset_name', type=str, default="bu", help='Name of the dataset (e.g., "bu").')
    parser.add_argument('--label_col', type=str, default="cancer1yr_updated",
                        help='Column name for the ground truth labels.')
    parser.add_argument('--align_images', action='store_true',
                        help='Set this flag to align images during data loading.')

    # --- Inference Arguments ---
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID to use for inference.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference. The code is optimized for a batch size of 1.')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the DataLoader.')
    parser.add_argument('--disable_shuffle', action='store_false', dest='shuffle_data',
                        help='Disable shuffling of the validation data.')

    # --- Debug and Logging Arguments ---
    parser.add_argument('--target_exam_id', type=str, default=None,
                        help='Specify a single exam_id to process for debugging (e.g., "E135880").')
    parser.add_argument('--save_interval', type=int, default=5, help='Save intermediate results every N batches.')

    cli_args = parser.parse_args()
    main(cli_args)
