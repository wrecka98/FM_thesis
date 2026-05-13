import torch
import argparse
from train import main as train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AsymMirai Model with Mammo-CLIP")

    # --- Core Training Arguments ---
    parser.add_argument('--risk_yr', type=int, default=2, help='Risk year for prediction (e.g., 1, 2).')
    parser.add_argument('--label_col', type=str, default="cancer", help='Name of the label column in the data file.')
    parser.add_argument('--loss_type', type=str, default="KD", help='Type of loss function (e.g., "KD", "CE").')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=30, help='Training batch size.')

    # --- Path and File Arguments ---
    parser.add_argument(
        '--data_file',
        default="/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_BUMC_image_level_mammo-clip_risk.csv",
        type=str, help='Full path to the merged dataframe CSV.'
    )
    parser.add_argument(
        '--chk_pt',
        type=str,
        default="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/img_text_embed_bu_mayo_clip/b5_detector_n_modernbert_2048/checkpoints/fold_0/Mammo-FM_BatmanlabTrained_CLIP.tar",
        help='Full path to the pretrained checkpoint file.'
    )

    parser.add_argument(
        '--save_dir', type=str, default="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out_multi_institution_chkpt",
        help='Directory to save the trained model.')
    parser.add_argument('--save_file_suffix', type=str, default="4_26_alt_ablation_flex_width_5_matrix_learned_dist",
                        help='Suffix for the saved model file.')

    # --- Model Architecture Arguments ---
    parser.add_argument('--model', type=str, default="mammo_clip", help='Model name.')
    parser.add_argument('--training_stage', type=str, default="stage1", help='Model name.')
    parser.add_argument('--stage2_model_path', type=str, default="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/codebase/AsymMirai-master/asymmetry_model/out/bu/breast_clip_det_b5_period_n_lp/loss_KD/risk_yr_1_KD_col_logit/training_preds/full_model_epoch_38_4_26_alt_ablation_flex_width_5_matrix_learned_dist_risk_yr_1.pt", help='Model name.')

    parser.add_argument('--arch', type=str, default="breast_clip_det_b5_period_n_lp", help='Model architecture.')
    parser.add_argument('--latent_h', type=int, default=5, help='Latent space height.')
    parser.add_argument('--latent_w', type=int, default=5, help='Latent space width.')
    # parser.add_argument('--initial_asym_mean', type=int, default=249, help='Initial asymmetry mean.')
    # parser.add_argument('--initial_asym_std', type=int, default=66, help='Initial asymmetry standard deviation.')
    parser.add_argument('--initial_asym_mean', type=int, default=364, help='Initial asymmetry mean.')
    parser.add_argument('--initial_asym_std', type=int, default=163, help='Initial asymmetry standard deviation.')
    parser.add_argument('--train_backbone', type=str, default="n", help='Training backbone.')
    parser.add_argument('--use_mean_risk', type=str, default="n", help='')

    # --- Boolean Flags ---
    parser.add_argument('--use_addon_layers', type=str, default="n", help='Flag to use addon layers.')
    parser.add_argument('--use_stretch_matrix', type=str, default="y",
                        help='Flag to use the stretch matrix.')
    parser.add_argument('--flexible_asymmetry', type=str, default="y", help='Flag for flexible asymmetry.')
    parser.add_argument('--align_images', type=str, default="n", help='Flag to align images.')
    parser.add_argument('--use_all_training_data', type=str, default="n", help='Flag to use all training data.')
    parser.add_argument('--multiple_pairs_per_exam', type=str, default="n",
                        help='Flag to use multiple image pairs per exam.')

    # --- Data and System Arguments ---
    parser.add_argument('--dataset', type=str, default="bu", help='Dataset identifier (e.g., "bu").')
    parser.add_argument('--KD_label', type=str, default="logit", help='Knowledge distillation label type.')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='List of GPU device IDs to use.')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of data loader workers.')
    parser.add_argument('--oversample_cancer_rate', type=float, default=None, help='Rate to oversample cancer cases.')
    parser.add_argument('--topk_for_heatmap', type=int, default=None, help='Top K value for heatmap generation.')

    args = parser.parse_args()
    torch.cuda.set_device(args.device_ids[0])
    torch.manual_seed(0)

    args.train_backbone = True if args.train_backbone.lower() == "y" else False
    args.use_addon_layers = True if args.use_addon_layers.lower() == "y" else False
    args.use_stretch_matrix = True if args.use_stretch_matrix.lower() == "y" else False
    args.flexible_asymmetry = True if args.flexible_asymmetry.lower() == "y" else False
    args.use_mean_risk = True if args.use_mean_risk.lower() == "y" else False
    args.align_images = True if args.align_images.lower() == "y" else False
    args.use_all_training_data = True if args.use_all_training_data.lower() == "y" else False
    args.multiple_pairs_per_exam = True if args.multiple_pairs_per_exam.lower() == "y" else False

    print("Starting training with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  --{arg}: {value}")

    # print("===================================================="
    #
    # print(args.train_backbone, type(args.train_backbone))
    # print(args.use_addon_layers, type(args.use_addon_layers))
    # print(args.use_stretch_matrix, type(args.use_stretch_matrix))
    # print(args.flexible_asymmetry, type(args.flexible_asymmetry))
    # print(args.align_images, type(args.align_images))
    # print(args.use_all_training_data, type(args.use_all_training_data))
    # print(args.multiple_pairs_per_exam, type(args.multiple_pairs_per_exam))
    # print(xxxx)
    # print("====================================================")

    # Call the original training function from your library
    train_model(
        device_ids=args.device_ids,
        training_stage=args.training_stage,
        stage2_model_path=args.stage2_model_path,
        save_dir=args.save_dir,
        risk_yr=args.risk_yr,
        loss_type=args.loss_type,
        model=args.model,
        label_col=args.label_col,
        KD_label=args.KD_label,
        dataset=args.dataset,
        data_file=args.data_file,
        arch=args.arch,
        train_backbone=args.train_backbone,
        chk_pt=args.chk_pt,
        batch_size=args.batch_size,
        lr=args.lr,
        use_addon_layers=args.use_addon_layers,
        use_stretch_matrix=args.use_stretch_matrix,
        initial_asym_mean=args.initial_asym_mean,
        initial_asym_std=args.initial_asym_std,
        flexible_asymmetry=args.flexible_asymmetry,
        max_workers=args.max_workers,
        align_images=args.align_images,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        save_file_suffix=args.save_file_suffix,
        use_all_training_data=args.use_all_training_data,
        oversample_cancer_rate=args.oversample_cancer_rate,
        topk_for_heatmap=args.topk_for_heatmap,
        multiple_pairs_per_exam=args.multiple_pairs_per_exam,
        use_mean_risk=args.use_mean_risk
    )
