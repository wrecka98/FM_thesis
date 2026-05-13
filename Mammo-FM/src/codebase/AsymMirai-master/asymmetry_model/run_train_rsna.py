from train import main
import torchvision
import torch

# Train AsymMirai and save the resulting model
torch.cuda.set_device(0)
torch.manual_seed(0)
# mammo-clip RSNA
# main(device_ids=[0],
#      model="mammo_clip",
#      label_col="cancer",
#      dataset="rsna",
#      data_file="/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_RSNA_image_level_mammo-clip_risk.csv",
#      arch="breast_clip_det_b5_period_n_lp",
#      chk_pt="/restricted/projectnb/batmanlab/ayak/PhD/Breast-CLIP/src/codebase/outputs/Mammo-FM/b5_detector_no_downstream_no_birads_prompt_period_no_cancer_no_ethnicity_n_modernbert_2048/checkpoints/fold_0/model-best.tar",
#      batch_size=30,
#      lr=0.005,
#      use_addon_layers=False,
#      use_stretch_matrix=True,
#      initial_asym_mean=2000,  # *20,
#      initial_asym_std=300,
#      flexible_asymmetry=True,
#      max_workers=8,
#      align_images=False,
#      latent_h=5,
#      latent_w=5,
#      save_file_suffix="4_26_alt_ablation_flex_width_5_matrix_learned_dist",
#      use_all_training_data=False,
#      oversample_cancer_rate=None,
#      topk_for_heatmap=None,
#      multiple_pairs_per_exam=False)

main(device_ids=[2],
     risk_yr=1,
     loss_type="CE",
     model="mammo_clip",
     label_col="cancer",
     dataset="rsna",
     data_file="/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/MIRAI/Mirai/results/merged_dataframe_MIRAI_RSNA_image_level_mammo-clip_risk.csv",
     arch="breast_clip_det_b5_period_n_lp",
     chk_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/Mammo-FM/b5_detector_n_modernbert_2048/checkpoints/fold_0/Mammo-FM_BatmanlabTrained_CLIP.tar",
     batch_size=30,
     lr=0.005,
     use_addon_layers=False,
     use_stretch_matrix=True,
     initial_asym_mean=356,  # *20,
     initial_asym_std=166,
     flexible_asymmetry=True,
     max_workers=8,
     align_images=False,
     latent_h=5,
     latent_w=5,
     save_file_suffix="4_26_alt_ablation_flex_width_5_matrix_learned_dist",
     use_all_training_data=False,
     oversample_cancer_rate=None,
     topk_for_heatmap=None,
     multiple_pairs_per_exam=False)

# torch.Size([3, 3, 1520, 912])
# torch.Size([3, 3, 1520, 912])
# torch.Size([3, 512, 48, 29])
# torch.Size([3, 512, 48, 29])

# initial_asym_mean = 2000,  # *20,
# initial_asym_std = 300,
