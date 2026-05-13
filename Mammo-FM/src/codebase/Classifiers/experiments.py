import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .models.breast_clip_classifier import BreastClipClassifier
from Datasets.dataset_utils import get_dataloader
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from metrics import all_classification_metrics, compute_opt_thres, compute_accuracy_np_array
from utils import seed_all, AverageMeter, timeSince
from sklearn.metrics import confusion_matrix


def _save_valid_predictions(args, predictions):
    pred_proba_col = f"{args.label}_pred_proba"
    pred_bin_col = f"{args.label}_prediction_bin"
    args.valid_folds[pred_proba_col] = predictions

    if (
            args.label.lower() == "density" or
            args.label.lower() == "birads" or
            args.label.lower() == "tissueden" or
            args.label.lower() == "bcsc_race_eth_idx_clf" or
            args.label.lower() == "race"
    ):
        print(f"Skipping binarization for multi-class label: {args.label}")
    else:
        th = compute_opt_thres(
            args.valid_folds[args.label].values,
            y_pred=args.valid_folds[pred_proba_col].values
        )
        args.valid_folds[pred_bin_col] = (args.valid_folds[pred_proba_col] >= th).astype(int)

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"seed_{args.seed}_fold{args.cur_fold}_{args.label}_valid_predictions.csv"
    args.valid_folds.to_csv(output_dir / output_name, index=False)
    print(args.valid_folds.head(10))
    print(args.valid_folds.columns)
    print(f"Saved validation csv w/ predictions to {output_dir / output_name}")


def do_experiments(args, device):
    if 'efficientnetv2' in args.arch:
        args.model_base_name = 'efficientv2_s'
    elif 'efficientnet_b5_ns' in args.arch:
        args.model_base_name = 'efficientnetb5'
    else:
        args.model_base_name = args.arch

    args.data_dir = Path(args.data_dir)
    oof_df = pd.DataFrame()
    for fold in range(args.start_fold, args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)

        if args.dataset.lower() == "rsna":
            args.df = pd.read_csv(args.data_dir / args.csv_file)
            print(f"df shape: {args.df.shape}")
            print(args.df.columns)
            if args.label.lower() == "density":
                args.df = args.df[args.df["density"].notna()]
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                args.df["density"] = args.df["density"].map(label_map)

            args.df = args.df.fillna(0)
            args.train_folds = args.df[
                (args.df['fold'] == 1) | (args.df['fold'] == 2)].reset_index(drop=True)
            args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)
            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")

        elif args.dataset.lower() == "vindr":
            args.df = pd.read_csv(args.data_dir / args.csv_file)
            args.df = args.df.fillna(0)
            print(f"df shape: {args.df.shape}")
            print(args.df.columns)
            args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
            args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)

        elif args.dataset.lower() == "cmmd" or args.dataset.lower() == "nlbreast":
            args.df = pd.read_csv(args.data_dir / args.csv_file)
            args.df = args.df.fillna(0)
            print(f"df shape: {args.df.shape}")
            print(args.df.columns)
            args.train_folds = args.df[
                (args.df['fold'] == 1) | (args.df['fold'] == 2)].reset_index(drop=True)
            args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)


        elif args.dataset.lower() == "embed":
            if args.label.lower() == "abnormal" or args.label.lower() == "abnormal-bal":
                args.train_folds = pd.read_csv(args.data_dir / args.csv_file.format("train"))
                args.valid_folds = pd.read_csv(args.data_dir / args.csv_file.format("test"))
                args.train_folds = args.train_folds.fillna(0)
                args.valid_folds = args.valid_folds.fillna(0)
            elif (
                    args.label.lower() == "cancer" or
                    args.label.lower() == "mass" or
                    args.label.lower() == "calc" or
                    args.label.lower() == "arch_distortion" or
                    args.label.lower() == "tissueden"
            ):
                args.df = pd.read_csv(args.data_dir / args.csv_file)
                args.df = args.df.fillna(0)
                args.df["tissueden"] = args.df["tissueden"].astype(int)
                args.df.loc[args.df["tissueden"] > 0, "tissueden"] -= 1
                args.train_folds = args.df[(args.df['fold'] == 1) | (args.df['fold'] == 2)].reset_index(drop=True)
                args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)

            elif args.label.lower() == "race":
                args.train_folds = pd.read_csv(args.data_dir / args.csv_file.format("train"))
                args.valid_folds = pd.read_csv(args.data_dir / args.csv_file.format("test"))
                args.train_folds = args.train_folds[
                    args.train_folds['ETHNICITY_DESC'].notna() & (args.train_folds['ETHNICITY_DESC'] != 'Multiple')]
                args.valid_folds = args.valid_folds[
                    args.valid_folds['ETHNICITY_DESC'].notna() & (args.valid_folds['ETHNICITY_DESC'] != 'Multiple')]

                args.train_folds = args.train_folds.fillna(0)
                args.valid_folds = args.valid_folds.fillna(0)

                ethnicity_mapping = {
                    'African American  or Black': 0,
                    'Asian': 1,
                    'Caucasian or White': 2,
                    'Native Hawaiian or Other Pacific Islander': 3,
                    'Unknown, Unavailable or Unreported': 4,
                    'American Indian or Alaskan Native': 5
                }

                args.train_folds['race'] = args.train_folds['ETHNICITY_DESC'].map(ethnicity_mapping)
                args.valid_folds['race'] = args.valid_folds['ETHNICITY_DESC'].map(ethnicity_mapping)

                print("Distribution of Race train set:")
                print(args.train_folds['race'].value_counts())
                print("------")
                print("Distribution of Race val set:")
                print(args.valid_folds['race'].value_counts())

            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")
            print(args.train_folds.columns)

            args.train_folds = args.train_folds.rename(columns={'ImageLateralityFinal': 'laterality'})
            args.valid_folds = args.valid_folds.rename(columns={'ImageLateralityFinal': 'laterality'})

        if args.inference_mode == 'y':
            _oof_df = inference_loop(args, device)
        else:
            _oof_df = train_loop(args, device)

        oof_df = pd.concat([oof_df, _oof_df])

    # if args.dataset.lower() == "rsna":
    oof_df = oof_df.reset_index(drop=True)
    if args.dataset.lower() == "embed":
        oof_df_agg = oof_df[['patient_id', args.label, 'prediction']].groupby(['patient_id']).max()
    elif args.dataset.lower() == "nlbreast":
        oof_df_agg = oof_df
    else:
        oof_df_agg = oof_df[['patient_id', args.label, 'prediction', 'fold']].groupby(['patient_id']).max()

    print(oof_df_agg.head(10))
    print('================ CV ================')
    if (
            args.label.lower() == "density" or
            args.label.lower() == "tissueden" or
            args.label.lower() == "bcsc_race_eth_idx_clf" or
            args.label.lower() == "race"
    ):
        correct_predictions = (oof_df_agg[args.label] == oof_df_agg['prediction']).sum()
        total_predictions = len(oof_df_agg)
        accuracy = correct_predictions / total_predictions
        oof_df_agg[args.label] = oof_df_agg[args.label].astype(int)
        oof_df_agg['prediction'] = oof_df_agg['prediction'].astype(int)
        f1 = f1_score(oof_df_agg[args.label], oof_df_agg['prediction'], average='macro')
        print(f"Accuracy: {accuracy * 100:.4f} %, F1 Score: {f1 * 100:.4f}")
        if args.label.lower() == "bcsc_race_eth_idx_clf" or args.label.lower() == "race":
            y_true = oof_df_agg[args.label].astype(int).to_numpy()
            y_pred = oof_df_agg['prediction'].astype(int).to_numpy()
            labels = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # per-class accuracy = TP / (TP + FN) = diagonal / row sum
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            support = cm.sum(axis=1)

            print("\nPer-class accuracy (race) — label: accuracy% (support)")
            for lbl, acc, sup in zip(labels, per_class_acc, support):
                print(f"{lbl}: {acc * 100:6.2f}% (n={sup})")

            if args.label.lower() == "bcsc_race_eth_idx_clf":
                idx_to_race = {
                    0: "Asian",
                    1: "Black",
                    2: "Hispanic",
                    3: "Other",
                    4: "Unknown",
                    5: "White",
                }
            elif args.label.lower() == "race":
                idx_to_race = {
                    0: 'African American  or Black',
                    1: 'Asian',
                    2: 'Caucasian or White',
                    3: 'Native Hawaiian or Other Pacific Islander',
                    4: 'Unknown, Unavailable or Unreported',
                    5: 'American Indian or Alaskan Native'
                }

            print("\nPer-class accuracy (race) — race: accuracy% (support)")
            for lbl, acc, sup in zip(labels, per_class_acc, support):
                name = idx_to_race.get(lbl, str(lbl))
                print(f"{name:8s}: {acc * 100:6.2f}% (n={sup})")

    else:
        th = compute_opt_thres(oof_df[args.label].values, y_pred=oof_df['prediction'].values)
        oof_df['prediction_bin'] = oof_df['prediction'].apply(lambda x: 1 if x >= th else 0)
        metrics = all_classification_metrics(gt=oof_df_agg[args.label].values, pred=oof_df_agg['prediction'].values)

        oof_df_agg_cancer = oof_df_agg[oof_df_agg[args.label] == 1]
        oof_df_agg_cancer['prediction'] = oof_df_agg_cancer['prediction'].apply(lambda x: 1 if x >= th else 0)
        acc_cancer = compute_accuracy_np_array(oof_df_agg_cancer[args.label].values,
                                               oof_df_agg_cancer['prediction'].values)

        print(f"Consolidated metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print(f'Accuracy +ve {args.label} patients: {acc_cancer * 100} %')
        print('\n')
        print(oof_df.head(10))
        print(f"Results shape: {oof_df.shape}")
        print('\n')
        print(args.output_path)
        oof_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_outputs.csv', index=False)


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    args.BCE_weights = {}
    args.BCE_weights[f"fold{args.cur_fold}"] = args.train_folds[args.train_folds[args.label] == 0].shape[0] / \
                                               args.train_folds[args.train_folds[args.label] == 1].shape[0]
    print(f"args.BCE_weights: {args.BCE_weights}")
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)

    if args.clip_chk_pt_path is not None:
        ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
        if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = None
        ckpt = None
    if args.running_interactive:
        args.train_folds = args.train_folds.sample(500)
        args.valid_folds = args.valid_folds.sample(n=200)

    train_loader, valid_loader = get_dataloader(args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    model = None
    if args.label.lower() == "density" or args.label.lower() == ("tissueden"):
        n_class = 4
    elif args.label.lower() == "bcsc_race_eth_idx_clf" or args.label.lower() == "race":
        n_class = 6
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 1

    optimizer = None
    scheduler = None
    scalar = None
    if 'breast_clip' in args.arch:
        print(f"Architecture: {args.arch}")
        print(args.image_encoder_type)
        model = BreastClipClassifier(args, ckpt=ckpt, n_class=n_class)
        print("Model is loaded")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmup_epochs == 0.1:
            warmup_steps = args.epochs
        elif args.warmup_epochs == 1:
            warmup_steps = len(train_loader)
        else:
            warmup_steps = 10
        lr_config = {
            'total_epochs': args.epochs,
            'warmup_steps': warmup_steps,
            'total_steps': len(train_loader) * args.epochs
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler()

    model = model.to(device)
    print(model)

    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')

    if (
            args.label.lower() == "density" or
            args.label.lower() == "birads" or
            args.label.lower() == "tissueden" or
            args.label.lower() == "bcsc_race_eth_idx_clf" or
            args.label.lower() == "race"

    ):
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        print("No weighted BCE")
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    best_aucroc = 0.
    best_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, logger, device)

        if (
                'efficientnetv2' in args.arch or 'efficientnet_b5_ns' in args.arch
                or 'efficientnet_b5_ns-detect' in args.arch or 'efficientnetv2-detect' in args.arch
        ):
            scheduler.step()

        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, args, device, epoch, logger=logger
        )
        args.valid_folds['prediction'] = predictions

        valid_agg = None
        if args.dataset.lower() == "vindr":
            valid_agg = args.valid_folds
        elif args.dataset.lower() == "rsna":
            valid_agg = args.valid_folds[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
                ['patient_id', 'laterality']).mean()
        elif args.dataset.lower() == "embed":
            valid_agg = args.valid_folds[['patient_id', 'laterality', args.label, 'prediction']].groupby(
                ['patient_id', 'laterality']).max()
        elif args.dataset.lower() == "cmmd":
            valid_agg = args.valid_folds[['patient_id', 'LeftRight', args.label, 'prediction', 'fold']].groupby(
                ['patient_id', 'LeftRight']).mean()
        elif args.dataset.lower() == "nlbreast":
            valid_agg = args.valid_folds

        if (
                args.label.lower() == "density" or
                args.label.lower() == "birads" or
                args.label.lower() == "tissueden" or
                args.label.lower() == "bcsc_race_eth_idx_clf" or
                args.label.lower() == "race"
        ):
            correct_predictions = (valid_agg[args.label] == valid_agg['prediction']).sum()
            total_predictions = len(valid_agg)
            accuracy = correct_predictions / total_predictions
            valid_agg[args.label] = valid_agg[args.label].astype(int)
            valid_agg['prediction'] = valid_agg['prediction'].astype(int)
            f1 = f1_score(valid_agg[args.label], valid_agg['prediction'], average='macro')

            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  '
                f'accuracy: {accuracy * 100:.4f}   f1: {f1 * 100:.4f}'
            )
            logger.add_scalar(f'valid/{args.label}/accuracy', accuracy, epoch + 1)

            if best_acc < accuracy:
                best_acc = accuracy
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save Best acc: {best_acc * 100:.4f} Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'accuracy': accuracy,
                        'f1': f1,
                    }, args.chk_pt_path / model_name
                )
        else:
            metrics = all_classification_metrics(valid_agg[args.label].values, valid_agg['prediction'].values)
            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s'
            )
            print(f'Epoch {epoch + 1} - Classification metrics:')
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            logger.add_scalar(f'valid/{args.label}/AUC-ROC', metrics["AUROC"], epoch + 1)

            if best_aucroc < metrics["AUROC"]:
                best_aucroc = metrics["AUROC"]
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'auroc': metrics["AUROC"],
                        'classification_metrics': metrics,
                    }, args.chk_pt_path / model_name
                )

        if (
                args.label.lower() == "density" or
                args.label.lower() == "birads" or
                args.label.lower() == "tissueden" or
                args.label.lower() == "bcsc_race_eth_idx_clf" or
                args.label.lower() == "race"
        ):
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
            print(f'[Fold{args.cur_fold}], Best Accuracy: {best_acc * 100:.4f}')
        else:
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
            print(f'[Fold{args.cur_fold}], AUC-ROC Score: {best_aucroc:.4f}')
        predictions = torch.load(args.chk_pt_path / model_name, map_location='cpu')['predictions']
        args.valid_folds['prediction'] = predictions
        _save_valid_predictions(args, predictions)

    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds


# def inference_loop(args):
#     print(f'================== fold: {args.cur_fold} validating ======================')
#     print(args.valid_folds.shape)
#     predictions = torch.load(
#         args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_score_ver084.pth',
#         map_location='cpu')['predictions']
#     print(f'predictions: {predictions.shape}', type(predictions))
#     args.valid_folds['prediction'] = predictions
#
#     valid_agg = args.valid_folds[['patient_id', 'laterality', 'cancer', 'prediction', 'fold']].groupby(
#         ['patient_id', 'laterality']).mean()
#     aucroc = auroc(valid_agg['cancer'].values, valid_agg['prediction'].values)
#     print(f'AUC-ROC: {aucroc}')
#     return args.valid_folds.copy()

def inference_loop(args, device):
    print(f'================== fold: {args.cur_fold} inference ======================')
    print(args.valid_folds.shape)

    if args.weighted_BCE == "y":
        args.BCE_weights = {}
        args.BCE_weights[f"fold{args.cur_fold}"] = args.train_folds[args.train_folds[args.label] == 0].shape[0] / \
                                                   args.train_folds[args.train_folds[args.label] == 1].shape[0]
        print(f"args.BCE_weights: {args.BCE_weights}")

    if args.clip_chk_pt_path is not None:
        clip_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
        if clip_ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
            args.image_encoder_type = clip_ckpt["config"]["model"]["image_encoder"]["model_type"]
        elif clip_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
            args.image_encoder_type = clip_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = None
        clip_ckpt = None

    if args.running_interactive:
        args.valid_folds = args.valid_folds.sample(n=200)

    _, valid_loader = get_dataloader(args)
    print(f'valid_loader: {len(valid_loader)}')

    if args.label.lower() == "density" or args.label.lower() == ("tissueden"):
        n_class = 4
    elif args.label.lower() == "bcsc_race_eth_idx_clf" or args.label.lower() == "race":
        n_class = 6
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 1

    if 'breast_clip' in args.arch:
        print(f"Architecture: {args.arch}")
        print(args.image_encoder_type)
        model = BreastClipClassifier(args, ckpt=clip_ckpt, n_class=n_class)
    else:
        print("Unsupported architecture for inference")
        return None

    ckpt = torch.load(args.chk_pt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(model)

    if (
            args.label.lower() == "density" or
            args.label.lower() == "birads" or
            args.label.lower() == "tissueden" or
            args.label.lower() == "bcsc_race_eth_idx_clf" or
            args.label.lower() == "race"
    ):
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        print("No weighted BCE")
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    avg_val_loss, predictions = valid_fn(
        valid_loader, model, criterion, args, device, epoch=0, logger=None
    )
    print(f'avg_val_loss: {avg_val_loss:.4f}')
    print(f'predictions: {predictions.shape}', type(predictions))
    args.valid_folds['prediction'] = predictions

    valid_agg = None
    if args.dataset.lower() == "vindr":
        valid_agg = args.valid_folds
    elif args.dataset.lower() == "rsna":
        valid_agg = args.valid_folds[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
    elif args.dataset.lower() == "embed":
        valid_agg = args.valid_folds[['patient_id', 'laterality', args.label, 'prediction']].groupby(
            ['patient_id', 'laterality']).max()
    elif args.dataset.lower() == "cmmd":
        valid_agg = args.valid_folds[['patient_id', 'LeftRight', args.label, 'prediction', 'fold']].groupby(
            ['patient_id', 'LeftRight']).mean()
    elif args.dataset.lower() == "nlbreast":
        valid_agg = args.valid_folds

    if (
            args.label.lower() == "density" or
            args.label.lower() == "birads" or
            args.label.lower() == "tissueden" or
            args.label.lower() == "bcsc_race_eth_idx_clf" or
            args.label.lower() == "race"
    ):
        correct_predictions = (valid_agg[args.label] == valid_agg['prediction']).sum()
        total_predictions = len(valid_agg)
        accuracy = correct_predictions / total_predictions
        valid_agg[args.label] = valid_agg[args.label].astype(int)
        valid_agg['prediction'] = valid_agg['prediction'].astype(int)
        f1 = f1_score(valid_agg[args.label], valid_agg['prediction'], average='macro')
        print(f"Accuracy: {accuracy * 100:.4f} %, F1 Score: {f1 * 100:.4f}")
    else:
        metrics = all_classification_metrics(valid_agg[args.label].values, valid_agg['prediction'].values)
        print(f'Inference metrics:')
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    _save_valid_predictions(args, predictions)
    return args.valid_folds.copy()


def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = end = time.time()

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        batch_size = inputs.size(0)

        with torch.cuda.amp.autocast(enabled=args.apex):
            y_preds = model(inputs)
        if (
                args.label == "density" or
                args.label.lower() == "birads" or
                args.label.lower() == "tissueden" or
                args.label.lower() == "bcsc_race_eth_idx_clf" or
                args.label.lower() == "race"
        ):
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # batch scheduler
        # scheduler.step()
        if 'breast_clip' in args.arch:
            scheduler.step()
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar('train/epoch', epoch, index)
            logger.add_scalar('train/iter_loss', losses.avg, index)
            logger.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], index)

    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, logger=None):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                         total=len(valid_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        batch_size = inputs.size(0)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        with torch.no_grad():
            y_preds = model(inputs)

        if (
                args.label == "density" or
                args.label.lower() == "birads" or
                args.label.lower() == "tissueden" or
                args.label.lower() == "bcsc_race_eth_idx_clf" or
                args.label.lower() == "race"
        ):
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        if (
                args.label == "density" or
                args.label.lower() == "birads" or
                args.label.lower() == "tissueden" or
                args.label.lower() == "bcsc_race_eth_idx_clf" or
                args.label.lower() == "race"
        ):
            _, predicted = torch.max(y_preds, 1)
            preds.extend(predicted.cpu().numpy())
        else:
            preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())

        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar('valid/iter_loss', losses.avg, index)

    if (
            args.label == "density" or
            args.label.lower() == "birads" or
            args.label.lower() == "tissueden" or
            args.label.lower() == "bcsc_race_eth_idx_clf" or
            args.label.lower() == "race"
    ):
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)

    return losses.avg, predictions
