"""Patient-level K-Fold cross-validation for SC-Net.

Splits data at the patient level (not artery level) to prevent data leakage.
For each fold, trains a fresh model and evaluates, then reports mean +/- std
across all folds.

Usage:
  python cross_validate.py --n_folds 5 --pattern pre_training --data_root ./dataset/train
  python cross_validate.py --config configs/pretrain_default.yaml --n_folds 5
"""

import os
import sys
import copy
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import augmentation as aug
from config import opt
from framework import sc_net_framework
from optimization import compute_sc_class_weights
from scheduler_utils import LinearWarmupCosineDecay, ModelEMA, build_param_groups
from eval import od_predictions_to_artery_level, targets_to_artery_level, compute_metrics

# Reuse parse_args from train.py (includes --config YAML support)
from train import parse_args as train_parse_args, get_device


class PatientKFoldSplitter:
    """Generate patient-level K-Fold splits over a dataset directory.

    Extracts patient IDs from filenames (format: {PatientID}_{Artery}.nii)
    and ensures all arteries from the same patient stay in the same fold.
    """

    def __init__(self, volumes_root, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed

        file_list = sorted(os.listdir(volumes_root))
        self.file_list = file_list
        self.n_files = len(file_list)

        # Map patient ID -> list of file indices
        patient_to_indices = defaultdict(list)
        for idx, fname in enumerate(file_list):
            # Strip extension, then split on last underscore
            base = fname.rsplit('.', 1)[0]  # remove .nii or .nii.gz
            patient_id = base.rsplit('_', 1)[0]
            patient_to_indices[patient_id].append(idx)

        self.patient_ids = sorted(patient_to_indices.keys())
        self.patient_to_indices = patient_to_indices
        self.n_patients = len(self.patient_ids)

    def split(self):
        """Yield (train_indices, val_indices) for each fold.

        Each set of indices refers to positions in the sorted file list.
        """
        rng = np.random.RandomState(self.seed)
        patient_order = np.arange(self.n_patients)
        rng.shuffle(patient_order)

        fold_size = self.n_patients // self.n_folds
        remainder = self.n_patients % self.n_folds

        folds = []
        start = 0
        for f in range(self.n_folds):
            end = start + fold_size + (1 if f < remainder else 0)
            folds.append(patient_order[start:end])
            start = end

        for fold_idx in range(self.n_folds):
            val_patients = folds[fold_idx]
            train_patients = np.concatenate(
                [folds[j] for j in range(self.n_folds) if j != fold_idx]
            )

            val_indices = []
            for p_idx in val_patients:
                pid = self.patient_ids[p_idx]
                val_indices.extend(self.patient_to_indices[pid])

            train_indices = []
            for p_idx in train_patients:
                pid = self.patient_ids[p_idx]
                train_indices.extend(self.patient_to_indices[pid])

            yield sorted(train_indices), sorted(val_indices)


def build_cv_args():
    """Parse cross-validation arguments (extends train.py args)."""
    # First get the base train args by injecting --n_folds before parsing
    # We need to add --n_folds to the train parser
    parser = argparse.ArgumentParser(
        description='SC-Net Patient-Level K-Fold Cross-Validation',
        add_help=False,
    )
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--cv_seed', type=int, default=42,
                        help='Random seed for patient fold assignment')

    # Parse known args to extract n_folds, pass rest to train parser
    cv_args, remaining = parser.parse_known_args()

    # Get the full train args from remaining argv
    train_args = train_parse_args(remaining)

    # Merge cv-specific args into train args
    train_args.n_folds = cv_args.n_folds
    train_args.cv_seed = cv_args.cv_seed

    return train_args


def run_fold(fold_idx, n_folds, train_indices, val_indices, args):
    """Train and evaluate a single fold. Returns a metrics dict."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1}/{n_folds}")
    print(f"  Train files: {len(train_indices)}, Val files: {len(val_indices)}")
    print(f"{'='*60}\n")

    device = get_device(args.device)
    num_classes = 3 if args.pattern == 'pre_training' else 6

    # Build model
    sc_class_weights = None
    if args.sc_class_weight:
        sc_class_weights = compute_sc_class_weights(num_classes)

    fw = sc_net_framework(
        pattern=args.pattern,
        state_dict_root=args.pretrained,
        data_root=args.data_root if args.data_root else opt.data_params["dataset_root"],
        delta=args.delta,
        sc_class_weights=sc_class_weights,
        temporal_encoder_layers=getattr(args, 'temporal_encoder_layers', None),
        temporal_heads=getattr(args, 'temporal_heads', None),
        spatial_encoder_layers=getattr(args, 'spatial_encoder_layers', None),
        spatial_decoder_layers=getattr(args, 'spatial_decoder_layers', None),
    )

    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)

    data_root = args.data_root if args.data_root else opt.data_params["dataset_root"]

    # Build datasets using file_indices
    train_dataset = aug.cubic_sequence_data(
        dataset_root=data_root,
        pattern='training',
        input_shape=opt.net_params["input_shape"],
        window=opt.data_params["window_lw"],
        augment=args.augment,
        num_classes=num_classes,
        file_indices=train_indices,
    )
    val_dataset = aug.cubic_sequence_data(
        dataset_root=data_root,
        pattern='validation',
        input_shape=opt.net_params["input_shape"],
        window=opt.data_params["window_lw"],
        augment=False,
        num_classes=num_classes,
        file_indices=val_indices,
    )

    batch_size = opt.data_params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=aug.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=aug.collate_fn)

    # Optimizer
    if args.layerwise_lr:
        param_groups = build_param_groups(model, args.lr)
    else:
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad],
                         'lr': args.lr}]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineDecay(optimizer, max_epochs=args.epochs,
                                        warmup_epochs=args.warmup_epochs)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # AMP
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_metrics = None

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    od_out, sc_out = model(images)
                    loss_dict = loss_fn(od_out, sc_out, targets)
                    loss = loss_dict['total']
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                od_out, sc_out = model(images)
                loss_dict = loss_fn(od_out, sc_out, targets)
                loss = loss_dict['total']
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)

        # --- Validate ---
        if ema is not None:
            ema.apply(model)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_sten_preds, all_sten_gts = [], []
        all_plaq_preds, all_plaq_gts = [], []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        od_out, sc_out = model(images)
                        ld = loss_fn(od_out, sc_out, targets_dev)
                else:
                    od_out, sc_out = model(images)
                    ld = loss_fn(od_out, sc_out, targets_dev)

                val_loss += ld['total'].item()
                val_batches += 1

                for i in range(images.shape[0]):
                    od_out_i = {
                        'pred_logits': od_out['pred_logits'][i],
                        'pred_boxes': od_out['pred_boxes'][i],
                    }
                    sten_pred, plaq_pred = od_predictions_to_artery_level(od_out_i, num_classes)
                    sten_gt, plaq_gt = targets_to_artery_level(targets[i], num_classes)
                    all_sten_preds.append(sten_pred)
                    all_sten_gts.append(sten_gt)
                    if plaq_pred != -1 and plaq_gt != -1:
                        all_plaq_preds.append(plaq_pred)
                        all_plaq_gts.append(plaq_gt)

        if ema is not None:
            ema.restore(model)

        avg_val_loss = val_loss / max(val_batches, 1)
        sten_metrics = compute_metrics(all_sten_gts, all_sten_preds, num_classes=3)
        if len(all_plaq_preds) > 0:
            plaq_metrics = compute_metrics(all_plaq_gts, all_plaq_preds, num_classes=3)
        else:
            plaq_metrics = {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0, 'spec': 0}

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = {
                'val_loss': avg_val_loss,
                'stenosis_acc': sten_metrics['acc'],
                'stenosis_prec': sten_metrics['prec'],
                'stenosis_recall': sten_metrics['recall'],
                'stenosis_f1': sten_metrics['f1'],
                'stenosis_spec': sten_metrics['spec'],
                'plaque_acc': plaq_metrics['acc'],
                'plaque_prec': plaq_metrics['prec'],
                'plaque_recall': plaq_metrics['recall'],
                'plaque_f1': plaq_metrics['f1'],
                'plaque_spec': plaq_metrics['spec'],
            }

        if (epoch + 1) % max(args.epochs // 10, 1) == 0 or epoch == args.epochs - 1:
            print(f"  Fold {fold_idx+1} Epoch [{epoch+1}/{args.epochs}] "
                  f"Train: {avg_train_loss:.4f} Val: {avg_val_loss:.4f} "
                  f"Sten-F1: {sten_metrics['f1']:.3f} Plaq-F1: {plaq_metrics['f1']:.3f}")

    if best_metrics is None:
        best_metrics = {
            'val_loss': float('inf'),
            'stenosis_acc': 0, 'stenosis_prec': 0, 'stenosis_recall': 0,
            'stenosis_f1': 0, 'stenosis_spec': 0,
            'plaque_acc': 0, 'plaque_prec': 0, 'plaque_recall': 0,
            'plaque_f1': 0, 'plaque_spec': 0,
        }

    print(f"\n  Fold {fold_idx+1} BEST val_loss: {best_metrics['val_loss']:.4f} "
          f"Sten-F1: {best_metrics['stenosis_f1']:.3f} "
          f"Plaq-F1: {best_metrics['plaque_f1']:.3f}")

    return best_metrics


def main():
    args = build_cv_args()

    data_root = args.data_root if args.data_root else opt.data_params["dataset_root"]
    volumes_root = os.path.join(data_root, 'volumes/')

    print(f"Cross-validation: {args.n_folds} folds, seed={args.cv_seed}")
    print(f"Data root: {data_root}")

    splitter = PatientKFoldSplitter(volumes_root, n_folds=args.n_folds,
                                    seed=args.cv_seed)
    print(f"Total files: {splitter.n_files}, Unique patients: {splitter.n_patients}")

    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split()):
        fold_metrics = run_fold(fold_idx, args.n_folds, train_idx, val_idx, args)
        all_fold_metrics.append(fold_metrics)

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS ({args.n_folds} folds)")
    print(f"{'='*60}")

    metric_keys = ['val_loss', 'stenosis_acc', 'stenosis_prec', 'stenosis_recall',
                   'stenosis_f1', 'stenosis_spec', 'plaque_acc', 'plaque_prec',
                   'plaque_recall', 'plaque_f1', 'plaque_spec']

    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {key:20s}: {mean:.4f} +/- {std:.4f}  "
              f"(per-fold: {', '.join(f'{v:.4f}' for v in values)})")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
