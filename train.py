"""Training script for SC-Net (Spatio-Temporal Contrast Network).

Supports:
  - Mixed-precision training (AMP)
  - Multi-GPU via DistributedDataParallel (launch with torchrun)
  - Layer-wise learning rates (backbone 0.1x, transformer 0.5x, heads 1.0x)
  - Linear warmup + cosine decay LR schedule
  - Exponential Moving Average (EMA) of model weights
  - Per-epoch validation with artery-level classification metrics

Example:
  # Single GPU
  python train.py --pattern pre_training --data_root ./data

  # Multi-GPU (2x)
  torchrun --nproc_per_node=2 train.py --distributed --pattern pre_training --data_root ./data
"""

import os
import sys
import random
import argparse
import time
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from framework import sc_net_framework
from config import opt
from optimization import compute_sc_class_weights
from scheduler_utils import LinearWarmupCosineDecay, CosineAnnealingWarmRestarts, ModelEMA, build_param_groups
from eval import od_predictions_to_artery_level, targets_to_artery_level, compute_metrics


def load_config(config_path):
    """Load a YAML configuration file and return its contents as a dict."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='SC-Net Training')

    # --- YAML config ---
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override YAML values)')

    # --- existing args ---
    parser.add_argument('--pattern', type=str, default='pre_training',
                        choices=['pre_training', 'fine_tuning'],
                        help='Training stage')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override dataset root path')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                        help='Gradient clipping max norm')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "auto", "cpu", "cuda:0", "cuda:1", etc.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pre-trained weights (for fine-tuning)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--print_every', type=int, default=1,
                        help='Print training stats every N batches')

    # --- new flags ---
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable mixed-precision training (default: True)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed-precision training')

    parser.add_argument('--distributed', action='store_true',
                        help='Enable DDP multi-GPU training (launch with torchrun)')

    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of LR warmup epochs')

    parser.add_argument('--layerwise_lr', action='store_true', default=True,
                        help='Enable layer-wise learning rates (default: True)')
    parser.add_argument('--no_layerwise_lr', action='store_true',
                        help='Disable layer-wise learning rates')

    parser.add_argument('--ema', action='store_true', default=True,
                        help='Enable Exponential Moving Average (default: True)')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay factor')

    parser.add_argument('--accumulate_steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1, no accumulation)')

    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Minimum val loss improvement for early stopping')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None = non-deterministic)')
    parser.add_argument('--eos_coef', type=float, default=None,
                        help='No-object class weight for detection loss (default: use config value 0.2)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader worker processes (default: 0)')
    parser.add_argument('--patient_split', action='store_true', default=False,
                        help='Use patient-level splitting to prevent data leakage')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for patient-level split (default: 42)')

    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation for the training set')

    parser.add_argument('--delta', type=float, default=1.0,
                        help='Weight for the dual-task contrastive loss term')
    parser.add_argument('--sc_class_weight', action='store_true', default=True,
                        help='Enable class weighting for SC loss (default: True)')
    parser.add_argument('--no_sc_class_weight', action='store_true',
                        help='Disable class weighting for SC loss')
    parser.add_argument('--boost_nonsig', action='store_true', default=False,
                        help='Double loss weight for Non-significant stenosis class (index 2)')
    parser.add_argument('--ordinal_weight', type=float, default=0.0,
                        help='Weight for ordinal EMD loss term (0=disabled, e.g. 0.5). '
                             'Penalises severity misjudgements proportional to ordinal distance '
                             '(Healthy↔Significant > Healthy↔NonSig).')

    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss instead of CE for SC branch')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma (focusing param) for focal loss (default: 2.0)')

    parser.add_argument('--multi_window', action='store_true', default=False,
                        help='Use 3 stacked HU windows as input channels (soft tissue, calcium, vascular)')

    # --- Ldc improvements ---
    parser.add_argument('--dc_warmup_hold', type=int, default=0,
                        help='Hold dc_weight=0 for this many epochs (default: 0)')
    parser.add_argument('--dc_warmup_ramp', type=int, default=0,
                        help='Linearly ramp dc_weight from 0 to delta over this many epochs '
                             'after the hold period (default: 0)')
    parser.add_argument('--dc_confidence_threshold', type=float, default=0.0,
                        help='Min softmax confidence for Ldc pseudo-labels (default: 0.0 = no gating)')
    parser.add_argument('--soft_dc', action='store_true', default=False,
                        help='Use soft probability pseudo-labels for Ldc (KL-div instead of hard CE)')
    parser.add_argument('--dc_temperature_start', type=float, default=3.0,
                        help='Initial softmax temperature for DC soft pseudo-labels; anneals to 1.0 over dc_warmup_ramp (default: 3.0)')
    parser.add_argument('--dc_confidence_start', type=float, default=0.7,
                        help='Initial confidence threshold for GT-free C⁻¹(ŷ_od) pseudo-labels; '
                             'linearly anneals to --dc_confidence_threshold over dc_warmup_ramp epochs '
                             '(default: 0.7). Only active when dc_warmup_ramp > 0.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon for SC classification loss (default: 0.0)')
    parser.add_argument('--balanced_sampling', action='store_true',
                        help='Use class-balanced batch sampling (WeightedRandomSampler)')

    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'cosine_warm_restarts'],
                        help='LR schedule: "cosine" (linear warmup + cosine decay, default) or '
                             '"cosine_warm_restarts" (CosineAnnealingWarmRestarts, good for long runs)')
    parser.add_argument('--lr_t0', type=int, default=50,
                        help='First cycle length in epochs for cosine_warm_restarts (default: 50)')
    parser.add_argument('--lr_t_mult', type=int, default=2,
                        help='Cycle length multiplier for cosine_warm_restarts (default: 2)')
    parser.add_argument('--swa', action='store_true', default=False,
                        help='Enable Stochastic Weight Averaging (SWA). Starts averaging after '
                             '--swa_start_epoch and updates SWA model every epoch.')
    parser.add_argument('--swa_start_epoch', type=int, default=None,
                        help='Epoch to start SWA averaging (default: half of --epochs)')
    parser.add_argument('--swa_lr', type=float, default=None,
                        help='SWA learning rate (default: half of --lr)')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='Base directory for TensorBoard logs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log batch-level metrics every N batches')

    # --- transformer hyperparameters ---
    parser.add_argument('--temporal_encoder_layers', type=int, default=None,
                        help='Number of temporal transformer encoder layers (default: 4)')
    parser.add_argument('--temporal_heads', type=int, default=None,
                        help='Number of temporal transformer attention heads (default: 8)')
    parser.add_argument('--spatial_encoder_layers', type=int, default=None,
                        help='Number of spatial transformer encoder layers (default: 4)')
    parser.add_argument('--spatial_decoder_layers', type=int, default=None,
                        help='Number of spatial transformer decoder layers (default: 4)')

    args = parser.parse_args(argv)

    # Apply YAML config as defaults — CLI args take precedence
    if args.config:
        yaml_config = load_config(args.config)
        for key, value in yaml_config.items():
            if key == 'config':
                continue  # skip self-reference
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    # Resolve negation flags
    if args.no_amp:
        args.amp = False
    if args.no_layerwise_lr:
        args.layerwise_lr = False
    if args.no_ema:
        args.ema = False
    if args.no_sc_class_weight:
        args.sc_class_weight = False

    return args


def get_device(device_str):
    """Select device, preferring the GPU with the most free memory."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                free_mem = []
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    reserved = torch.cuda.memory_reserved(i)
                    free_mem.append(total - reserved)
                best_gpu = max(range(len(free_mem)), key=lambda i: free_mem[i])
                return torch.device(f'cuda:{best_gpu}')
            return torch.device('cuda:0')
        return torch.device('cpu')
    return torch.device(device_str)


class Trainer:
    """Full-featured trainer for SC-Net."""

    def __init__(self, args):
        self.args = args

        # ---- seed setup ----
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ---- distributed setup ----
        self.distributed = args.distributed
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        if self.distributed:
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = get_device(args.device)

        self.is_main = (self.rank == 0)

        # ---- num_classes (needed for metrics) ----
        # Mirror framework.py: pre_training uses index [0], fine_tuning uses index [1]
        if args.pattern == 'pre_training':
            self.num_classes = opt.net_params["num_classes"][0]
        else:
            self.num_classes = opt.net_params["num_classes"][1]

        # ---- build components ----
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()

        # ---- EMA ----
        self.ema = None
        if args.ema:
            self.ema = ModelEMA(self.model, decay=args.ema_decay)

        # ---- AMP ----
        self.use_amp = args.amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # ---- bookkeeping ----
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_stenosis_f1 = 0.0   # used as primary checkpoint metric
        self.global_step = 0
        self.patience_counter = 0

        # ---- TensorBoard ----
        self.writer = None
        if self.is_main:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_subdir = os.path.join(args.log_dir,
                                      f'{args.pattern}_{timestamp}')
            self.writer = SummaryWriter(log_dir=log_subdir)

        # ---- resume ----
        if args.resume:
            self._load_checkpoint(args.resume)

        # ---- print summary ----
        if self.is_main:
            self._print_summary()

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def setup_model(self):
        """Instantiate the model and loss function, optionally wrap in DDP."""
        # Compute SC class weights if enabled
        sc_class_weights = None
        if self.args.sc_class_weight:
            boost_nonsig = self.args.boost_nonsig
            if boost_nonsig and self.args.pattern != 'fine_tuning':
                print("  [Warning] --boost_nonsig has no meaningful effect in pre_training mode; ignoring.")
                boost_nonsig = False
            sc_class_weights = compute_sc_class_weights(
                num_classes=self.num_classes,
                boost_nonsig=boost_nonsig,
                nonsig_idx=2,
            )

        fw = sc_net_framework(
            pattern=self.args.pattern,
            state_dict_root=self.args.pretrained,
            data_root=self.args.data_root,
            delta=self.args.delta,
            sc_class_weights=sc_class_weights,
            use_focal=self.args.focal_loss,
            focal_gamma=self.args.focal_gamma,
            dc_confidence_threshold=self.args.dc_confidence_threshold,
            eos_coef=self.args.eos_coef,
            label_smoothing=self.args.label_smoothing,
            use_soft_dc=self.args.soft_dc,
            patient_split=self.args.patient_split,
            split_seed=self.args.split_seed,
            temporal_encoder_layers=getattr(self.args, 'temporal_encoder_layers', None),
            temporal_heads=getattr(self.args, 'temporal_heads', None),
            spatial_encoder_layers=getattr(self.args, 'spatial_encoder_layers', None),
            spatial_decoder_layers=getattr(self.args, 'spatial_decoder_layers', None),
            multi_window=self.args.multi_window,
            ordinal_weight=getattr(self.args, 'ordinal_weight', 0.0),
        )
        self._fw = fw  # keep reference for data setup

        self.model = fw.model.to(self.device)
        self.loss_fn = fw.loss_fn.to(self.device)

        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

    def setup_data(self):
        """Build train and eval dataloaders, respecting DDP and augmentation."""
        import augmentation as aug
        from torch.utils.data import WeightedRandomSampler

        fw = self._fw

        # If augmentation requested, rebuild the training dataset with augment=True
        if self.args.augment:
            train_dataset = aug.cubic_sequence_data(
                dataset_root=fw.data_root,
                pattern='training',
                train_ratio=fw.train_ratio,
                input_shape=fw.input_shape,
                window=fw.window_lw,
                augment=True,
                num_classes=fw.model_num_classes,
                file_indices=getattr(fw, 'train_indices', None),
                multi_window=self.args.multi_window
            )
        else:
            train_dataset = fw.dataLoader_train.dataset

        eval_dataset = aug.cubic_sequence_data(
            dataset_root=fw.data_root,
            pattern='validation',  # must be 'validation' — 'eval' maps to test split
            train_ratio=fw.train_ratio,
            input_shape=fw.input_shape,
            window=fw.window_lw,
            augment=False,
            num_classes=fw.model_num_classes,
            file_indices=getattr(fw, 'eval_indices', None),  # eval_indices = val_indices when patient_split
            multi_window=self.args.multi_window
        )
        batch_size = opt.data_params["batch_size"]

        # Samplers
        self.train_sampler = None
        eval_sampler = None
        shuffle_train = True

        if self.distributed:
            self.train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True,
            )
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=False,
            )
            shuffle_train = False  # sampler handles shuffling
        elif self.args.balanced_sampling:
            # Class-balanced sampling: weight each sample by inverse class frequency
            sample_weights = self._compute_sample_weights(train_dataset)
            self.train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(train_dataset), replacement=True)
            shuffle_train = False  # sampler handles ordering

        pin = (self.device.type == 'cuda')
        nw = self.args.num_workers

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=shuffle_train, sampler=self.train_sampler,
            collate_fn=aug.collate_fn,
            num_workers=nw, pin_memory=pin,
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size,
            shuffle=False, sampler=eval_sampler,
            collate_fn=aug.collate_fn,
            num_workers=nw, pin_memory=pin,
        )

    def _compute_sample_weights(self, dataset):
        """Compute per-sample weights for balanced sampling based on stenosis class.

        Uses the dataset's file_pairs API (vol_path, sten_path, plaq_path, comb_path)
        to load label files directly, avoiding dependency on the deprecated
        labels_file_list attribute.
        """
        import augmentation as _aug

        class_counts = {0: 0, 1: 0, 2: 0}
        sample_classes = []

        for i in range(len(dataset)):
            if dataset.file_indices is not None:
                actual_idx = dataset.file_indices[i]
            else:
                actual_idx = i + dataset.data_start

            _, sten_path, plaq_path, comb_path = dataset.file_pairs[actual_idx]

            if comb_path is not None:
                labels = np.loadtxt(comb_path).astype(np.int32)
            else:
                sten = np.loadtxt(sten_path).astype(np.int32)
                plaq = np.loadtxt(plaq_path).astype(np.int32)
                labels = _aug.merge_new_labels(sten, plaq)

            max_label = int(labels.max())
            if max_label == 0:
                sc = 0  # healthy
            elif max_label <= 3:
                sc = 1  # non-significant
            else:
                sc = 2  # significant
            sample_classes.append(sc)
            class_counts[sc] += 1

        total = len(sample_classes)
        num_classes_bal = len(class_counts)
        class_weights = {c: total / (num_classes_bal * max(count, 1))
                         for c, count in class_counts.items()}
        sample_weights = [class_weights[c] for c in sample_classes]

        if self.is_main:
            print(f"  [Balanced Sampling] Class distribution: {class_counts}")
            wt_str = {c: round(w, 2) for c, w in class_weights.items()}
            print(f"  [Balanced Sampling] Class weights: {wt_str}")

        return sample_weights

    def setup_optimizer(self):
        """Create optimizer with optional layer-wise LR, warmup+cosine scheduler, and SWA."""
        raw_model = self.model.module if self.distributed else self.model
        if self.args.layerwise_lr:
            param_groups = build_param_groups(raw_model, self.args.lr)
        else:
            param_groups = [{'params': [p for p in raw_model.parameters() if p.requires_grad],
                             'lr': self.args.lr}]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        lr_schedule = getattr(self.args, 'lr_schedule', 'cosine')
        if lr_schedule == 'cosine_warm_restarts':
            t0 = getattr(self.args, 'lr_t0', 50)
            t_mult = getattr(self.args, 'lr_t_mult', 2)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                t_0=t0,
                t_mult=t_mult,
                warmup_epochs=self.args.warmup_epochs,
            )
            if self.is_main:
                print(f"  [Scheduler] CosineAnnealingWarmRestarts T0={t0}, T_mult={t_mult}")
        else:
            self.scheduler = LinearWarmupCosineDecay(
                self.optimizer,
                max_epochs=self.args.epochs,
                warmup_epochs=self.args.warmup_epochs,
            )

        # SWA setup
        self.swa_model = None
        self.swa_scheduler = None
        if getattr(self.args, 'swa', False):
            from torch.optim.swa_utils import AveragedModel, SWALR
            swa_start = getattr(self.args, 'swa_start_epoch', None)
            if swa_start is None:
                swa_start = self.args.epochs // 2
            self.swa_start_epoch = swa_start
            swa_lr = getattr(self.args, 'swa_lr', None)
            if swa_lr is None:
                swa_lr = self.args.lr / 2
            self.swa_model = AveragedModel(raw_model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=swa_lr,
                                       anneal_epochs=max(1, (self.args.epochs - swa_start) // 5))
            if self.is_main:
                print(f"  [SWA] Enabled: start_epoch={swa_start}, swa_lr={swa_lr:.2e}")

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, val_loss, path=None):
        """Save training state to disk (rank 0 only)."""
        if not self.is_main:
            return
        if path is None:
            path = os.path.join(self.args.checkpoint_dir,
                                f'checkpoint_epoch_{epoch}.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        raw_model = self.model.module if self.distributed else self.model
        state = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': val_loss,
        }
        if self.ema is not None:
            state['ema_state_dict'] = copy.deepcopy(self.ema.shadow)
        torch.save(state, path)

    def _load_checkpoint(self, path):
        """Resume training from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)

        raw_model = self.model.module if self.distributed else self.model
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('loss', float('inf'))

        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.shadow = checkpoint['ema_state_dict']

        if self.is_main:
            print(f"Resumed from epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def _compute_dc_weight(self, epoch):
        """Compute dc loss weight for this epoch (delayed ramp schedule)."""
        hold = self.args.dc_warmup_hold
        ramp = self.args.dc_warmup_ramp

        if hold == 0 and ramp == 0:
            return self.args.delta  # no scheduling

        if epoch < hold:
            return 0.0
        elif ramp > 0 and epoch < hold + ramp:
            progress = (epoch - hold) / ramp
            return self.args.delta * progress
        else:
            return self.args.delta

    def _compute_dc_confidence(self, epoch: int) -> float:
        """Anneal the GT-free C⁻¹ confidence threshold from dc_confidence_start to floor.

        Mirrors the dc_weight schedule:
          - During hold: return dc_confidence_start (aggressive filtering)
          - During ramp: linearly interpolate from start → dc_confidence_threshold
          - After ramp: return dc_confidence_threshold (floor)
          - No schedule (hold=ramp=0): return dc_confidence_threshold immediately
        """
        hold  = self.args.dc_warmup_hold
        ramp  = self.args.dc_warmup_ramp
        start = getattr(self.args, 'dc_confidence_start', 0.7)
        floor = self.args.dc_confidence_threshold

        if hold == 0 and ramp == 0:
            return floor
        if epoch < hold:
            return start
        elif ramp > 0 and epoch < hold + ramp:
            progress = (epoch - hold) / ramp
            return start - progress * (start - floor)
        else:
            return floor

    def train_one_epoch(self, epoch):
        """Run one training epoch. Returns (avg_loss, avg_od, avg_sc, avg_dc)."""
        self.model.train()
        if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(epoch)

        # Update dc weight and confidence threshold for delayed ramp
        dc_weight = self._compute_dc_weight(epoch)
        self.loss_fn.set_dc_weight(dc_weight)
        dc_confidence = self._compute_dc_confidence(epoch)
        self.loss_fn.set_dc_confidence(dc_confidence)

        # DC temperature annealing: high temperature early (soft targets), anneal to 1.0 by end of ramp
        dc_temp_start = getattr(self.args, 'dc_temperature_start', 3.0)
        hold = getattr(self.args, 'dc_warmup_hold', 0)
        ramp = getattr(self.args, 'dc_warmup_ramp', 0)
        if ramp > 0 and epoch < hold + ramp:
            frac = max(0.0, (epoch - hold) / ramp)   # 0→1 during ramp
            dc_temp = dc_temp_start - frac * (dc_temp_start - 1.0)
        else:
            dc_temp = 1.0
        self.loss_fn.set_dc_temperature(dc_temp)

        total_loss = 0.0
        total_od = 0.0
        total_sc = 0.0
        total_dc = 0.0
        num_batches = 0
        accum = self.args.accumulate_steps
        grad_norm = 0.0
        num_train_batches = len(self.train_loader)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Zero gradients at the start of each accumulation window
            if batch_idx % accum == 0:
                self.optimizer.zero_grad()

            is_accum_step = ((batch_idx + 1) % accum == 0 or
                             (batch_idx + 1) == num_train_batches)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    od_outputs, sc_outputs = self.model(images)
                    loss_dict = self.loss_fn(od_outputs, sc_outputs, targets)
                    loss = loss_dict['total'] / accum

                self.scaler.scale(loss).backward()

                if is_accum_step:
                    if self.args.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.grad_clip)
                    else:
                        grad_norm = self._compute_grad_norm()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.ema is not None:
                        self.ema.update(self.model)
            else:
                od_outputs, sc_outputs = self.model(images)
                loss_dict = self.loss_fn(od_outputs, sc_outputs, targets)
                loss = loss_dict['total'] / accum

                loss.backward()

                if is_accum_step:
                    if self.args.grad_clip > 0:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.grad_clip)
                    else:
                        grad_norm = self._compute_grad_norm()

                    self.optimizer.step()

                    if self.ema is not None:
                        self.ema.update(self.model)

            # Record unscaled loss for logging
            batch_loss = loss.item() * accum
            batch_od = loss_dict['od'].item()
            batch_sc = loss_dict['sc'].item()
            batch_dc = loss_dict['dc'].item()

            total_loss += batch_loss
            total_od += batch_od
            total_sc += batch_sc
            total_dc += batch_dc
            num_batches += 1
            self.global_step += 1

            # Per-batch TensorBoard logging
            if self.writer is not None and (batch_idx + 1) % self.args.log_every == 0:
                self.writer.add_scalar('Batch/loss', batch_loss,
                                       self.global_step)
                self.writer.add_scalar('Batch/grad_norm',
                                       grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                                       self.global_step)

            if self.is_main and (batch_idx + 1) % self.args.print_every == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{num_train_batches}] "
                      f"Loss: {batch_loss:.4f} (OD: {batch_od:.4f} "
                      f"SC: {batch_sc:.4f} DC: {batch_dc:.4f}) "
                      f"Avg: {avg_loss:.4f}")

        n = max(num_batches, 1)
        return total_loss / n, total_od / n, total_sc / n, total_dc / n

    def _compute_grad_norm(self):
        """Compute total gradient L2 norm across all parameters."""
        params = [p for p in self.model.parameters()
                  if p.grad is not None]
        if len(params) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
        return total_norm

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation: compute loss and artery-level classification metrics.

        When EMA is enabled the shadow weights are applied for the forward
        passes and then restored afterwards.
        """
        # Swap to EMA weights if available
        if self.ema is not None:
            self.ema.apply(self.model)

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_stenosis_preds = []
        all_stenosis_gts = []
        all_plaque_preds = []
        all_plaque_gts = []

        for images, targets in self.eval_loader:
            images = images.to(self.device)
            targets_dev = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    od_outputs, sc_outputs = self.model(images)
                    loss_dict = self.loss_fn(od_outputs, sc_outputs, targets_dev)
            else:
                od_outputs, sc_outputs = self.model(images)
                loss_dict = self.loss_fn(od_outputs, sc_outputs, targets_dev)

            total_loss += loss_dict['total'].item()
            num_batches += 1

            # Collect artery-level predictions / ground truths
            batch_size = images.shape[0]
            for i in range(batch_size):
                od_out_i = {
                    'pred_logits': od_outputs['pred_logits'][i],
                    'pred_boxes': od_outputs['pred_boxes'][i],
                }
                target_i = targets[i]  # original (possibly CPU) targets

                stenosis_pred, plaque_pred = od_predictions_to_artery_level(
                    od_out_i, self.num_classes)
                stenosis_gt, plaque_gt = targets_to_artery_level(
                    target_i, self.num_classes)

                all_stenosis_preds.append(stenosis_pred)
                all_stenosis_gts.append(stenosis_gt)

                if plaque_pred != -1 and plaque_gt != -1:
                    all_plaque_preds.append(plaque_pred)
                    all_plaque_gts.append(plaque_gt)

        val_loss = total_loss / max(num_batches, 1)

        # Synchronise val_loss across all ranks so early-stopping is consistent.
        # Without this, each rank sees only its own subset of the eval set and
        # patience_counter can diverge — one rank breaks early while the other
        # hangs indefinitely at the next NCCL ALLREDUCE.
        if self.distributed:
            val_loss_t = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
            val_loss = val_loss_t.item()

        # Compute classification metrics
        stenosis_metrics = compute_metrics(all_stenosis_gts, all_stenosis_preds,
                                           num_classes=3)
        if len(all_plaque_preds) > 0:
            plaque_metrics = compute_metrics(all_plaque_gts, all_plaque_preds,
                                             num_classes=3)
        else:
            plaque_metrics = {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0, 'spec': 0}

        if self.is_main:
            print(f"  [Val] Stenosis -- ACC: {stenosis_metrics['acc']:.3f}  "
                  f"Prec: {stenosis_metrics['prec']:.3f}  "
                  f"Rec: {stenosis_metrics['recall']:.3f}  "
                  f"F1: {stenosis_metrics['f1']:.3f}  "
                  f"Spec: {stenosis_metrics['spec']:.3f}")
            print(f"  [Val] Plaque   -- ACC: {plaque_metrics['acc']:.3f}  "
                  f"Prec: {plaque_metrics['prec']:.3f}  "
                  f"Rec: {plaque_metrics['recall']:.3f}  "
                  f"F1: {plaque_metrics['f1']:.3f}  "
                  f"Spec: {plaque_metrics['spec']:.3f}")

        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.model)

        return val_loss, stenosis_metrics, plaque_metrics

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full training loop."""
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        if self.is_main:
            print(f"\nStarting {self.args.pattern} for {self.args.epochs} epochs")
            print(f"  Train batches: {len(self.train_loader)}")
            print(f"  Val batches:   {len(self.eval_loader)}")
            print()

        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start = time.time()

            train_loss, train_od, train_sc, train_dc = self.train_one_epoch(epoch)
            val_loss, stenosis_metrics, plaque_metrics = self.validate(epoch)

            # SWA: update averaged model after swa_start_epoch
            if self.swa_model is not None and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(
                    self.model.module if self.distributed else self.model)
                self.swa_scheduler.step()
            else:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - epoch_start

            if self.is_main:
                dc_w    = self._compute_dc_weight(epoch)
                dc_conf = self._compute_dc_confidence(epoch)
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f} | DC_w: {dc_w:.3f} | "
                      f"DC_conf: {dc_conf:.3f} | Time: {elapsed:.1f}s")

                # ---- TensorBoard epoch-level logging ----
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    self.writer.add_scalar('Loss/train_od', train_od, epoch)
                    self.writer.add_scalar('Loss/train_sc', train_sc, epoch)
                    self.writer.add_scalar('Loss/train_dc', train_dc, epoch)

                    self.writer.add_scalar('Metrics/stenosis_acc',
                                           stenosis_metrics['acc'], epoch)
                    self.writer.add_scalar('Metrics/stenosis_f1',
                                           stenosis_metrics['f1'], epoch)
                    self.writer.add_scalar('Metrics/stenosis_prec',
                                           stenosis_metrics['prec'], epoch)
                    self.writer.add_scalar('Metrics/stenosis_recall',
                                           stenosis_metrics['recall'], epoch)
                    self.writer.add_scalar('Metrics/plaque_acc',
                                           plaque_metrics['acc'], epoch)
                    self.writer.add_scalar('Metrics/plaque_f1',
                                           plaque_metrics['f1'], epoch)
                    self.writer.add_scalar('Metrics/plaque_prec',
                                           plaque_metrics['prec'], epoch)
                    self.writer.add_scalar('Metrics/plaque_recall',
                                           plaque_metrics['recall'], epoch)
                    self.writer.add_scalar('Metrics/best_stenosis_f1',
                                           self.best_stenosis_f1, epoch)

                    self.writer.add_scalar('LR/learning_rate',
                                           current_lr, epoch)
                    self.writer.add_scalar('Schedule/dc_weight',
                                           dc_w, epoch)
                    self.writer.add_scalar('Schedule/dc_confidence',
                                           dc_conf, epoch)

                    # Per-group LRs when layer-wise LR is enabled
                    if self.args.layerwise_lr:
                        for pg in self.optimizer.param_groups:
                            if 'name' in pg:
                                self.writer.add_scalar(
                                    f'LR/lr_{pg["name"]}',
                                    pg['lr'], epoch)

                    self.writer.flush()

                # Save best model — use stenosis F1 as primary metric.
                # Val loss is unsuitable: it includes the DC term which grows
                # monotonically as dc_weight ramps 0→1, so best_model would be
                # frozen at the last pre-DC epoch regardless of classification quality.
                current_f1 = stenosis_metrics['f1']
                if current_f1 > self.best_stenosis_f1 + self.args.min_delta:
                    self.save_checkpoint(
                        epoch, val_loss,
                        os.path.join(self.args.checkpoint_dir, 'best_model.pth'),
                    )
                    print(f"  -> New best model saved "
                          f"(stenosis_f1={current_f1:.4f}, val_loss={val_loss:.4f})")
                elif self.args.patience > 0:
                    print(f"  -> No improvement ({self.patience_counter + 1}/{self.args.patience})")

                # Periodic checkpoint
                if (epoch + 1) % self.args.save_every == 0:
                    self.save_checkpoint(epoch, val_loss)

            # Early stopping tracking (all ranks, so DDP stays in sync)
            current_f1 = stenosis_metrics['f1']
            if current_f1 > self.best_stenosis_f1 + self.args.min_delta:
                self.best_val_loss = val_loss        # keep for logging continuity
                self.best_stenosis_f1 = current_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.args.patience > 0 and self.patience_counter >= self.args.patience:
                if self.is_main:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {self.args.patience} epochs)")
                break

        # Update SWA BatchNorm statistics and save SWA model
        if self.swa_model is not None and self.is_main:
            from torch.optim.swa_utils import update_bn
            train_loader_for_bn = self.train_loader
            update_bn(train_loader_for_bn, self.swa_model, device=self.device)
            swa_path = os.path.join(self.args.checkpoint_dir, 'swa_model.pth')
            torch.save({'model_state_dict': self.swa_model.module.state_dict()}, swa_path)
            print(f"SWA model saved to {swa_path}")

        # Final checkpoint
        if self.is_main:
            self.save_checkpoint(
                self.args.epochs - 1,
                val_loss,
                os.path.join(self.args.checkpoint_dir, 'final_model.pth'),
            )
            print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

        # ---- cleanup ----
        if self.writer is not None:
            self.writer.close()

        if self.distributed:
            dist.destroy_process_group()

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        raw_model = self.model.module if self.distributed else self.model
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters()
                               if p.requires_grad)

        print("=" * 60)
        print("SC-Net Trainer")
        print("=" * 60)
        print(f"  Pattern:          {self.args.pattern}")
        print(f"  Device:           {self.device}")
        print(f"  Distributed:      {self.distributed}"
              + (f" (world_size={self.world_size})" if self.distributed else ""))
        print(f"  AMP:              {self.use_amp}")
        print(f"  EMA:              {self.ema is not None}"
              + (f" (decay={self.args.ema_decay})" if self.ema else ""))
        print(f"  Layer-wise LR:    {self.args.layerwise_lr}")
        print(f"  Warmup epochs:    {self.args.warmup_epochs}")
        print(f"  Learning rate:    {self.args.lr}")
        print(f"  Weight decay:     {self.args.weight_decay}")
        print(f"  Grad clip:        {self.args.grad_clip}")
        batch_size = opt.data_params["batch_size"]
        effective_bs = batch_size * self.world_size * self.args.accumulate_steps
        print(f"  Accum steps:      {self.args.accumulate_steps}")
        print(f"  Effective batch:  {effective_bs} "
              f"({batch_size} x {self.world_size} GPUs x {self.args.accumulate_steps} accum)")
        if self.args.patience > 0:
            print(f"  Early stopping:   patience={self.args.patience}, "
                  f"min_delta={self.args.min_delta}")
        else:
            print(f"  Early stopping:   disabled")
        print(f"  Augmentation:     {self.args.augment}")
        print(f"  Delta (DC wt):    {self.args.delta}")
        if self.args.dc_warmup_hold > 0 or self.args.dc_warmup_ramp > 0:
            dc_conf_start = getattr(self.args, 'dc_confidence_start', 0.7)
            print(f"  DC warmup:        hold={self.args.dc_warmup_hold}, "
                  f"ramp={self.args.dc_warmup_ramp}, "
                  f"conf_start={dc_conf_start:.2f}→{self.args.dc_confidence_threshold:.2f}")
        if self.args.dc_confidence_threshold > 0:
            print(f"  DC confidence:    {self.args.dc_confidence_threshold}")
        if self.args.balanced_sampling:
            print(f"  Balanced sample:  True")
        print(f"  SC class weight:  {self.args.sc_class_weight}")
        print(f"  NonSig boost:     {self.args.boost_nonsig}")
        print(f"  Focal loss:       {self.args.focal_loss}"
              + (f" (gamma={self.args.focal_gamma})" if self.args.focal_loss else ""))
        print(f"  Num classes:      {self.num_classes}")
        if self.args.config:
            print(f"  YAML config:      {self.args.config}")
        if getattr(self.args, 'temporal_encoder_layers', None) is not None:
            print(f"  Temporal enc layers: {self.args.temporal_encoder_layers}")
        if getattr(self.args, 'temporal_heads', None) is not None:
            print(f"  Temporal heads:      {self.args.temporal_heads}")
        if getattr(self.args, 'spatial_encoder_layers', None) is not None:
            print(f"  Spatial enc layers:  {self.args.spatial_encoder_layers}")
        if getattr(self.args, 'spatial_decoder_layers', None) is not None:
            print(f"  Spatial dec layers:  {self.args.spatial_decoder_layers}")
        print(f"  Parameters:       {total_params:,} total, {trainable_params:,} trainable")
        if self.args.seed is not None:
            print(f"  Seed:             {self.args.seed}")
        if self.args.eos_coef is not None:
            print(f"  eos_coef:         {self.args.eos_coef}")
        if self.args.num_workers > 0:
            print(f"  Num workers:      {self.args.num_workers}")
        if self.args.patient_split:
            print(f"  Patient split:    True (seed={self.args.split_seed})")
        if self.writer is not None:
            print(f"  TensorBoard:      {self.writer.log_dir}")
        print("=" * 60)


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
