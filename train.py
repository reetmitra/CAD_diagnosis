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
import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from framework import sc_net_framework
from config import opt
from scheduler_utils import LinearWarmupCosineDecay, ModelEMA, build_param_groups
from eval import od_predictions_to_artery_level, targets_to_artery_level, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='SC-Net Training')

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

    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation for the training set')

    args = parser.parse_args()

    # Resolve negation flags
    if args.no_amp:
        args.amp = False
    if args.no_layerwise_lr:
        args.layerwise_lr = False
    if args.no_ema:
        args.ema = False

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
        self.num_classes = 3 if args.pattern == 'pre_training' else 6

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
        fw = sc_net_framework(
            pattern=self.args.pattern,
            state_dict_root=self.args.pretrained,
            data_root=self.args.data_root,
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
                num_classes=self.num_classes,
            )
        else:
            train_dataset = fw.dataLoader_train.dataset

        eval_dataset = fw.dataLoader_eval.dataset
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

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=shuffle_train, sampler=self.train_sampler,
            collate_fn=aug.collate_fn,
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size,
            shuffle=False, sampler=eval_sampler,
            collate_fn=aug.collate_fn,
        )

    def setup_optimizer(self):
        """Create optimizer with optional layer-wise LR and warmup+cosine scheduler."""
        # Param groups
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

        self.scheduler = LinearWarmupCosineDecay(
            self.optimizer,
            max_epochs=self.args.epochs,
            warmup_epochs=self.args.warmup_epochs,
        )

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

    def train_one_epoch(self, epoch):
        """Run one training epoch. Returns average loss."""
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    od_outputs, sc_outputs = self.model(images)
                    loss_dict = self.loss_fn(od_outputs, sc_outputs, targets)
                    loss = loss_dict['total']

                self.scaler.scale(loss).backward()

                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.args.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                od_outputs, sc_outputs = self.model(images)
                loss_dict = self.loss_fn(od_outputs, sc_outputs, targets)
                loss = loss_dict['total']

                loss.backward()

                if self.args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.args.grad_clip)

                self.optimizer.step()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item()
            num_batches += 1

            if self.is_main and (batch_idx + 1) % self.args.print_every == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} (OD: {loss_dict['od'].item():.4f} "
                      f"SC: {loss_dict['sc'].item():.4f} DC: {loss_dict['dc'].item():.4f}) "
                      f"Avg: {avg_loss:.4f}")

        return total_loss / max(num_batches, 1)

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

        return val_loss

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

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - epoch_start

            if self.is_main:
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch, val_loss,
                        os.path.join(self.args.checkpoint_dir, 'best_model.pth'),
                    )
                    print(f"  -> New best model saved (val_loss={val_loss:.4f})")

                # Periodic checkpoint
                if (epoch + 1) % self.args.save_every == 0:
                    self.save_checkpoint(epoch, val_loss)

        # Final checkpoint
        if self.is_main:
            self.save_checkpoint(
                self.args.epochs - 1,
                val_loss,
                os.path.join(self.args.checkpoint_dir, 'final_model.pth'),
            )
            print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

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
        print(f"  Augmentation:     {self.args.augment}")
        print(f"  Num classes:      {self.num_classes}")
        print(f"  Parameters:       {total_params:,} total, {trainable_params:,} trainable")
        print("=" * 60)


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
