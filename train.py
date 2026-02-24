"""Training script for SC-Net (Spatio-Temporal Contrast Network)."""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn

from framework import sc_net_framework
from config import opt


def parse_args():
    parser = argparse.ArgumentParser(description='SC-Net Training')
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
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            # Prefer GPU with most free memory
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


def train_one_epoch(model, loss_fn, dataloader, optimizer, device, epoch,
                    grad_clip, print_every):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        od_outputs, sc_outputs = model(images)
        loss = loss_fn(od_outputs, sc_outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % print_every == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} Avg: {avg_loss:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        od_outputs, sc_outputs = model(images)
        loss = loss_fn(od_outputs, sc_outputs, targets)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Initialize framework
    fw = sc_net_framework(
        pattern=args.pattern,
        state_dict_root=args.pretrained,
        data_root=args.data_root,
    )

    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)
    train_loader = fw.dataLoader_train
    eval_loader = fw.dataLoader_eval

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nStarting {args.pattern} for {args.epochs} epochs")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(eval_loader)}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Grad clip:     {args.grad_clip}")
    print()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device,
            epoch, args.grad_clip, args.print_every
        )

        val_loss = evaluate(model, loss_fn, eval_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.checkpoint_dir, 'best_model.pth')
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, val_loss,
        os.path.join(args.checkpoint_dir, 'final_model.pth')
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
