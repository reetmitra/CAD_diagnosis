import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from eval import _load_model_from_checkpoint, get_device
from visualize import load_volume_and_labels, get_file_pairs
from functions import normalize_ct_data, normalize_ct_data_multiwindow, DEFAULT_MULTI_WINDOWS


def parser_args():
    parser = argparse.ArgumentParser(description="Monte Carlo Dropout Uncertainty Estimation")
    parser.add_argument('--data_root', type=str, required=True, help="Dataset root dir")
    parser.add_argument('--pattern', type=str, default='testing', help="Dataset split")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument('--model_pattern', type=str, default='fine_tuning', choices=['pre_training', 'fine_tuning'])
    parser.add_argument('--device', type=str, default='auto', help="Device")
    parser.add_argument('--mc_samples', type=int, default=10, help="Number of MC forward passes")
    parser.add_argument('--multi_window', action='store_true', help="Use multi-window inference")
    parser.add_argument('--output_json', type=str, default='./uncertainty_results.json', help="Output json path")
    return parser.parse_args()


def enable_dropout(model):
    """Enable dropout layers during eval mode."""
    for m in model.modules():
        if 'Dropout' in m.__class__.__name__:
            m.train()


@torch.no_grad()
def estimate_uncertainty(model, volume, device, mc_samples=10):
    """Run MC Dropout and return mean probabilities and predictive entropy (uncertainty)."""
    # 1. Prepare tensor
    if hasattr(model, 'in_channels') and model.in_channels > 1:
        vol_norm = normalize_ct_data_multiwindow(volume, DEFAULT_MULTI_WINDOWS)
    else:
        vol_norm = normalize_ct_data(volume, hu_min=-150, hu_max=750)
    
    tensor = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Enable Dropout
    model.eval()
    enable_dropout(model)

    mc_probs = []
    for i in range(mc_samples):
        outputs = model(tensor)
        od_out_raw = outputs[0]  # Object detection branch
        
        # Taking max probability among the 16 queries for each class
        # logits shape: [1, 16, C+1]
        logits = od_out_raw['pred_logits'][0] # [16, C+1]
        probs = F.softmax(logits, dim=-1)     # [16, C+1]
        
        # Max probability across all proposals for each class
        max_probs = probs.max(dim=0)[0]       # [C+1]
        mc_probs.append(max_probs.cpu().numpy())

    mc_probs = np.stack(mc_probs) # [N, C+1]

    # Calculate Mean and Variance
    mean_probs = mc_probs.mean(axis=0)
    variance = mc_probs.var(axis=0)
    
    # Calculate Predictive Entropy: -sum( p * log(p) )
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9))

    return mean_probs, variance, entropy


def main():
    args = parser_args()
    device = get_device(args.device)

    model = _load_model_from_checkpoint(
        args.checkpoint, args.model_pattern, device, args.data_root, args.multi_window)
    
    pairs = get_file_pairs(args.data_root, args.pattern)
    print(f"Running MC Dropout on {len(pairs)} records with N={args.mc_samples} passes.")

    results = {}
    for vol_path, lbl_path, artery_id in pairs:
        vol, labels = load_volume_and_labels(vol_path, lbl_path)
        
        mean_probs, variance, entropy = estimate_uncertainty(model, vol, device, args.mc_samples)
        
        results[artery_id] = {
            'mean_probs': mean_probs.tolist(),
            'variance_probs': variance.tolist(),
            'entropy': float(entropy)
        }
        print(f"Processed {artery_id}: Entropy={entropy:.4f}")

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved {args.output_json}")

if __name__ == "__main__":
    main()
