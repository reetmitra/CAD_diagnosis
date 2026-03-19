import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval import _load_model_from_checkpoint, get_device
from visualize import load_volume_and_labels, get_file_pairs
from functions import normalize_ct_data, normalize_ct_data_multiwindow, DEFAULT_MULTI_WINDOWS


def parser_args():
    parser = argparse.ArgumentParser(description="3D Grad-CAM for SC-Net")
    parser.add_argument('--data_root', type=str, required=True, help="Dataset root dir")
    parser.add_argument('--pattern', type=str, default='testing', help="Dataset split")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument('--model_pattern', type=str, default='fine_tuning', choices=['pre_training', 'fine_tuning'])
    parser.add_argument('--device', type=str, default='auto', help="Device")
    parser.add_argument('--multi_window', action='store_true', help="Use multi-window inference")
    parser.add_argument('--output_dir', type=str, default='./viz_gradcam', help="Output directory")
    parser.add_argument('--target_class', type=int, default=2, help="Target class to visualize (e.g., 2=Significant)")
    parser.add_argument('--max_samples', type=int, default=0, help="Stop after N samples (0 = unlimited)")
    return parser.parse_args()


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, target_class_index, outputs):
        # outputs[0]['pred_logits'] shape: [1, 16, C+1]
        logits = outputs[0]['pred_logits'][0] # [16, C+1]
        
        # Find the query with the highest score for the target class
        target_score_per_query = logits[:, target_class_index]
        best_query_idx = torch.argmax(target_score_per_query)
        target_score = target_score_per_query[best_query_idx]

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]  # [C, D, H, W]
        activations = self.activations.cpu().data.numpy()[0] # [C, D, H, W]

        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2, 3)) # [C]

        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32) # [D, H, W]
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap


def main():
    args = parser_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)

    model = _load_model_from_checkpoint(
        args.checkpoint, args.model_pattern, device, args.data_root, args.multi_window)
    
    # Target the last 3D convolution block in the spatial branch
    target_layer = model.object_detection_framework.feature_3d._3d_extraction_blocks[-1]
    grad_cam = GradCAM3D(model, target_layer)

    pairs = get_file_pairs(args.data_root, args.pattern)
    print(f"Running Grad-CAM on '{args.pattern}' split, outputting to {args.output_dir}")

    saved = 0
    for vol_path, lbl_path, artery_id in pairs:
        if args.max_samples > 0 and saved >= args.max_samples:
            break

        vol, labels = load_volume_and_labels(vol_path, lbl_path)
        
        if hasattr(model, 'in_channels') and model.in_channels > 1:
            vol_norm = normalize_ct_data_multiwindow(vol, DEFAULT_MULTI_WINDOWS)
        else:
            vol_norm = normalize_ct_data(vol, hu_min=-150, hu_max=750)
            
        tensor = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        
        outputs = model(tensor)
        heatmap = grad_cam.generate_heatmap(args.target_class, outputs)
        
        # Interpolate heatmap to match original volume size
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0) # [1, 1, D', H', W']
        heatmap_resized = F.interpolate(heatmap_tensor, size=vol.shape, mode='trilinear', align_corners=False)
        heatmap_resized = heatmap_resized.squeeze().numpy() # [256, 64, 64]
        
        # Extact center slice (longitudinal view)
        strip = vol[:, 32, :].T
        strip_norm = normalize_ct_data(strip, hu_min=-150, hu_max=750)
        heatmap_slice = heatmap_resized[:, 32, :].T

        plt.figure(figsize=(12, 4))
        plt.imshow(strip_norm, cmap='gray', aspect='auto', origin='upper', vmin=0, vmax=1)
        plt.imshow(heatmap_slice, cmap='jet', aspect='auto', origin='upper', alpha=0.5)
        
        CLASS_NAMES = ['Healthy', 'Non-significant', 'Significant']
        target_name = CLASS_NAMES[args.target_class] if args.target_class < len(CLASS_NAMES) else f"Class {args.target_class}"
        plt.title(f"{artery_id} - Grad-CAM for {target_name}")
        plt.axis('off')
        
        out_path = os.path.join(args.output_dir, f"{artery_id}_gradcam_cls{args.target_class}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"  [{saved+1}] Saved {out_path}")
        saved += 1

if __name__ == "__main__":
    main()
