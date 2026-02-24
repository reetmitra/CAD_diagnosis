"""Generate dummy NIfTI data for testing the SC-Net training pipeline."""

import os
import numpy as np
import nibabel as nib


def generate_dummy_dataset(output_root, num_samples=13, volume_shape=(64, 64, 256),
                           num_classes=3):
    """Generate synthetic NIfTI volumes and label files.

    Args:
        output_root: Directory to write volumes/ and labels/ into.
        num_samples: Total number of samples (80% train, 20% val by default).
        volume_shape: Shape of each NIfTI volume (H, W, D) — note NIfTI convention.
                      After transpose in data loader this becomes (D, H, W) = (256, 64, 64).
        num_classes: Number of lesion classes (excluding background=0).
    """
    vol_dir = os.path.join(output_root, 'volumes')
    lbl_dir = os.path.join(output_root, 'labels')
    os.makedirs(vol_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(num_samples):
        # Generate volume with realistic HU range for cardiac CT
        volume = rng.normal(loc=100, scale=200, size=volume_shape).astype(np.float32)
        volume = np.clip(volume, -1024, 2000)

        affine = np.eye(4)
        nii = nib.Nifti1Image(volume, affine=affine)
        nib.save(nii, os.path.join(vol_dir, f'sample_{i:04d}.nii'))

        # Generate per-slice labels: 0=background, 1..num_classes=lesion types
        depth = volume_shape[2]  # 256 slices after transpose
        labels = np.zeros(depth, dtype=np.int32)

        # Insert 1-3 random lesion segments
        num_lesions = rng.integers(1, 4)
        for _ in range(num_lesions):
            cls = int(rng.integers(1, num_classes + 1))
            start = int(rng.integers(10, depth - 30))
            length = int(rng.integers(5, 25))
            end = min(start + length, depth)
            labels[start:end] = cls

        labels_str = '\n'.join(str(x) for x in labels)
        with open(os.path.join(lbl_dir, f'sample_{i:04d}.txt'), 'w') as f:
            f.write(labels_str)

    print(f"Generated {num_samples} samples in {output_root}")
    print(f"  Volumes: {vol_dir}/ ({num_samples} .nii files)")
    print(f"  Labels:  {lbl_dir}/ ({num_samples} .txt files)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate dummy data for SC-Net')
    parser.add_argument('--output', type=str, default='./dummy_data',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=13,
                        help='Number of samples to generate')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of lesion classes')
    args = parser.parse_args()

    generate_dummy_dataset(args.output, args.num_samples, num_classes=args.num_classes)
