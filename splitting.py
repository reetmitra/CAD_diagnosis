"""Patient-level data splitting utilities for SC-Net.

Ensures all arteries from the same patient stay in the same split
(train / val / test) to prevent data leakage.

Filename convention:  {PatientID}_{Artery}.nii  (or .nii.gz)
Example:  P001_LAD.nii, P001_LCX.nii, P002_RCA.nii.gz
"""

import os
from collections import defaultdict

import numpy as np


def get_patient_id(filename):
    """Extract patient ID from a filename like 'P001_LAD.nii' → 'P001'.

    Strips file extension(s) first, then splits on the *last* underscore
    so patient IDs that contain underscores are handled correctly.
    """
    base = filename
    # Remove .nii.gz or .nii
    if base.endswith('.nii.gz'):
        base = base[:-7]
    elif base.endswith('.nii'):
        base = base[:-4]
    # Split on last underscore: {PatientID}_{Artery}
    parts = base.rsplit('_', 1)
    return parts[0] if len(parts) > 1 else base


def patient_level_split(file_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split a sorted file list into train/val/test by patient.

    Args:
        file_list: list of filenames (sorted).
        train_ratio: fraction of patients for training (default 0.7).
        val_ratio: fraction of patients for validation (default 0.15).
            test_ratio is inferred as 1 - train_ratio - val_ratio.
        seed: random seed for reproducible patient ordering.

    Returns:
        (train_indices, val_indices, test_indices) — lists of integer
        indices into file_list.
    """
    # Group files by patient
    patient_to_indices = defaultdict(list)
    for idx, fname in enumerate(file_list):
        pid = get_patient_id(fname)
        patient_to_indices[pid].append(idx)

    patient_ids = sorted(patient_to_indices.keys())
    n_patients = len(patient_ids)

    # Shuffle patients deterministically
    rng = np.random.RandomState(seed)
    order = np.arange(n_patients)
    rng.shuffle(order)

    # Compute split boundaries
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    # Remaining go to test

    train_patients = order[:n_train]
    val_patients = order[n_train:n_train + n_val]
    test_patients = order[n_train + n_val:]

    def gather(patient_indices):
        indices = []
        for p_idx in patient_indices:
            pid = patient_ids[p_idx]
            indices.extend(patient_to_indices[pid])
        return sorted(indices)

    train_idx = gather(train_patients)
    val_idx = gather(val_patients)
    test_idx = gather(test_patients)

    # Report
    print(f"[patient_level_split] {n_patients} patients → "
          f"train={len(train_patients)} ({len(train_idx)} files), "
          f"val={len(val_patients)} ({len(val_idx)} files), "
          f"test={len(test_patients)} ({len(test_idx)} files)")

    return train_idx, val_idx, test_idx
