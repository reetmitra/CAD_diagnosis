import os
import random
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from einops import rearrange
import nibabel as nib
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import zoom, rotate

from architecture import spatio_temporal_semantic_learning
import functions as funcs
import optimization as opt_fn
from config import opt

class clinically_credible_augmentation(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape

    def data_resize(self, original_data):
        data, label = original_data['cpr_volume'], original_data['label']
        inp_size = data.shape

        if data.shape[0] == self.input_shape[0]:
            return original_data

        data_zoom_factors = (
        self.input_shape[0] / inp_size[0], self.input_shape[1] / inp_size[1], self.input_shape[2] / inp_size[2])
        resized_data = zoom(data, data_zoom_factors, order=3)
        label_zoom_factors = (self.input_shape[0] / inp_size[0])
        resized_label = zoom(label, label_zoom_factors, order=1)
        return {"cpr_volume": resized_data, "label": resized_label}

    def data_generator(self, foreground_data, background_data, blend_margin=3):
        foreground_data = self.data_resize(foreground_data)
        background_data = self.data_resize(background_data)

        f_data, f_label = foreground_data['cpr_volume'], foreground_data['label']
        b_data, b_label = background_data['cpr_volume'], background_data['label']

        ret_data = np.full((256, 64, 64), -1024, dtype=np.float64)
        ret_label = f_label
        b_indices = np.where(b_label == 0)[0]
        ret_data[b_indices, :, :] = b_data[b_indices, :, :]
        f_indices = np.where(f_label > 0)[0]

        # --- Intensity matching ---
        # Match foreground slice intensity to background at the boundary
        if len(f_indices) > 0 and len(b_indices) > 0:
            # Compute background mean/std from nearby slices
            b_mean = b_data[b_indices].mean()
            b_std = max(b_data[b_indices].std(), 1e-6)
            f_mean = f_data[f_indices].mean()
            f_std = max(f_data[f_indices].std(), 1e-6)
            # Normalize foreground to match background intensity distribution
            f_matched = (f_data[f_indices] - f_mean) * (b_std / f_std) + b_mean
            ret_data[f_indices, :, :] = f_matched
        else:
            ret_data[f_indices, :, :] = f_data[f_indices, :, :]

        # --- Soft blending at boundaries ---
        # Find transition points between foreground and background
        if blend_margin > 0 and len(f_indices) > 0:
            # Find contiguous foreground runs
            f_set = set(f_indices.tolist())
            for idx in f_indices:
                for offset in range(1, blend_margin + 1):
                    # Blend at the start of foreground (transition from background)
                    prev_idx = idx - offset
                    if prev_idx >= 0 and prev_idx not in f_set:
                        alpha = 0.5 * (1 + np.cos(np.pi * offset / blend_margin))  # cosine blend
                        ret_data[prev_idx] = (1 - alpha) * ret_data[prev_idx] + alpha * ret_data[idx]
                    # Blend at the end of foreground
                    next_idx = idx + offset
                    if next_idx < 256 and next_idx not in f_set:
                        alpha = 0.5 * (1 + np.cos(np.pi * offset / blend_margin))
                        ret_data[next_idx] = (1 - alpha) * ret_data[next_idx] + alpha * ret_data[idx]

        ret_label = np.where(ret_label > 0, ((ret_label - 1) % 3) + 1, ret_label)
        return {'cpr_volume': ret_data.astype(np.int32), 'label': ret_label}

    def read_data(self, volumes_file, labels_file):

        nii_file = nib.load(volumes_file)
        affine_matrix = nii_file.affine
        ret_volumes = nii_file.get_fdata()
        ret_volumes = ret_volumes.transpose(2, 0, 1)
        ret_volumes = np.array(ret_volumes)
        ret_labels = np.loadtxt(labels_file).astype(np.int32)

        return {'cpr_volume': ret_volumes, 'label': ret_labels}, affine_matrix

    def write_data(self, data_info, idx, augmented_root, affine_matrix):

        volumes = data_info['cpr_volume']
        labels = data_info['label']
        nii_image = nib.Nifti1Image(volumes, affine=affine_matrix)
        nib.save(nii_image, os.path.join(augmented_root, f'volumes/gen_{idx}.nii'))
        labels_str = '\n'.join(map(str, labels))
        with open(os.path.join(augmented_root, f'labels/gen_{idx}.txt'), 'w') as file:
            file.write(labels_str)

        return

    def forward(self, generated_num, original_root=r'original_data_root', augmented_root=r'augmented_data_root'):

        volumes_root = os.path.join(original_root, 'volumes/')
        labels_root = os.path.join(original_root, 'labels/')
        volumes_file_list = os.listdir(volumes_root)
        volumes_file_list = sorted(volumes_file_list)
        labels_file_list = os.listdir(labels_root)
        labels_file_list = sorted(labels_file_list)
        file_total = len(volumes_file_list)

        for i in range(generated_num):

            selected_f_idx, selected_d_idx = random.randint(0, file_total - 1), random.randint(0, file_total - 1)
            volumes_f_file = os.path.join(volumes_root, volumes_file_list[selected_f_idx])
            labels_f_file = os.path.join(labels_root, labels_file_list[selected_f_idx])
            volumes_d_file = os.path.join(volumes_root, volumes_file_list[selected_d_idx])
            labels_d_file = os.path.join(labels_root, labels_file_list[selected_d_idx])

            ori_f_data, affine_matrix = self.read_data(volumes_f_file, labels_f_file)
            ori_b_data, affine_matrix = self.read_data(volumes_d_file, labels_d_file)

            gen_data = self.data_generator(ori_f_data, ori_b_data)
            self.write_data(gen_data, i, augmented_root, affine_matrix)

        return


def online_augment(volume, labels):
    """Apply random online augmentations to a numpy volume and labels array.

    Args:
        volume: numpy array of shape [D, H, W]
        labels: numpy array of shape [D]

    Returns:
        Augmented volume and labels (numpy arrays).
    """
    # Random rotation along vessel axis (Z) — same angle for all axial slices
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        for d in range(volume.shape[0]):
            volume[d] = rotate(volume[d], angle, reshape=False, order=1)

    # Intensity jitter — uniform offset in [-50, +50] HU
    if random.random() < 0.5:
        offset = random.uniform(-50, 50)
        volume = volume + offset

    # Random flip along depth (axis 0), also reverse labels
    if random.random() < 0.5:
        volume = np.flip(volume, axis=0).copy()
        labels = np.flip(labels, axis=0).copy()

    # Gaussian noise injection
    if random.random() < 0.3:
        noise_std = random.uniform(5, 25)  # HU-scale noise
        noise = np.random.normal(0, noise_std, volume.shape).astype(volume.dtype)
        volume = volume + noise

    # Gaussian blur (simulate lower resolution / motion)
    if random.random() < 0.2:
        from scipy.ndimage import gaussian_filter
        sigma = random.uniform(0.3, 1.0)
        volume = gaussian_filter(volume, sigma=sigma)

    # Intensity scaling (contrast adjustment)
    if random.random() < 0.3:
        scale = random.uniform(0.85, 1.15)
        mean_val = volume.mean()
        volume = (volume - mean_val) * scale + mean_val

    # Random erasing (cutout) — mask a small random region to zero
    if random.random() < 0.2:
        D, H, W = volume.shape
        erase_d = random.randint(D // 16, D // 4)
        erase_h = random.randint(H // 4, H // 2)
        erase_w = random.randint(W // 4, W // 2)
        d0 = random.randint(0, max(D - erase_d, 1))
        h0 = random.randint(0, max(H - erase_h, 1))
        w0 = random.randint(0, max(W - erase_w, 1))
        volume[d0:d0+erase_d, h0:h0+erase_h, w0:w0+erase_w] = 0

    return volume, labels


def merge_new_labels(stenosis, plaque):
    """Merge separate stenosis (0-4) and plaque (0-3) arrays into combined (0-6).

    Maps:
      stenosis in {1,2} + plaque in {1,2,3} → combined in {1,2,3}
      stenosis in {3,4} + plaque in {1,2,3} → combined in {4,5,6}
    """
    combined = np.zeros_like(stenosis)
    nonsig = (stenosis >= 1) & (stenosis <= 2)
    sig    = stenosis >= 3
    combined[nonsig] = plaque[nonsig]      # 1, 2, or 3
    combined[sig]    = plaque[sig] + 3     # 4, 5, or 6
    return combined


def _build_file_pairs(volumes_root, labels_root):
    """Build (vol_path, sten_path_or_None, plaq_path_or_None, comb_path_or_None) tuples.

    Matches volumes to labels by stem name. Prefers new .nii.gz + separate label format.
    """
    lbl_set = set(os.listdir(labels_root))
    pairs = []
    seen_stems = set()
    for vf in sorted(os.listdir(volumes_root)):
        if vf.endswith('.nii.gz'):
            stem = vf[:-7]
        elif vf.endswith('.nii'):
            stem = vf[:-4]
        else:
            continue
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        sten_f = stem + '_stenosis.txt'
        plaq_f = stem + '_plaque.txt'
        comb_f = stem + '.txt'
        # Prefer new format: .nii.gz with separate stenosis/plaque labels
        if vf.endswith('.nii.gz') and sten_f in lbl_set and plaq_f in lbl_set:
            pairs.append((
                os.path.join(volumes_root, vf),
                os.path.join(labels_root, sten_f),
                os.path.join(labels_root, plaq_f),
                None,
            ))
        elif comb_f in lbl_set:
            # Fall back to old format: .nii with combined label
            old_vf = stem + '.nii'
            vol_path_candidate = os.path.join(volumes_root, old_vf)
            if os.path.exists(vol_path_candidate):
                vol_path = vol_path_candidate
            else:
                vol_path = os.path.join(volumes_root, vf)
            pairs.append((vol_path, None, None, os.path.join(labels_root, comb_f)))
    return pairs


class cubic_sequence_data(data.Dataset):
    def __init__(self, dataset_root, pattern='training', train_ratio=0.8, input_shape=[256,64,64], window=[300, 900], augment=False, num_classes=None, file_indices=None, multi_window=False):

        self.volumes_root = os.path.join(dataset_root, 'volumes/')
        self.labels_root = os.path.join(dataset_root, 'labels/')
        self.input_shape = input_shape
        self.window = [window[0] - window[1] / 2, window[0] + window[1] / 2]
        self.multi_window = multi_window
        if multi_window:
            self.multi_windows = funcs.DEFAULT_MULTI_WINDOWS
        else:
            self.multi_windows = None
        self.augment = augment
        self.pattern = pattern
        self.num_classes = num_classes

        # Build matched (vol, sten, plaq, comb) pairs instead of separate lists
        self.file_pairs = _build_file_pairs(self.volumes_root, self.labels_root)
        self.file_total = len(self.file_pairs)

        # file_indices overrides the default train_ratio-based split
        self.file_indices = file_indices
        if file_indices is not None:
            self.data_start = 0
            self.data_end = 0
            self.length = len(file_indices)
        elif pattern == 'all':
            self.data_start = 0
            self.data_end = self.file_total
            self.length = self.data_end - self.data_start
        elif pattern == 'training':
            self.data_start = 0
            self.data_end = int(self.file_total * train_ratio)
            self.length = self.data_end - self.data_start
        elif pattern == 'validation':
            self.data_start = int(self.file_total * train_ratio)
            self.data_end = int(self.file_total * (train_ratio + (1 - train_ratio) / 2))
            self.length = self.data_end - self.data_start
        else:  # 'testing'
            self.data_start = int(self.file_total * (train_ratio + (1 - train_ratio) / 2))
            self.data_end = self.file_total
            self.length = self.data_end - self.data_start
        return

    def read_data(self, vol_path, sten_path, plaq_path, comb_path):
        """Load volume and labels, handling both new (separate) and old (combined) formats.

        Args:
            vol_path: path to volume (.nii or .nii.gz)
            sten_path: path to stenosis label or None
            plaq_path: path to plaque label or None
            comb_path: path to combined label or None
        """
        # Load volume
        nii_file = nib.load(vol_path)
        vol = nii_file.get_fdata()
        if vol.shape[0] == vol.shape[1]:  # (H, W, D) → (D, H, W)
            vol = vol.transpose(2, 0, 1)

        # Resize volume to input_shape if needed
        target = self.input_shape
        if list(vol.shape) != target:
            zf = [target[i] / vol.shape[i] for i in range(3)]
            vol = zoom(vol, zf, order=1)

        # Load labels
        if comb_path is not None:
            labels = np.loadtxt(comb_path).astype(np.int32)
        else:
            sten = np.loadtxt(sten_path).astype(np.int32)
            plaq = np.loadtxt(plaq_path).astype(np.int32)
            labels = merge_new_labels(sten, plaq)

        # Resize labels to match vessel axis length
        if len(labels) != target[0]:
            labels = zoom(labels.astype(float), target[0] / len(labels), order=0).astype(np.int32)

        return np.array(vol), labels

    def detection_targets(self, labels_data):

        boxes, labels = [], []
        start, label, length, last = None, 0, self.input_shape[0], -1

        for i in range(labels_data.shape[0]):
            if start is not None:
                if labels_data[i] != last:
                    x1 = (start + 1) / length
                    x2 = min((i + 1) / length, 1.0)
                    center = (x1 + x2) / 2.0
                    width = x2 - x1
                    boxes.append([center, width])
                    labels.append(label - 1)
                    if labels_data[i] != 0:
                        start, label, last = i, labels_data[i], labels_data[i]
                    else:
                        start, label, last = None, 0, -1
                else:
                    continue
            else:
                if labels_data[i] == 0:
                    start, label, last = None, 0, -1
                else:
                    start, label, last = i, labels_data[i], labels_data[i]

        if start is not None:
            x1 = (start + 1) / length
            x2 = 1.0
            center = (x1 + x2) / 2.0
            width = x2 - x1
            boxes.append([center, width])
            labels.append(label - 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return {"labels": labels, "boxes": boxes}

    def __getitem__(self, index):

        if self.file_indices is not None:
            actual_index = self.file_indices[index]
        else:
            actual_index = index + self.data_start
        vol_path, sten_path, plaq_path, comb_path = self.file_pairs[actual_index]
        ret_volumes, ret_labels = self.read_data(vol_path, sten_path, plaq_path, comb_path)
        if self.augment and self.pattern == 'training':
            ret_volumes, ret_labels = online_augment(ret_volumes, ret_labels)
        # Remap labels when num_classes < max label (e.g. pre_training 3-class on 6-class data)
        if self.num_classes is not None:
            ret_labels = np.where(ret_labels > 0, ((ret_labels - 1) % self.num_classes) + 1, ret_labels)
        # Apply HU windowing
        if self.multi_windows is not None:
            ret_volumes = funcs.normalize_ct_data_multiwindow(ret_volumes, self.multi_windows)
            return {'image': torch.tensor(ret_volumes, dtype=torch.float32), 'target': self.detection_targets(ret_labels)}
        else:
            ret_volumes = funcs.normalize_ct_data(ret_volumes, hu_min=self.window[0], hu_max=self.window[1])
            return {'image': torch.tensor(ret_volumes, dtype=torch.float32), 'target': self.detection_targets(ret_labels)}

    def __len__(self):
        return self.length


def collate_fn(batch):

    images, targets = [], []
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    images = torch.stack(images, dim=0)

    return images, targets