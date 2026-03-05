# CPR Visualization Tool Design
Date: 2026-03-05

## Purpose

A standalone `visualize.py` script for inspecting CPR (Curved Planar Reformation) images
alongside their ground truth annotations and, optionally, model predictions. Supports both
dataset QC (no model) and error analysis (with model checkpoint).

## Data Structure Summary

- **CPR volumes**: NIfTI files, shape `(64, 64, 256)` — 64×64 cross-sections at 256 positions
  along the vessel axis. HU range ~-977 to 1349. Stored in `dataset/{split}/volumes/`.
- **Labels**: Text files with 256 integer values (one per axial position) in
  `dataset/{split}/labels/`. Raw values:
  - `0` = background / Healthy
  - `1` = Non-significant stenosis
  - `2` = Significant stenosis
  - `4` = Calcified plaque
  - `5` = Non-calcified plaque
  - `6` = Mixed plaque
- **Bounding boxes**: Derived from contiguous segments of the same non-zero label value,
  represented as `(center, width)` in normalised [0,1] coordinates along the vessel axis.
- **Dataset class**: `cubic_sequence_data` in `augmentation.py`. Samples are split
  deterministically by file index using `train_ratio` (default 0.8).

## Output: One PNG per Artery

### Figure layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ Title: APNHC00002_LAD  |  GT Stenosis: Non-significant               │
│        (if model) Pred Stenosis: Significant  [WRONG]                │
├──────────────────────────────────────────────────────────────────────┤
│  Row 1: Longitudinal CPR strip  (vessel axis = x-axis, 256 px wide)  │
│    - Grayscale CT image: volume[:, 32, :] transposed to (64, 256)    │
│    - Semi-transparent coloured axvspans for each GT label segment    │
│    - Dashed vertical line-pairs for predicted bounding boxes         │
│      (colour = predicted class, label printed above the box)         │
├────────────┬────────────┬────────────┬────────────────────────────── │
│  Row 2: Cross-sectional slices at centres of labelled segments       │
│   up to 4 slices, each 64×64, with a title showing slice index       │
│   and the raw label at that position                                 │
├──────────────────────────────────────────────────────────────────────┤
│  Row 3 (right-aligned): Legend panel                                 │
│    Stenosis GT colours | Plaque GT colours | Pred box style          │
└──────────────────────────────────────────────────────────────────────┘
```

### Colour scheme (GT bands = filled, alpha=0.35; Pred boxes = dashed)

| Class              | Colour   |
|--------------------|----------|
| Non-sig stenosis   | Yellow   |
| Significant sten.  | Red      |
| Calcified plaque   | Cyan     |
| Non-calcified      | Green    |
| Mixed plaque       | Magenta  |

### Output filenames

- Without checkpoint: `{artery_id}__sten_{GT}.png`
  e.g. `APNHC00002_LAD__sten_NonSig.png`
- With checkpoint: `{artery_id}__sten_{GT}_pred_{PRED}_{CORRECT|WRONG}.png`
  e.g. `APNHC00002_LAD__sten_NonSig_pred_Sig_WRONG.png`

## CLI Interface

```bash
# Ground truth only (no model)
python visualize.py \
  --data_root ./dataset/test \
  --pattern testing \
  --output_dir ./viz_gt

# With model predictions + calibration thresholds
python visualize.py \
  --data_root ./dataset/test \
  --pattern testing \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --thresholds calibration_thresholds_v7_constrained.json \
  --use_constrained \
  --output_dir ./viz_v7ft

# Only save incorrect predictions
python visualize.py ... --filter incorrect

# Only save a specific stenosis class
python visualize.py ... --filter sig

# Limit number of outputs (useful for quick checks)
python visualize.py ... --max_samples 50
```

### Arguments

| Argument          | Default       | Description                                      |
|-------------------|---------------|--------------------------------------------------|
| `--data_root`     | required      | Dataset root containing `volumes/` and `labels/`|
| `--pattern`       | `testing`     | `training` / `validation` / `testing`            |
| `--output_dir`    | `./viz`       | Directory to save PNGs                           |
| `--checkpoint`    | None          | Optional: model checkpoint for predictions       |
| `--thresholds`    | None          | Optional: calibration JSON from `calibrate.py`  |
| `--use_constrained` | False       | Use `constrained_stenosis_thresholds` from JSON  |
| `--filter`        | `all`         | `all` / `correct` / `incorrect` / `healthy` / `nonsig` / `sig` |
| `--max_samples`   | 0 (unlimited) | Stop after N arteries                            |
| `--device`        | `auto`        | `auto` / `cuda` / `cpu`                         |

## Implementation Plan (modules in visualize.py)

1. **`parse_args()`** — argparse setup
2. **`load_checkpoint(ckpt_path, device)`** — load model + framework (reuses `sc_net_framework` and `eval.py`'s checkpoint loading)
3. **`load_thresholds(path, use_constrained)`** — parse calibration JSON; returns `(stenosis_thresholds, plaque_thresholds)` or `(None, None)`
4. **`decode_label_segments(labels)`** — converts 256-length label array into list of `(start, end, raw_label)` tuples for contiguous non-zero runs
5. **`run_single_inference(model, volume_tensor, device)`** — single-sample forward pass; returns raw OD outputs
6. **`apply_calibration(od_outputs, thresholds, num_classes)`** — applies per-class threshold scaling and returns `(stenosis_pred, plaque_pred)`
7. **`render_artery(artery_id, volume, labels, od_outputs, stenosis_pred, plaque_pred, save_path)`** — full matplotlib figure construction and save
8. **`main()`** — iterate over dataset split, run inference if checkpoint given, render and save each artery

## Key Notes

- `cubic_sequence_data.__getitem__` returns a normalised tensor; for visualisation we load
  the raw NIfTI and apply the same `normalize_ct_data` windowing used during training
  (`window=[300, 900]`) to get a consistent grayscale image.
- The longitudinal strip uses `volume[:, 32, :]` (center cross-section row), transposed to
  `(64, 256)` so the vessel axis runs left-to-right.
- Prediction bounding boxes come from raw OD query outputs (before artery-level aggregation),
  allowing us to show all predicted segments, not just the winning class.
- No new dependencies — only `matplotlib`, `nibabel`, `numpy`, `torch` (all existing).
