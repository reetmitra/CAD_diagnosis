# v16 Fine-tuning Plan
**Date:** 29 April 2026 | **Starting point:** v15 checkpoint (epoch 149, F1=0.825 raw)

---

## 1. Where We Stand and What Remains

v15 is a strong baseline:
- Stenosis F1=0.825 (raw), Sig Recall=0.806, AUC=0.868
- Plaque F1=0.638 (calibrated), AUC=0.811
- Architecture is correct and stable — no collapse, all classes learned

**Remaining gaps:**

| Issue | Evidence | Root |
|-------|----------|------|
| Sig→Non-sig misses | 40/217 Sig arteries wrong (18.4%) | Loss does not penalise under-escalation hard enough |
| Healthy→Non-sig misses | 18/98 Healthy wrong (18.4%) | OD head fires spurious queries on clean vessels |
| Calibration hurts stenosis on test | F1 drops 0.825→0.736 after calibration | Calibration has no Sig recall constraint; thresholds overfit to val |
| Non-sig→Sig over-escalation | 20/163 Non-sig wrong (12.3%) | Coupled with the Sig class weight changes — needs monitoring |

**Target for v16:** Sig Recall ≥ 0.85 (raw), Stenosis F1 ≥ 0.840, Plaque F1 ≥ 0.640.

---

## 2. Changes — Three Targeted Interventions

### 2.1 Boost Significant class weight in SC loss

**Problem:** `compute_sc_class_weights` currently gives all lesion classes weight=1.5. In 6-class fine-tuning, Sig maps to indices 4,5,6 (S+Calc, S+NonCalc, S+Mix). These are not up-weighted relative to Non-sig indices 2,3 (NS+Calc, NS+NonCalc, NS+Mix) despite Sig being the clinically higher-priority class.

**Fix:** Add `boost_sig=True, sig_multiplier=2.0` to `compute_sc_class_weights`. Mirror the existing `boost_nonsig` pattern exactly — double the weight for Sig-related indices (4,5,6 in fine_tuning mode).

**Code change:** `optimization.py:compute_sc_class_weights` + `train.py` CLI arg `--boost_sig`.

**Expected effect:** Model penalises Sig→Non-sig misses 2× more → Sig recall improves, some risk of Non-sig→Sig over-escalation (monitor).

---

### 2.2 Increase ordinal weight 0.5 → 1.5

**Problem:** `OrdinalEMDLoss` penalises severity-direction errors (Sig→Non-sig, Non-sig→Healthy) but its current weight of 0.5 is too weak to overcome the cross-entropy signal. The 40 Sig→Non-sig misses are exactly the ordinal-direction errors this loss targets.

**Fix:** Set `ordinal_weight: 1.5` in config. No code change needed.

**Expected effect:** Combined with boost_sig, ordinal penalty specifically targets Sig under-escalation. The two levers are complementary: class weight acts at training time on the loss surface; ordinal loss acts on the error direction.

---

### 2.3 Add Sig-recall constraint to calibrate.py

**Problem:** Calibration on val optimises macro-F1 without any Sig recall floor, and the found thresholds (t_sig=0.287) push Sig recall down to 0.571 on test. The raw model (0.806) is better, making calibration counter-productive for stenosis.

**Fix:** Add `--constrain_sig_recall FLOAT` to `calibrate.py`, mirroring the existing `--constrain_nonsig_recall`. The constrained search will require Sig recall ≥ 0.70 (configurable), preventing the threshold search from sacrificing Sig recall for macro-F1.

**Code change:** `calibrate.py` — ~20 lines mirroring the existing Non-sig constrained path.

**Expected effect:** Calibrated Sig recall ≥ 0.70 (up from 0.581), enabling calibration to add value on top of the already-strong raw model rather than degrading it.

---

### 2.4 Minor: raise eos_coef 0.15 → 0.20

**Problem:** 18/98 Healthy arteries get spurious OD foreground predictions. `eos_coef` controls how much the OD loss penalises background-class predictions — a higher value makes the matcher prefer the ∅ (no-object) slot for clean vessels.

**Fix:** Set `eos_coef: 0.20` in config. No code change needed.

**Risk:** Very low — this was 0.20 in earlier configs and was only lowered to 0.15 in v12. The DC fix in v15 means we no longer need eos_coef to fight feedback-loop noise.

---

## 3. Code Changes Required

### 3.1 `optimization.py` — boost_sig in compute_sc_class_weights

```python
def compute_sc_class_weights(num_classes, boost_nonsig=False, nonsig_idx=2,
                              boost_sig=False, sig_idx=3, sig_multiplier=2.0):
    weights = torch.ones(num_classes + 1, dtype=torch.float32)
    weights[0] = 0.5       # background
    weights[1:] = 1.5      # all lesion classes
    if boost_nonsig and nonsig_idx <= num_classes:
        weights[nonsig_idx] = weights[nonsig_idx] * 2.0
    if boost_sig and sig_idx <= num_classes:
        # In fine_tuning (6 classes): sig_idx=3 maps to first Sig+plaque class.
        # All three Sig+plaque classes (indices 4,5,6) are boosted.
        # sig_idx=3 is the 1-based Sig slot; actual tensor indices = sig_idx, sig_idx+1, sig_idx+2
        # when num_classes=6 and Sig occupies the last 3 slots.
        for i in range(sig_idx, num_classes + 1):
            weights[i] = weights[i] * sig_multiplier
    return weights
```

> **Important check before implementing:** Confirm which weight tensor indices correspond to Sig+plaque classes in fine_tuning mode. The SC branch outputs 7 logits (6 classes + background). Background=index 0; Healthy=1; NS+Calc=2, NS+NonCalc=3, NS+Mix=4; S+Calc=5, S+NonCalc=6 — or an equivalent mapping. Verify against `framework.py` / `augmentation.py` label encoding before hardcoding `sig_idx`.

### 3.2 `train.py` — add --boost_sig CLI argument

```python
parser.add_argument('--boost_sig', action='store_true', default=False,
                    help='Double SC class weight for Significant stenosis classes')
```

And in `setup_model`:
```python
sc_class_weights = compute_sc_class_weights(
    num_classes=...,
    boost_nonsig=boost_nonsig,
    nonsig_idx=2,
    boost_sig=self.args.boost_sig,
    sig_idx=3,
)
```

### 3.3 `calibrate.py` — add --constrain_sig_recall

Mirror the existing `--constrain_nonsig_recall` path:
```python
parser.add_argument('--constrain_sig_recall', type=float, default=0.0,
                    help='If > 0, constrain calibration search to maintain Sig recall >= this value')
```

In the threshold search loop, add a guard:
```python
if args.constrain_sig_recall > 0:
    sig_recall = recall_score(y_true, y_pred, labels=[2], average=None)[0]
    if sig_recall < args.constrain_sig_recall:
        continue  # skip this threshold combination
```

---

## 4. What We Are NOT Changing

| Item | Reason |
|------|--------|
| Architecture | v14 backbone working well — SE gates, parallel 2D/3D, learnable pos enc |
| DC loss direction | GT-based OD→SC fix is correct and stable |
| lr=3.0e-5 | v12-proven; v15 confirmed stable |
| focal_gamma=2.0 | v8 showed gamma=3.0 destabilises SC branch |
| SWA start epoch 80 | Working well in v15 |
| Patient split / seed | Same split for valid comparison |
| boost_nonsig | Keep True — Non-sig recall already at 0.847 which is good |

---

## 5. v16 Config

```yaml
# configs/finetune_v16.yaml
# v15 base + Sig boost + stronger ordinal + Sig-recall-constrained calibration

pattern: fine_tuning
data_root: ./dataset/train

epochs: 250
lr: 3.0e-5
weight_decay: 1.0e-4
grad_clip: 0.1
warmup_epochs: 10

lr_schedule: cosine_warm_restarts
lr_t0: 60
lr_t_mult: 2

swa: true
swa_start_epoch: 80

dc_warmup_hold: 20
dc_warmup_ramp: 40
delta: 0.5
dc_confidence_threshold: 0.4
dc_confidence_start: 0.7

label_smoothing: 0.1
ordinal_weight: 1.5          # ← increased from 0.5; stronger ordinal penalty targets Sig under-escalation

patience: 100
min_delta: 0.001

balanced_sampling: true
sc_class_weight: true
boost_nonsig: true
boost_sig: true              # ← new; doubles weight for Sig+plaque SC classes
focal_loss: true
focal_gamma: 2.0
eos_coef: 0.20               # ← increased from 0.15; reduces spurious OD on Healthy vessels

accumulate_steps: 4

amp: true
ema: true
ema_decay: 0.999
layerwise_lr: true
augment: true
num_workers: 8
seed: 42
patient_split: true
split_seed: 42

temporal_encoder_layers: 4
temporal_heads: 8
spatial_encoder_layers: 4
spatial_decoder_layers: 4

checkpoint_dir: ./checkpoints_v16_finetune
save_every: 10
print_every: 1
log_dir: ./runs
log_every: 10
```

---

## 6. Pipeline

```bash
# Training
torchrun --nproc_per_node=2 --master_port=29509 train.py --distributed \
  --config configs/finetune_v16.yaml \
  --pretrained ./checkpoints_v15_finetune/best_model.pth

# Calibration — standard
python calibrate.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --output calibration_thresholds_v16.json --grid_steps 50

# Calibration — Sig-recall constrained
python calibrate.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --output calibration_thresholds_v16_sig_constrained.json \
  --grid_steps 50 --constrain_sig_recall 0.70

# Evaluation
python eval.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --data_split testing --detailed

python eval.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --data_split testing \
  --thresholds calibration_thresholds_v16_sig_constrained.json --use_constrained --detailed
```

> **Note on pretrained checkpoint:** Starting from `checkpoints_v15_finetune/best_model.pth` (epoch 149) rather than the v14 pre-trained backbone. This is a fine-tuning-on-fine-tuning approach — we are refining the already-fine-tuned model with stronger class weights. If this causes instability (loss spikes), fall back to `checkpoints_v14/best_model.pth`.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| boost_sig causes Non-sig→Sig over-escalation | Medium | Monitor Non-sig recall during training; if it drops below 0.70 reconsider |
| ordinal_weight=1.5 destabilises early training | Low | DC hold=20 gives warm-up buffer; ordinal loss is bounded |
| eos_coef=0.20 hurts Sig detection | Very low | OD queries for true Sig arteries are high-confidence; background threshold only affects marginal queries |
| Starting from v15 ft checkpoint causes mode collapse | Low | LR=3.0e-5 is appropriate; EMA + SWA provide stability |

---

## 8. Implementation Order

1. Verify Sig class indices in fine_tuning SC branch (check `augmentation.py` / `framework.py` label encoding)
2. Implement `boost_sig` in `optimization.py` + `train.py`
3. Implement `--constrain_sig_recall` in `calibrate.py`
4. Write `configs/finetune_v16.yaml`
5. Run 30/30 tests to verify no regressions
6. Launch training
