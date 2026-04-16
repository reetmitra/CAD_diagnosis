# SC-Net Pipeline Flowchart
*Updated: 2026-04-15 — reflects v12-ft best results & all Phase 17–19 architectural changes*

---

## 1. High-Level System Overview

```mermaid
flowchart TD
    RAW["Raw CTCA Data\n(NIfTI volumes + label .txt files)"]
    PREP["Data Preprocessing\n(augmentation.py)"]
    MODEL["SC-Net Model\n(architecture.py)"]
    LOSS["Loss Computation\n(optimization.py)"]
    TRAIN["Training Loop\n(train.py)"]
    EVAL["Evaluation & Inference\n(eval.py)"]
    CAL["Threshold Calibration\n(calibrate.py)"]
    VIZ["CPR Visualization\n(visualize.py)"]

    RAW --> PREP --> MODEL --> LOSS --> TRAIN
    TRAIN --> EVAL
    EVAL --> CAL
    EVAL --> VIZ

    style RAW fill:#dbeafe
    style MODEL fill:#fef9c3
    style LOSS fill:#fce7f3
    style TRAIN fill:#dcfce7
    style EVAL fill:#f3e8ff
    style CAL fill:#ffedd5
```

---

## 2. Data Pipeline

```mermaid
flowchart TD
    DIR["Dataset Directory\ndataset/train/"]
    VOL["volumes/*.nii.gz\n3D CT volumes"]
    LBL["labels/\n*_stenosis.txt\n*_plaque.txt\n(or legacy combined .txt)"]

    DIR --> VOL & LBL

    PAIR["_build_file_pairs()\nMatch volumes → labels"]
    VOL & LBL --> PAIR

    LOAD["cubic_sequence_data\nLoad NIfTI volume\n[H, W, D] → transpose → [D, H, W]"]
    PAIR --> LOAD

    MERGE["merge_new_labels()\nCombine stenosis + plaque\n→ 0–6 class label per slice"]
    LOAD --> MERGE

    RESIZE["data_resize()\nZoom to input_shape\n[256, 64, 64]"]
    MERGE --> RESIZE

    HU["normalize_ct_data()\nHU clip [-150, 750]\nScale to [0, 1]"]
    RESIZE --> HU

    REMAP{"Pattern?"}
    HU --> REMAP

    PRE["Remap: 0–2 classes\n(Pre-training)"]
    FINE["Remap: 0–5 classes\n(Fine-tuning)"]
    REMAP -->|pre_training| PRE
    REMAP -->|fine_tuning| FINE

    AUG{"Training?"}
    PRE & FINE --> AUG

    AUGYES["online_augment()\n± axial rotation ±15°\n± depth flip\n± HU intensity shift ±50"]
    AUGNO["No augmentation\n(val / test)"]
    AUG -->|yes| AUGYES
    AUG -->|no| AUGNO

    DET["detection_targets()\nConvert per-slice labels → 1D boxes\n[cx, w] ∈ [0,1] per lesion region"]
    AUGYES & AUGNO --> DET

    BATCH["collate_fn()\nStack → image [B, 1, 256, 64, 64]\ntargets: list of {labels, boxes}"]
    DET --> BATCH

    style DIR fill:#dbeafe
    style BATCH fill:#dcfce7
```

---

## 3. Model Architecture

```mermaid
flowchart TD
    INPUT["Input\n[B, 1, 256, 64, 64]"]

    subgraph TEMPORAL["Temporal Branch — Sampling Point Classification"]
        CUBES["_3d_cubes_selection()\nExtract 32 overlapping cubes\n[B, 32, 25, 25, 25]\nstep=8 vox along vessel axis"]
        CNN3D["Conv3d — 4 levels\n1→16→32→64→128 ch\nConv3d + BN + ReLU + MaxPool3d(2)\nper level"]
        FLAT_T["Flatten & Project\n128×d³ → 512 dim\nLinear projection"]
        POS_T["Learnable Position Embedding\n[1, 32, 512]\nEncodes proximal→distal ordering"]
        TRANS_T["Transformer Encoder\n4 layers, 8 heads, 512-dim\nSelf-attention over all 32 cubes"]
        HEAD_T["Classification MLP\n512 → 128 → num_classes\nLogit per cube"]

        CUBES --> CNN3D --> FLAT_T
        POS_T --> FLAT_T
        FLAT_T --> TRANS_T --> HEAD_T
    end

    subgraph SPATIAL["Spatial Branch — Object Detection (DETR-style)"]
        subgraph FEAT["feature_extraction_3d — 4 levels"]
            B3D["3D Path\n_3d_extraction_block\nConv3d blocks\n1→16→32→64→128 ch\nMaxPool3d(2) per level"]
            B2D["2D Path (PARALLEL)\n_2d_extraction_block\n4 orthogonal views extracted:\n  · mid horiz slice\n  · mid vert slice\n  · diagonal\n  · anti-diagonal\nEach: Conv2d + BN + ReLU + MaxPool2d\nWeighted sum → project back to 3D"]
            GATE["_FusionGate (SE-style)\nSqueeze: AdaptiveAvgPool3d → [B, 2C]\nFC: 2C→C/2→C, Sigmoid → α\nFuse: α·x_3d + (1-α)·x_2d\n(per-channel, learned)"]

            B3D & B2D --> GATE
        end

        FLAT_S["Flatten & Project\n128×d'×h'×w' → 512 dim\nConv3d(k=1) + Linear"]
        QUERY["16 Learnable Query Embeddings\n[16, 512]"]
        TRANS_S["DETR Transformer\nEncoder: 4 layers on image features\nDecoder: 4 layers, queries cross-attend\nto encoder output"]
        HEAD_CLS["Class Head MLP\n512→128→num_classes+1\n(includes ∅ no-object)\n[B, 16, C+1]"]
        HEAD_BOX["Box Head MLP\n512→128→2, Sigmoid\n→ [cx, w] ∈ [0,1]\n[B, 16, 2]"]

        FEAT --> FLAT_S
        QUERY --> TRANS_S
        FLAT_S --> TRANS_S
        TRANS_S --> HEAD_CLS & HEAD_BOX
    end

    INPUT --> TEMPORAL & SPATIAL

    SC_OUT["SC Output\npred_logits: [B, 32, num_classes]"]
    OD_OUT["OD Output\npred_logits: [B, 16, C+1]\npred_boxes:  [B, 16, 2]"]

    HEAD_T --> SC_OUT
    HEAD_CLS & HEAD_BOX --> OD_OUT

    style INPUT fill:#dbeafe
    style SC_OUT fill:#fef9c3
    style OD_OUT fill:#fef9c3
    style GATE fill:#fce7f3
```

---

## 4. Loss Computation

```mermaid
flowchart TD
    SC_OUT["SC Output\n[B, 32, num_classes]"]
    OD_OUT["OD Output\n[B, 16, C+1] + [B, 16, 2]"]
    GT["Ground Truth\ntargets: {labels, boxes}"]

    subgraph OD_LOSS["Object Detection Loss  L_OD"]
        HUN["HungarianMatcher\nMinimize cost matrix:\n  cost_class + 5·cost_L1 + 2·cost_GIoU\nOne-to-one assignment:\n  16 queries ↔ GT boxes"]
        L_CLS["Class CE Loss\nF.cross_entropy(logits, matched_labels\n  weight=class_weights\n  eos_coef=0.2 for ∅ class)"]
        L_BOX["Box Regression\n5.0 × L1(pred_box, gt_box)\n+ 2.0 × (1 - GIoU(pred, gt))\ndivided by num_matched_boxes"]

        HUN --> L_CLS & L_BOX
    end

    subgraph SC_LOSS["Sampling Point Loss  L_SC"]
        L_CE["CE or Focal Loss\nPer-cube classification\nFocal: (1-p_t)^γ · CE, γ=2.0\nClass weights: bg=0.5, lesion=1.5"]
        L_EMD["OrdinalEMDLoss (optional)\nEarth Mover Distance:\n  penalizes Healthy↔Sig > Healthy↔NonSig\nλ_ordinal ∈ [0,1]"]
        L_CE --> L_SC_TOTAL["L_SC = L_CE + λ_ord·L_EMD"]
        L_EMD --> L_SC_TOTAL
    end

    subgraph DC_LOSS["Dual-Task Contrastive Loss  δ·L_DC"]
        SC2OD["sc2od_targets()\nReshape SC preds [B,32]\n→ pseudo OD boxes"]
        OD2SC["od2sc_targets()\nReshape OD preds boxes\n→ pseudo SC labels [B,32]"]

        DC_MODE{"soft_dc?"}
        HARD["Hard CE on pseudo-labels\nRun L_OD on SC→OD targets\n+ L_SC on OD→SC targets"]
        SOFT["Soft KL-Divergence\nBuild soft prob distributions\nfrom OD logits at each point\nKL(log_softmax(SC) ‖ soft_target)\nTemperature T anneals 3.0→1.0"]

        SC2OD & OD2SC --> DC_MODE
        DC_MODE -->|no| HARD
        DC_MODE -->|yes| SOFT

        WARMUP["DC Warmup Schedule\nHold δ=0 for dc_warmup_hold epochs\nLinear ramp: 0→δ over dc_warmup_ramp epochs"]
        HARD & SOFT --> WARMUP
    end

    OD_OUT & GT --> OD_LOSS
    SC_OUT & GT --> SC_LOSS
    SC_OUT & OD_OUT --> DC_LOSS

    TOTAL["L_total = L_OD + L_SC + δ·L_DC"]
    OD_LOSS --> TOTAL
    L_SC_TOTAL --> TOTAL
    DC_LOSS --> TOTAL

    style TOTAL fill:#fce7f3
    style WARMUP fill:#ffedd5
```

---

## 5. Training Loop

```mermaid
flowchart TD
    START["Start Training\ntrain.py — Trainer class"]

    INIT["Initialize\n· Model (pre-train or fine-tune weights)\n· AdamW optimizer\n  - backbone: 0.1×lr\n  - transformer: 0.5×lr\n  - heads: 1.0×lr\n· LR scheduler (warmup + cosine)\n· EMA (decay=0.9995)\n· GradScaler (AMP)"]
    START --> INIT

    EPOCH["For each epoch"]
    INIT --> EPOCH

    subgraph TRAIN_PHASE["Training Phase"]
        BATCH_T["Load batch\n[B, 1, 256, 64, 64] + targets"]
        FWD["Forward pass (AMP float16)\nod_outputs, sc_outputs = model(images)"]
        LOSS_C["Compute loss\nL_total = L_OD + L_SC + δ·L_DC"]
        BWD["scaler.scale(loss).backward()"]
        CLIP["Gradient clip (norm ≤ 0.1)"]
        OPT["scaler.step(optimizer)\nscaler.update()"]
        EMA_U["EMA.update(model)"]
        SCHED["scheduler.step()"]

        BATCH_T --> FWD --> LOSS_C --> BWD --> CLIP --> OPT --> EMA_U --> SCHED
    end

    EPOCH --> TRAIN_PHASE

    subgraph VAL_PHASE["Validation Phase (each epoch)"]
        BATCH_V["Load val batch\nmodel.eval(), no_grad"]
        FWD_V["Forward pass\nod_outputs = model(images)"]
        AGG["od_predictions_to_artery_level()\nMax logits across 16 queries\n→ artery-level prediction"]
        METRICS["compute_metrics()\nACC, Precision, Recall, F1, Specificity\nStenosis & Plaque (3-class each)"]
        BEST{"Stenosis F1\n> best_f1?"}
        SAVE["Save checkpoint\nbest_model.pth\nReset patience counter"]
        PATIENCE["patience_counter += 1"]
        STOP{"counter ≥ patience?"}
        DONE["Early stop"]

        BATCH_V --> FWD_V --> AGG --> METRICS --> BEST
        BEST -->|yes| SAVE
        BEST -->|no| PATIENCE
        PATIENCE --> STOP
        STOP -->|yes| DONE
        STOP -->|no| EPOCH
    end

    TRAIN_PHASE --> VAL_PHASE

    SWA{"SWA enabled\n& epoch ≥ swa_start?"}
    SAVE --> SWA
    SWA -->|yes| SWA_U["swa_model.update_parameters(model)"]
    SWA -->|no| EPOCH
    SWA_U --> EPOCH

    style DONE fill:#fce7f3
    style SAVE fill:#dcfce7
```

---

## 6. Evaluation & Inference Pipeline

```mermaid
flowchart TD
    CKPT["Load Checkpoint\nbest_model.pth"]
    LOAD_M["Initialize model\nLoad state dict\nmodel.eval()"]
    CKPT --> LOAD_M

    TEST["Test DataLoader\ncubic_sequence_data (test split)"]
    LOAD_M & TEST --> LOOP

    subgraph LOOP["Inference Loop"]
        FWD_I["Forward pass\nod_outputs, sc_outputs = model(image)"]
        CONV_P["od_predictions_to_artery_level()\nMax or mean across 16 queries\n→ [B, C+1] logits per artery"]
        CONV_T["targets_to_artery_level()\nAggregate per-slice GT\n→ artery-level class"]
        TTA{"TTA enabled?"}
        TTA_RUN["Generate K augmented versions\nRun inference on each\nAverage logits"]
        ENS{"Ensemble?"}
        ENS_RUN["Average logits across N models"]

        FWD_I --> CONV_P
        CONV_P --> TTA
        TTA -->|yes| TTA_RUN --> ENS
        TTA -->|no| ENS
        ENS -->|yes| ENS_RUN
    end

    THR{"Calibrated\nthresholds?"}
    ENS_RUN & ENS -->|no| THR
    THR -->|yes| CAL_DEC["pred = argmax(p_i / t_i)\nPer-class threshold scaling"]
    THR -->|no| ARG["pred = argmax(p_i)\nStandard argmax"]

    METRICS2["compute_metrics()\nPer-class: ACC, Precision, Recall, F1\nSpecificity, Confusion Matrix, AUC-ROC"]
    CAL_DEC & ARG --> METRICS2

    OUT["Output\nJSON results + plots"]
    METRICS2 --> OUT

    style CKPT fill:#dbeafe
    style OUT fill:#dcfce7
```

---

## 7. Threshold Calibration

```mermaid
flowchart TD
    VAL_P["Validation Set Predictions\nSoftmax probabilities per artery"]
    GT_C["Ground Truth Labels"]

    VAL_P & GT_C --> GRID

    subgraph GRID["Grid Search"]
        G2D["2D Search (default)\nt_healthy ∈ [0.1, 3.0]\nt_sig     ∈ [0.05, 1.5]\nt_nonsig  = 1.0 (fixed)"]
        G3D["3D Constrained Search\nAll 3 thresholds searched\nConstraint: NonSig Recall ≥ target"]
        OPT_F1["Optimize macro-F1\npred = argmax(p / t)"]

        G2D & G3D --> OPT_F1
    end

    JSON["calibration_thresholds.json\n{healthy: t_h, nonsig: t_n, sig: t_s}"]
    OPT_F1 --> JSON

    APPLY["eval.py --thresholds calibration_thresholds.json\nApply to test set"]
    JSON --> APPLY

    style JSON fill:#ffedd5
    style APPLY fill:#dcfce7
```

---

## 8. Current Best Results — v12-ft

*Checkpoint: `checkpoints_v12_finetune/best_model.pth` (epoch 250)*
*Calibration: constrained — thresholds [H=2.80, NS=0.65, Sig=0.20]*

### Stenosis Classification (3-class)

| Class | F1 | Recall |
|---|---|---|
| Healthy | 0.868 | — |
| Non-significant | 0.613 | 0.639 |
| Significant | 0.735 | 0.733 |
| **Overall** | **0.739** | **0.736** |

Overall ACC: **0.736** · Precision: **0.743** · Specificity: **0.867**

### Plaque Composition (3-class)

| Class | F1 |
|---|---|
| Calcified | 0.790 |
| Non-calcified | 0.500 |
| Mixed | 0.214 |
| **Overall** | **0.502** |

### Version History (Best Calibrated Results)

| Version | Sten F1 | Sten ACC | Sten AUC | Sig Rec | NonSig Rec | Plaque F1 |
|---|---|---|---|---|---|---|
| v1 (pretrain) | 0.413 | 0.702 | 0.554 | — | — | 0.100 |
| v6-ft | 0.393 | 0.435 | 0.604 | 0.553 | — | 0.181 |
| v7-ft | 0.585 | 0.580 | 0.713 | 0.595 | 0.581 | 0.463 |
| v9-ft | 0.643 | 0.645 | 0.803 | 0.456 | 0.456 | 0.488 |
| **v12-ft** | **0.739** | **0.736** | — | **0.733** | **0.639** | **0.502** |

---

## 9. Key Architectural Changes (Phase 17–19, v13-ready)

```mermaid
flowchart LR
    subgraph OLD["Before (v12 and earlier)"]
        O1["2D stream fed x_3d\nas input at levels i>1\n(bug — streams identical)"]
        O2["Scalar _3d_weight fusion\nfixed blend: 0.75·3D + 0.25·2D\n(same for all channels)"]
        O3["No DC temperature\nfixed softmax temperature"]
    end

    subgraph NEW["After (Phase 17–19, v13)"]
        N1["2D stream fed x_2d\nas input at all levels\n(truly independent streams)"]
        N2["SE Fusion Gate\nlearned per-channel α\nα·x_3d + (1-α)·x_2d\n+220K params"]
        N3["DC temperature annealing\nT: 3.0 → 1.0 over dc_warmup_ramp\nsofter pseudo-labels early on"]
    end

    O1 -->|fixed| N1
    O2 -->|replaced| N2
    O3 -->|added| N3

    style OLD fill:#fce7f3
    style NEW fill:#dcfce7
```

---

## 10. File Map

```
CAD_diagnosis/
├── architecture.py       Model: spatio_temporal_semantic_learning, temporal, spatial, FusionGate
├── framework.py          High-level API: model init, dataset loading, loss setup
├── augmentation.py       Dataset class, online augment, clinically_credible_augmentation
├── config.py             DefaultConfig: all hyperparameters
├── train.py              Trainer class: full training loop + CLI args
├── optimization.py       Loss: L_OD + L_SC + δ·L_DC, OrdinalEMD, FocalLoss
├── scheduler_utils.py    LinearWarmupCosineDecay, CosineAnnealingWR, ModelEMA, build_param_groups
├── functions.py          HungarianMatcher, normalize_ct_data, _3d_cubes_selection, box utils
├── splitting.py          patient_level_split (prevent data leakage)
├── eval.py               Evaluation pipeline: metrics, TTA, ensemble, thresholds
├── calibrate.py          Per-class threshold grid search
├── visualize.py          CPR visualization, dual-bar (stenosis + plaque), paper Fig.3 style
├── cross_validate.py     Patient-level K-Fold cross validation
├── uncertainty.py        Monte Carlo Dropout uncertainty estimation
├── gradcam.py            3D Grad-CAM visualization on spatial backbone
├── generate_dummy_data.py Synthetic NIfTI volumes for pipeline testing
├── configs/
│   ├── pretrain_default.yaml
│   ├── finetune_default.yaml
│   ├── finetune_v9.yaml
│   └── finetune_v13.yaml   ← next run (Phase 17–19 changes)
└── tests/
    └── test_matcher.py     HungarianMatcher + class weight unit tests
```
