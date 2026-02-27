Below is a step-by-step training plan to **replicate the paper's
SC-Net** (Spatio-temporal Contrast Network) and then **push performance
further in low-data settings**, written as a practical "do this next"
workflow. Everything here is anchored on what the paper actually does:
**(1) clinically-credible lesion augmentation, (2) spatio-temporal
dual-task learning (object detection + sampling-point classification),
(3) dual-task prediction-contrast optimization**, plus the
dataset/training setup they describe.

------------------------------------------------------------------------

## 1) Define exactly what you're trying to match (targets + outputs)

SC-Net is built around **two coupled predictions** from the same CPR
coronary segment:

1.  **Spatial task: Object detection on the CPR volume**

-   Detect lesion **RoIs** along the CPR and classify lesion category.
-   Implemented DETR-style: a fixed number of "queries" (paper uses
    **Q=16**) predicts up to 16 lesions per segment.

2.  **Temporal task: Sampling-point classification along the vessel**

-   Sample a sequence of locations along the vessel (paper uses **32
    cubes per CPR**, uniformly spaced) and classify each location's
    lesion/stenosis state.

**Your replication goal** should be: "Given a CPR segment, produce
lesion localization + lesion characterization + per-location
stenosis/plaque assessment, and keep them consistent with each other."

------------------------------------------------------------------------

## 2) Data pipeline plan (this is where most replications fail)

### 2.1 Build CPR volumes consistently

The paper trains on **CPR volumes of main coronary branches**,
reconstructed from CCTA (they report **1163 CPR volumes from 218
patients**).

Your pipeline should be deterministic and standardized:

-   **Coronary centerline extraction** (segmentation + centerline).

-   **CPR generation** along the centerline.

-   **Resample** to fixed tensor shapes (paper uses):

    -   CPR volume: **256 × 64 × 64**
    -   2D views: **256 × 64**

-   Normalize intensities in a stable, medical-reasonable way (e.g.,
    consistent HU windowing + scaling), and apply the same method
    everywhere.

**Sanity checks you must do before training:**

-   Randomly visualize CPRs across patients and confirm the vessel stays
    centered and continuous.
-   Plot distribution of vessel diameters/lengths after resampling (you
    want consistent scaling).
-   Confirm lesion annotations line up with CPR coordinates after
    reformatting.

### 2.2 Create the 4 "primary 2D views"

The spatial branch uses **four 2D projections** from the CPR volume
(sagittal, coronal, and two diagonal axes).

Plan:

-   For every CPR, generate the exact same four views in the same order.
-   Confirm visually that lesions are visible in at least one or two
    views (they often are).

### 2.3 Prepare labels for both tasks (make them convertible)

SC-Net's key trick is that the two tasks can supervise each other via
conversion functions **C(·)** and **C⁻¹(·)**.

So your labels must be structured so that:

-   From sampling-point labels, you can derive approximate RoIs (group
    consecutive abnormal points into segments).
-   From detected RoIs, you can label points inside those RoIs as
    abnormal (and assign categories).

Concretely:

-   **Sampling-point labels**: for each of the 32 points, store class
    label(s).
-   **Detection labels**: RoI representation + lesion category. The
    paper represents RoI with center coordinate and a "weight/width"
    style vector; they train with L1 + IoU losses.

### 2.4 Follow the paper's cube sampling

For the temporal branch, the paper samples:

-   **32 cubes per CPR**, interval **8 voxels**
-   Cube size **25 × 25 × 25**

Plan:

-   Implement this exactly first.
-   Add a visualization utility: show cube centers overlaid on the CPR
    centerline to confirm spacing and coverage.

------------------------------------------------------------------------

## 3) Clinically-credible data augmentation (replicate it exactly first)

This is a major reason SC-Net works with limited labels: they create
"more lesion variety" without hallucinating impossible anatomy.

### 3.1 Build two sets: Background B and Foreground F

-   **B (background set):** CPR volumes/regions treated as background
    (healthy context).
-   **F (foreground set):** lesion RoIs (foreground).

### 3.2 Create augmented samples A by overlaying lesions on backgrounds

They form augmented sample **a** by removing background at lesion
indices and inserting a lesion foreground there (overlay/recombine).

Your plan for "clinically credible" overlay:

-   Match the lesion foreground's **scale** to the target vessel
    diameter (avoid obviously mismatched plaques).
-   Match basic **intensity statistics** locally (so the paste doesn't
    create sharp artificial edges).
-   Keep it limited to "plaque appearance learning," not unrealistic
    stenosis geometry changes.

### 3.3 Two-stage training using augmentation (paper's recipe)

The paper explicitly does:

1.  **Pre-train on augmented set A** focusing on **plaque composition**
    learning.
2.  **Fine-tune on clinical set B** for both **plaque composition +
    stenosis degree**.

That sequencing matters---replicate it.

------------------------------------------------------------------------

## 4) Model build plan (match the architecture pieces, then tune)

### 4.1 Spatial semantic learning = DETR-style lesion detection

Inputs:

-   Full CPR volume (3D)
-   4 primary 2D views

Core steps to replicate:

1.  **3D-CNN** encodes CPR → feature map

2.  **2D-CNN** encodes each view → 2D features

3.  **Multi-view spatial relationship analysis**

    -   Lift 2D features into 3D "zero maps" at corresponding positions
    -   Weighted fusion into one spatial feature map

4.  **Transformer decoder with Q query embeddings**

    -   Paper uses **Q=16**, embedding dim 512

5.  Two heads per query:

    -   **RoI regression** (Sigmoid head)
    -   **Category classification** (Softmax head)

**Replication checklist:**

-   Start with Q=16.
-   Make sure your RoI parameterization matches your loss (IoU must be
    computable).

### 4.2 Temporal semantic learning = sequence classification along vessel

Inputs:

-   32 cubes per CPR (25³), sampled along vessel

Core steps:

1.  Shallow **3D-CNN** per cube → local feature
2.  Flatten + projection → per-location embedding (dim 512)
3.  **Transformer encoder** across locations to model correlations
4.  **MLP classifier** per location (Softmax)

**Replication checklist:**

-   Confirm the transformer sees locations in a consistent "proximal →
    distal" order.
-   Make sure your sampling points map cleanly back to CPR coordinates.

------------------------------------------------------------------------

## 5) Losses and training objective (this is the "secret sauce")

SC-Net trains: **Loverall = Lod + Lsc + Ldc**

### 5.1 Lod (object detection loss)

DETR-style:

-   Hungarian matching between predictions and ground truth
-   Classification loss + RoI loss (L1 + IoU)
-   Paper uses **λ_iou = 2**, **λ_L1 = 5**

### 5.2 Lsc (sampling-point classification loss)

-   Cross-entropy across the 32 points

### 5.3 Ldc (dual-task contrastive / prediction-contrast supervision)

This is not "contrastive embeddings" in the usual SimCLR sense; it's
**prediction-to-prediction supervision**:

-   Convert sampling-point predictions → RoIs and supervise detection
-   Convert detection predictions → point labels and supervise point
    classifier

**Practical training stability plan (highly recommended):**

-   **Warm-up phase:** train only Lod + Lsc for a short period (e.g.,
    first 10--20 epochs) so both heads produce non-random outputs.
-   **Ramp Ldc gradually:** linearly increase the weight of Ldc from 0 →
    1 over the next chunk of training. This usually prevents collapse
    where one weak task poisons the other early.

------------------------------------------------------------------------

## 6) Training schedule (replicate paper settings, then improve)

### 6.1 Data split and epochs (paper)

-   70% train (balanced across lesion categories)
-   Remaining 30% split into validation and test
-   Train **200 epochs**, keep best validation checkpoint

### 6.2 Stage-wise training plan (do this in order)

#### Stage A --- Pipeline verification (1--2 days)

Goal: prove the model *can* learn.

-   Train on a tiny subset (e.g., 5--10 CPRs).

-   Expect near-perfect training performance (overfit).

-   If you can't overfit, stop and fix:

    -   label alignment
    -   cube sampling alignment
    -   RoI formatting / IoU computation
    -   data normalization

#### Stage B --- Pre-train with clinically-credible augmentation A (paper step)

Goal: learn robust plaque appearance features with more diversity.

-   Build A using foreground F and background B overlays.
-   Supervise primarily plaque composition (as described).
-   Save the best checkpoint.

#### Stage C --- Fine-tune on real clinical data B (paper step)

Goal: final performance on real distributions.

-   Train full SC-Net using the combined objective.
-   Use Ldc warm-up + ramp strategy (stability).
-   Track both tasks + consistency between them.

### 6.3 What to track every epoch (don't skip this)

1.  Detection metrics:

-   # predicted lesions per CPR segment

-   localization quality (IoU distribution)

-   class confusion

2.  Sampling-point metrics:

-   per-point accuracy/F1
-   false negatives near true lesions (missed stenosis zones)

3.  Cross-task consistency metrics (SC-Net's spirit):

-   \% points inside predicted RoIs that are classified abnormal
-   \% predicted RoIs that overlap runs of abnormal points

------------------------------------------------------------------------

## 7) Evaluation plan (match the paper's reporting)

The paper reports artery-level:

-   ACC, Precision, Recall, F1, Specificity
-   for stenosis degree and plaque components

To mirror their "data-efficient" claim, you should evaluate at:

-   **100% training data**
-   **50% training data** (and optionally 25%/10% for your own study)

Also replicate their ablations:

-   Remove clinically-credible augmentation (CDA)
-   Remove spatial task or temporal task
-   Remove Ldc

If you reproduce the *direction* of these drops, you're very likely
aligned with the method.

------------------------------------------------------------------------

## 8) Where to improve beyond the paper (high-impact ideas)

These are ordered by "most likely to move the needle" in low-data
medical imaging.

### 8.1 Make augmentation more realistic (without breaking clinical credibility)

Keep their overlay idea, but improve its realism:

-   Local intensity matching (mean/variance match in a narrow band
    around the insert)
-   Soft blending at borders (avoid sharp paste seams)
-   Diameter-aware scaling rules (only paste plaques into similar
    caliber vessels)

### 8.2 Add self-supervised pretraining on *unlabeled* CPRs (huge for low data)

Before any labels:

-   Masked reconstruction of CPR volumes or cubes
-   View-consistency objectives between CPR and its 2D projections
-   Temporal order prediction along vessel (helps the transformer learn
    continuity)

Then fine-tune with SC-Net losses. This often gives a big jump when
labels are scarce.

### 8.3 Fix class imbalance explicitly

The paper motivates imbalance between healthy and lesion regions.
Practical upgrades:

-   Class-balanced sampling at CPR level (ensure each batch has lesions)
-   Loss reweighting or focal-style emphasis on hard lesion examples
-   Hard-negative mining for "looks-like-plaque but isn't" regions

### 8.4 Improve the dual-task coupling (make Ldc smarter)

Instead of using raw predictions as pseudo-labels:

-   Only use high-confidence predictions to supervise the other task
    (confidence threshold)
-   Use soft labels (probabilities) rather than hard argmax labels
-   Delay cross-supervision until both tasks reach minimum validation
    quality

### 8.5 Calibrate and measure reliability (important for clinical-style outputs)

Add:

-   Probability calibration (so confidence means something)
-   Uncertainty estimates (ensembles or dropout at inference)
-   Report: "high confidence coverage vs accuracy" curves

------------------------------------------------------------------------

## 9) A practical "weekly" execution roadmap (so you don't get lost)

### Week 1 --- Data + labels + sanity

-   CPR generation locked and reproducible
-   4 view generation correct
-   Cube sampling correct
-   Overfit test passes

### Week 2 --- Baseline SC-Net replication

-   Spatial branch training stable (Lod works)
-   Temporal branch training stable (Lsc works)
-   Joint training without Ldc stable

### Week 3 --- Add augmentation + 2-stage training

-   Implement B/F sets + overlay augmentation
-   Pretrain on A, fine-tune on B (paper recipe)

### Week 4 --- Add Ldc with stability controls

-   Warm-up + ramp
-   Cross-task consistency metrics improve
-   Run 50% vs 100% training comparisons

### Week 5+ --- Improvements

-   Stronger augmentation realism
-   Self-supervised pretraining
-   Confidence-filtered Ldc
-   Imbalance handling upgrades

------------------------------------------------------------------------

## 10) Final replication checklist (quick pass/fail)

You're "matching the paper" when:

-   You use the same core inputs (CPR + 4 views + 32 cubes) and shapes
    (256×64×64, 25³ cubes, Q=16).
-   You do the same two-stage training: pretrain on augmented A then
    fine-tune on B.
-   Your objective is Lod + Lsc + Ldc, and Ldc is implemented as
    prediction-contrast between tasks (via conversion).
-   Removing augmentation or Ldc causes a noticeable performance drop
    (ablation matches the paper's story).

------------------------------------------------------------------------

If you tell me what you currently have (even just in words)---e.g., "I
already have CPR volumes + point labels but no RoIs" or "I only have
lesion boxes but no centerline sampling labels"---I can adapt this plan
into a precise checklist for *your* situation (what to build, what to
skip, and what to approximate) while still staying faithful to SC-Net.

Absolutely --- here's a **proper, in-depth 7-day plan** designed to get
you as close as possible to replicating the paper's SC-Net within **one
week**, with clear daily deliverables and "if-stuck" fallbacks.

This plan is built around the paper's three core pieces:

-   **Clinically-credible data augmentation** (build augmented set A
    from background B + lesion foreground F, then **pretrain on A** and
    **fine-tune on B**)
-   **Spatio-temporal dual-task learning**: spatial **object
    detection** + temporal **sampling-point classification**
-   **Dual-task prediction-contrast optimization**: overall loss
    **Loverall = Lod + Lsc + Ldc**, where Ldc cross-supervises tasks via
    conversion functions C and C⁻¹

------------------------------------------------------------------------

# Before you start (rules for the week)

To succeed in 7 days, you need to be strict about scope:

### Non-negotiables (must replicate first)

-   Input formatting and sampling:

    -   CPR volume shape **256×64×64**, 2D views **256×64**
    -   **32** cubes, interval **8 voxels**, cube **25×25×25**
    -   **Q=16** detection queries
    -   Four 2D views: sagittal, coronal, and two diagonal projections

-   Training objective:

    -   **Loverall = Lod + Lsc + Ldc**
    -   Ldc uses conversions between tasks: **Ldc = Lod(C(ŷsc), ŷod) +
        Lsc(C⁻¹(ŷod), ŷsc)**

-   Paper hyperparams you should match initially:

    -   λIoU=2, λL1=5

### One allowed improvement (do only 1)

-   **Warm-up / ramp Ldc** (start at 0 then ramp up). This is the best
    single "stability" improvement in these mutual-supervision setups.

------------------------------------------------------------------------

# Day 1 --- Lock the data and labels (no training yet)

**Goal:** You should be 100% confident your inputs + labels are correct
and consistent across both tasks.

## Tasks (do in this order)

1.  **Create a "gold sample pack"** (10--20 CPRs)

-   Mix: no-lesion, single-lesion, multi-lesion, long continuous lesion
    region.
-   You'll use this pack every day for quick checks.

2.  **Verify the exact inputs the paper expects**

-   For each CPR sample, confirm you can produce:

    -   CPR volume tensor (target: 256×64×64)
    -   Four 2D views (target: 256×64)
    -   32 cubes of size 25×25×25 sampled every 8 voxels

3.  **Build dual labels that can convert both ways** SC-Net depends on
    converting:

-   sampling-point results → RoIs + categories via **C(·)**
-   RoIs → sampling-point categories via **C⁻¹(·)**

So you must define:

-   Sampling-point label format: for each of the 32 points, what class
    does it have?
-   Detection label format: lesion RoIs + categories (and "no object").

4.  **Write down your exact conversion rules (very important)** Keep it
    simple:

-   **C(point predictions → RoIs):**

    -   Group consecutive abnormal points into segments.
    -   Convert segment start/end into an RoI center + width (or
        equivalent).
    -   Assign category by majority vote or max confidence within the
        segment.

-   **C⁻¹(RoIs → point labels):**

    -   For each point, if its position lies inside an RoI interval →
        label it as that RoI class.
    -   If overlaps multiple RoIs → choose closest center or highest
        confidence.

## Day 1 Deliverables (must finish)

-   A saved "gold sample pack"

-   A one-page spec with:

    -   input shapes
    -   sampling-point definition
    -   RoI parameterization
    -   exact C and C⁻¹ rules

-   A visual sanity check (even screenshots) proving:

    -   cube centers march along the vessel correctly
    -   RoIs line up with the right locations

**If you get stuck today:** Stop and fix label alignment. If your
conversions are wrong, Day 4--6 will collapse.

------------------------------------------------------------------------

# Day 2 --- Make each task learn alone (single-task sanity)

**Goal:** Ensure both sub-tasks can overfit the gold pack independently.

## Morning: Spatial task only (Object Detection)

-   Train object detection head only with **Lod**
-   Confirm Hungarian matching works and "no-object" is handled (the
    paper uses a no-object category ∅ in matching)
-   Confirm your RoI loss uses IoU + L1 with λIoU=2 and λL1=5

**Pass condition:** On the gold pack, the detector should predict the
right number of lesions and roughly correct locations.

## Afternoon: Temporal task only (Sampling-point classification)

-   Train sampling-point classifier only with **Lsc (cross entropy)**

**Pass condition:** On the gold pack, it should learn to tag points near
lesions as abnormal, and healthy regions as healthy.

## Evening: Debug checklist

If either task won't learn:

-   Check normalization (medical images are sensitive)
-   Check label noise (especially lesion boundaries)
-   Check extreme class imbalance (most points are healthy)

## Day 2 Deliverables

-   Two tiny overfit runs (detector-only and point-only) that clearly
    learn.
-   Logged metrics that move strongly in the right direction.

------------------------------------------------------------------------

# Day 3 --- Combine tasks without Ldc (Lod + Lsc only)

**Goal:** Build the "two-head SC-Net" without cross-supervision yet.

## Morning: Build the combined forward pass

-   The paper's spatial branch uses CPR + four 2D views
-   The temporal branch uses 32 cubes with transformer encoder across
    locations

You do not need to perfectly match every architectural detail to test
training stability, but you MUST produce both outputs every step:

-   Detection predictions (Q queries, class + RoI)
-   Sampling-point predictions (32 points, class probs)

## Afternoon: Train with Loverall = Lod + Lsc (no Ldc yet)

-   You are intentionally skipping Ldc today.
-   You want stable training curves first.

**Pass condition:** Both tasks improve together on validation without
one collapsing.

## Evening: Establish your baseline evaluation

The paper evaluates:

-   train split 70%, remaining 30% split into val/test
-   best validation checkpoint after 200 epochs used for test
-   artery-level ACC/Prec/F1/Spec for stenosis degree and plaque
    components

In a 1-week timeline, you can:

-   run fewer epochs for quick iteration
-   run an overnight longer run later (Day 6)

## Day 3 Deliverables

-   A stable baseline run (Lod + Lsc) with sensible metrics.
-   A saved baseline checkpoint + evaluation report.

------------------------------------------------------------------------

# Day 4 --- Add Ldc (dual-task contrast) carefully

**Goal:** Turn on the paper's key idea: mutual correction through
prediction contrast.

The paper defines:

-   **Loverall = Lod + Lsc + Ldc**
-   **Ldc = Lod(C(ŷsc), ŷod) + Lsc(C⁻¹(ŷod), ŷsc)**

## Morning: Implement Ldc with guardrails

Do NOT just switch it on full strength immediately.

Use this schedule (simple + effective):

-   Epochs 1--N: train only Lod + Lsc (Ldc weight = 0)
-   Next phase: linearly ramp Ldc weight from 0 → 1

Why: early predictions are noisy; Ldc will otherwise push noise into
both heads.

## Afternoon: Train short runs + verify "consistency metrics"

Track these sanity signals:

-   \% abnormal sampling-points that fall inside predicted RoIs
-   \% predicted RoIs that overlap abnormal point runs
-   number of physiologically weird cases (e.g., plaques predicted but
    stenosis says healthy everywhere)

The paper claims Ldc reduces missed/misdiagnosis by correcting
generalization errors

## Evening: Decide whether Ldc helps

You'll see one of three outcomes:

1.  Metrics improve + consistency improves (great)
2.  Metrics flat but consistency improves (still good --- keep tuning)
3.  Training becomes unstable (reduce ramp speed, lower Ldc weight cap,
    or delay Ldc longer)

## Day 4 Deliverables

-   A stable run with Ldc enabled and ramped.
-   A clear comparison vs Day 3 baseline.

------------------------------------------------------------------------

# Day 5 --- Build clinically-credible augmentation + pretraining (paper's biggest boost in low data)

**Goal:** Replicate the paper's "CDA + pretrain → finetune" sequence.

Paper method:

-   Create background set **B** (CPR volumes as background)
-   Create lesion foreground set **F** (lesion RoIs)
-   Overlay random f onto random b to form augmented set **A**
-   Pretrain on A for plaque composition, then fine-tune on clinical B
    for plaque + stenosis

## Morning: Build B, F, and A

Make it clinically sensible:

-   Only paste lesions into similar vessel diameters (paper notes
    diameter variation matters)
-   Avoid obvious edge artifacts (basic blending / intensity matching)

## Afternoon: Run the pretraining stage

Keep it aligned with the paper:

-   Pretraining focuses on plaque composition due to diameter variation

You're not trying to perfect numbers here --- you're trying to learn
better lesion appearance features before fine-tuning.

## Evening: Quality check your augmentation

Randomly inspect 50 augmented samples:

-   Are lesions placed in plausible vessel locations?
-   Do you see harsh seams?
-   Are lesion labels still correct after pasting?

## Day 5 Deliverables

-   Augmented dataset A built
-   Pretraining run completed + checkpoint saved
-   Quick sanity report: "augmentation looks clinically plausible"

------------------------------------------------------------------------

# Day 6 --- Full fine-tuning run + ablations (choose the minimum set)

**Goal:** Produce your "closest-to-paper" result and prove which
components matter.

## Morning: Fine-tune on clinical data B (full objective)

-   Start from the pretrained checkpoint (Day 5)
-   Train with **Lod + Lsc + ramped Ldc**
-   Use paper's key hyperparams first (λIoU=2, λL1=5)

## Afternoon: Run the *minimal* ablations that matter most

The paper's ablation focuses on:

-   removing CDA harms performance, especially with less data
-   removing SOD or TSC changes performance (spatio-temporal benefit)
-   removing Ldc reduces reliability / correction

In one week, do just these 3 comparisons:

1.  **Full model** (CDA + Lod + Lsc + Ldc)
2.  **No CDA** (skip pretraining, train only on B)
3.  **No Ldc** (set Ldc=0)

(If you have time: detector-only or point-only as extra.)

## Evening: Overnight "long run"

If you can afford it, this is where you do your longest run (closest to
their training length). The paper uses **200 epochs** and selects best
validation checkpoint You may not reach 200, but run as long as your
compute allows overnight.

## Day 6 Deliverables

-   Best "full system" checkpoint
-   2--3 ablation checkpoints
-   A table of metrics for all three runs

------------------------------------------------------------------------

# Day 7 --- Final evaluation, error analysis, and a tight improvement pass

**Goal:** Turn results into something you can confidently present and
improve quickly.

## Morning: Final evaluation exactly how you'll report it

Match the paper's metric set as closely as you can:

-   ACC, Precision, Recall, F1, Specificity (artery-level if you can)
    Also, report performance with reduced data if relevant (paper
    highlights 50% vs 100%)

## Afternoon: Error analysis (this is what improves your model fastest)

Take your worst 30--50 mistakes and bucket them:

-   Missed lesions (false negatives)
-   False lesions (false positives)
-   Wrong plaque type
-   Wrong stenosis degree
-   Cross-task inconsistency (detector says lesion but point classifier
    says healthy everywhere)

For each bucket, identify the main cause:

-   Label noise near boundaries?
-   Sampling interval too coarse?
-   Too many "no-object" matches?
-   Ldc ramp too aggressive?

## Evening: One improvement pass (pick only ONE)

Best single improvement in 7 days:

### Tune Ldc schedule for stability

Try one of these (choose one):

-   Delay Ldc longer before ramp
-   Ramp slower
-   Cap Ldc weight at 0.5 instead of 1 for the whole run
-   Only use high-confidence pseudo-labels in C and C⁻¹ during early
    epochs

Re-run a short training (not full) to see if your biggest error bucket
improves.

## Day 7 Deliverables

-   Final metric table (Full vs No CDA vs No Ldc)

-   Short written summary:

    -   what helped most (usually CDA + Ldc)
    -   what still fails (your top error bucket)
    -   your next 3 improvements if you had more time

------------------------------------------------------------------------

# What "success" looks like at the end of the week

You've succeeded if you can say:

1.  "I replicated SC-Net's training recipe: CDA augmentation → pretrain
    on A → fine-tune on B"
2.  "I trained dual tasks (object detection + sampling-point
    classification) and used the prediction-contrast Ldc exactly as
    described"
3.  "Ablations show CDA and Ldc matter --- removing them hurts."

------------------------------------------------------------------------

If you tell me *one thing* (no long back-and-forth): **what you already
have labeled** (RoIs only, point labels only, or both), I can adjust
this 7-day plan to fit your reality without wasting a day on the wrong
label pipeline.
