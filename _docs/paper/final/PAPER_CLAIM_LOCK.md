# Paper claim lock

Every claim below was checked against the frozen result artifacts, not against an
earlier document. Where a previously drafted claim did not survive that check, the
correction is stated explicitly rather than quietly dropped.

Machine-readable twin: `PAPER_CLAIM_LOCK.json`.

## Core thesis

> Synthetic supervision alone yields a strong pallet keypoint estimator without any
> real pose label. Self-training on unlabeled target RGB substantially improves
> real-domain detection coverage under difficult acquisition conditions and improves
> confidence-based separation of positives from negatives. Under a controlled
> exposure-matched protocol it does **not** improve fine keypoint localisation over
> the synthetic-only model, and adding projective and equivariant consistency to the
> pseudo-label selection rule does not recover a localisation gain. Post-hoc
> diagnostics show the limitation is not explained by any single filtering
> implementation: pseudo-label purity can be improved without moving the student,
> and pooling additional observations from the same teacher improves the median
> while worsening the tail.

## Claims the evidence supports

### A — Synthetic-only supervision is already a strong real-domain baseline

```text
R0 detection ALL      0.975   (319 frames)
R0 detection daytime  1.000   (70)
R0 AUROC              0.9921  (319 positive vs 2,689 negative)
R0 box AP50           0.9363
R0 box AP50-95        0.7688
```

No real image and no real pose label entered R0's training.

### B — Self-training improves nighttime detection coverage

```text
nighttime detection (N = 50, plastic only)
  R0                0.840
  R1 naive          0.960
  R2 confidence     0.980
  R3 + reprojection 0.960
  R4 + removal      0.960
  R5 full filter    0.960
```

**Correction to an earlier draft claim.** The pair 0.840 → 0.960 is real and is in
the frozen table, but it is **not** attributable to the geometry filter. Naive
self-training reaches 0.960 on its own, and confidence-only selection reaches 0.980,
higher than the full filter. The supportable sentence is *"self-training improves
nighttime detection coverage"*, never *"our geometric filter improves nighttime
detection."*

**Second correction.** The overall detection gain is not separated from noise:

```text
R0 vs R5 detection, paired bootstrap
  frame-level        p_better = 0.121
  session-clustered  p_better = 0.244
```

So the detection improvement must be reported with its uncertainty, and the
nighttime subgroup (N = 50) must carry its sample size in the text.

### C — Self-training improves confidence-based ranking, and here the full filter is the best arm

```text
                     AUROC     FPR95
  R0                 0.9921    0.0417
  R0-CONT            0.9872    0.0573
  R1 naive           0.9913    0.0558
  R2 confidence      0.9923    0.0469
  R3 + reprojection  0.9920    0.0487
  R4 + removal       0.9911    0.0502
  R5 full filter     0.9953    0.0283
```

Nighttime is where it is largest:

```text
  nighttime AUROC    R0 0.9689  ->  R5 0.9920
  nighttime FPR95    R0 0.1949  ->  R5 0.0588
```

This is the one axis on which the full consistency filter is the best of all seven
arms. If the paper makes any positive claim about the proposed filter, it should be
this one — a 32 percent relative reduction in FPR95 — and not a detection claim.

Metric names must be copied exactly: the evaluator emits `box_ap50`, `box_ap50_95`,
`auroc`, `fpr95`. There is **no** metric named "ranking AP" anywhere in the
artifacts; do not invent one.

### D — Fine keypoint localisation does not improve

```text
supervised keypoints, original-image pixel error, ALL population
                     n_kp   median px   p90 px    gross20
  R0                 2756     6.616     38.670     0.172
  R0-CONT            2729     6.911     45.187     0.182
  R1 naive           2770     7.120     35.119     0.180
  R2 confidence      2784     7.037     43.606     0.194
  R3 + reprojection  2788     7.044     41.294     0.194
  R4 + removal       2788     6.999     39.335     0.194
  R5 full filter     2779     7.210     41.380     0.197
```

No arm beats R0's median. The direction is a small degradation, and it is supported
at the frame level and marginal when clustered by session:

```text
R0 vs R5 corner error, paired bootstrap, probability that R5 is better
  frame-level        0.028
  session-clustered  0.065
```

The honest statement is that self-training **did not improve** localisation, with a
small degradation that is frame-level significant and not session-level significant.
Do not upgrade this into "self-training harms localisation" without the session-level
caveat.

### E — Geometry-based filtering shows no clear additional downstream localisation gain

Confidence-only (R2) and the full consistency filter (R5) are within noise of each
other on localisation, and both are worse than R0. Reprojection and keypoint-removal
consistency (R3, R4) do not separate either.

### F — The selection signals are not random

Post-hoc separability analysis finds real, if modest, discrimination at both frame
and corner level. This is **diagnostic evidence** and must be labelled as such
wherever it appears.

### G — Improved pseudo-label quality did not move the student

The reliability-weighting track improved the expected quality of the labels the
student saw, and student localisation did not follow. This is the most direct
evidence for the teacher-ceiling reading. **Development evidence, Tier B.**

### H — Near-square projections carry a 90-degree semantic-axis ambiguity

```text
axis-permutation rate (frames, criterion in AXIS_FAILURES.json)
  R0        0.047
  R2_CONF   0.053
  R5        0.050
```

Diagnosed for R0, R2, and R5 only. Diagnostic evidence; the frames are judged by
maximum corner error, because the error distribution is bimodal and the median hides
the failure.

## Claims that are forbidden

These sentences must not appear anywhere in paper-facing text:

```text
our method improves keypoint localisation
our geometry filter improves pose accuracy
our method improves 6D pose accuracy
our method reduces yaw error
geometry filtering provides a clear downstream gain
our geometric filter improves nighttime detection
label-free 6D pose improvement
held-out improvement
unseen-pallet generalisation
arbitrary pallet generalisation
no real annotation was used anywhere
state-of-the-art
```

### Why each is blocked

```text
localisation / pose gain     contradicted by D; no arm beats R0
filter improves night det.   contradicted by B; naive reaches the same value
6D pose of any kind          POSE_METRICS_STATUS = BLOCKED
held-out                     PAPER_EVAL role = DEV, held_out_final = false
unseen / arbitrary pallet    no experiment on an unstudied pallet category
label-free                   manual real annotations exist and are used to evaluate
```

The permitted phrasing for the training condition is:

> **without manual target-domain pose labels during training**

## Pose metrics

```text
POSE_METRICS_STATUS = BLOCKED
blocked_reason      = POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;
                      FINAL_MANIFEST_NOT_FROZEN;
                      wood: CANONICAL_MIGRATION_NOT_PASS;
                      SYMMETRY_NOT_FROZEN
```

Consequently **no performance sentence** may be written about yaw, rotation,
translation, ADD, ADD-S, 3D IoU, 5cm5deg, or 6D pose AUC.

Describing the pipeline is permitted:

```text
allowed    "keypoints are converted to a 6D pose using PnP"
forbidden  "our method improves 6D pose"
```

## Population and evidence-tier rules

```text
PAPER_EVAL 319   population_contract.role = DEV
                 held_out_final = false
```

PAPER_EVAL was consumed as a development population by every diagnostic track. No
number computed on it may be called held-out or independently confirmed.

V3-B is **not** retroactively the proposed method. It was designed after seeing
PAPER_EVAL diagnostics and is classified as a post-hoc ambiguity-aware development
variant. The V1 frozen full filter and the V3-B development variant are never
presented at the same evidence level.

## Subgroup hazard

Two different nighttime subgroups exist in the artifacts and must never be mixed:

```text
subgroups.Nighttime        N = 50    plastic only     used by TABLE_M2 / M5
subgroups.Lighting_night   N = 106   plastic + wood   broad lighting split
```

Every reported subgroup number carries its N.

## Terminology

```text
use                                              never
──────────────────────────────────────────────────────────────────────────────
single-keypoint-removal reprojection consistency  LOO
keypoint-removal reprojection consistency         leave-one-out
horizontal-flip keypoint consistency              flip reprojection
projective and equivariant consistency            geometric filter (as an umbrella)
box AP50 / box AP50-95 / AUROC / FPR95            ranking AP
original-image pixel error                        corner error (unqualified)
```
