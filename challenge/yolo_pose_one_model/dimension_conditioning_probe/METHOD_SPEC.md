# METHOD_SPEC — dimension conditioning probe

Pre-registered before any Phase A number was computed. Gates live in
`CONTRACT.json` and are not editable after results exist.

The Phase A interpretation clarifications below were added after the derived
artifact audit. They correct keypoint provenance and distinguish a pose oracle
from a static mapping check; no frozen gate threshold or prediction changed.

## Question

Does known object geometry help real pose, and at which insertion point?

```
Q1  Is the current W/D parity selector failure a dominant real pose bottleneck?
Q2  Are dimensions alone enough, or is an image+dimensions parity classifier needed?
Q3  Does a dimension-conditioned parity classifier recover the oracle headroom?
Q4  Does any argument survive for pushing dimensions into the keypoint head?
```

## Phase A — training-0 oracle headroom

One frozen checkpoint, one prediction cache. A0/A1/A2/A4 consume the identical
box / confidence / nine predicted 2D keypoints. A3 is the explicit GT-v2
keypoint replay exception.

```
arm  parity source            keypoints      dimensions    deployable
────────────────────────────────────────────────────────────────────
A0   frozen prediction-only   predicted      correct       yes
A1   GT                       predicted      correct       NO  (ORACLE)
A2   argmin GT pose error     predicted      correct       NO  (ORACLE)
A3   GT                       GT-v2 annotation replay  correct       NO  (consistency replay)
A4   frozen prediction-only   predicted      WRONG object  NO  (control)
```

A1 is `GT_USED_FOR_PARITY`, `PAPER_ELIGIBLE = FALSE`. Predicted-keypoint A2 is
`GT_USED_FOR_SELECTION`: it chooses the lower restricted ADD-S solution and is
a pose-oracle diagnostic. It need not agree with A1 when predicted keypoints
are noisy or their correspondences are permuted.

The static GT parity / canonical mapping / solver check solves both parities
from `GT-v2 keypoint_annotations.xy`. That separate check must agree with GT
parity on all 140 frames. A failure there, rather than a predicted-keypoint A2
disagreement, indicts the static contract.

A3 also uses `GT-v2 keypoint_annotations.xy`. Those annotations and the
canonical pose candidates share legacy annotation/PnP provenance. A3 is
therefore a solver consistency replay, not an independent estimate of
annotation, intrinsics, or PnP noise.

The executed `A4` direction deliberately feeds wood dimensions to plastic
frames. The reverse wood-images/plastic-dimensions direction is blocked and was
not evaluated because wood pose prerequisites are not approved. A4 exists only
to prove the metric is sensitive to geometry input; it is never a deployment
arm and never enters a gate.

## Metrics

Restricted ADD-S is the minimum corresponding-point ADD over the explicit
proper rotations of the frozen symmetry contract, normalized by the model
diameter. AUC is the unconditional area over `[0, 0.1]` diameter, so a frame
without a valid pose contributes zero rather than being dropped.

Reported: parity accuracy, PnP solve rate, pose-valid N over total, restricted
ADD-S AUC, symmetry-aware rotation median, translation median, symmetry-aware
yaw median. Subgroups: ALL / DAY / NIGHT / session / PLASTIC / WOOD.
95% CI by session-cluster bootstrap, 1,000 resamples, seed 20260828.

`5cm5deg` is not computed and not reported.

The implemented Phase-A G3 scope is the primary Plastic DEV140 `ALL` summary.
This follows the contrast with G1, which explicitly names overall, NIGHT, and
minimum-session checks. The user protocol did not explicitly resolve whether
G3 instead quantifies over every reported subgroup; the final report therefore
also gives the gate-changing strict-subgroup sensitivity result.

## Populations

```
PLASTIC  DEV_POS140            140  selector diagnostic     DAY 112 / NIGHT 28
PLASTIC  COMMON_DEV_PLASTIC_POS128  128  model comparison
WOOD     DEV_WOOD_POS45         45  CROSS_SHAPE_DEV         2 sessions
```

Wood symmetry is `UNREVIEWED` and wood intrinsics are
`SENSOR_PROFILE_SCALED`. All wood Phase A evaluation is therefore
`BLOCKED_NOT_EVALUATED`; the cached wood predictions are retained only as
immutable evidence for a later approved audit.

## Phases B–D

Phase B (synthetic dimension/parity supervision audit), Phase C (frozen-feature
probe arms B0–B5) and Phase D (deployment wrapper) execute only on a Phase A
`ORACLE_PARITY_HEADROOM_PRESENT` verdict. A Phase A failure terminates the
track with `DIMENSION_PARITY_NOT_MAIN_LEVER` and no parity head is trained.

Phase C follows the same one-pose inference contract as Phase A: it caches the
highest-confidence final YOLO detection per frame. Lower-ranked detections are
not additional parity-training examples. This avoids silently changing the
deployment unit from one image/one pose to a detector-candidate dataset.

The Phase-C model-comparison and P1–P8 gate population is the frozen
`COMMON_DEV_PLASTIC_POS128` manifest. `DEV_PLASTIC_POS140` remains available in
the result JSON as a selector-diagnostic view; it is not substituted for the
128-frame model-comparison population. The 28 NIGHT frames are shared by both.

The image feature is tapped from the immediate producer of the active one2one
classification readout after discovering the pose head by capability. No
executable module path is hard-coded. Instrumented and uninstrumented final
candidate count, confidence, box, and keypoints must match exactly; the mapped
source logit's sigmoid must reproduce final confidence within numerical
precision.

Each of B0–B3 runs Linear and `[128, 32]` SiLU MLP probes for seeds 0/1/2 with
the same 40 epochs, balanced sampler/loss, optimizer, and source splits. The
architecture/seed lock is selected by synthetic DEV balanced accuracy and is
written before synthetic TEST or real labels are opened. B4 and B5 reuse the
selected B3 checkpoint without retraining.

The Phase-B asset split is held out for the tiny probe head only. The frozen
YOLO feature extractor was previously trained on the original G38 membership,
which spans all four source assets. Synthetic TEST is therefore not an
end-to-end asset-heldout claim, and topology separation is also uncertified.
Wood is outside the observed G38 dimension neighborhood and remains an
exploratory parity diagnostic, not multishape evidence.

Phase D is forbidden unless a Phase-C candidate clears the source-performance
gates and then the latency gate. A failed gate intentionally produces no
deployment wrapper or integration latency artifact.
