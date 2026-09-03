# Final status — first pass

```text
POSE_METRIC_CLOSURE_V1_STATUS = BLOCKED_SELECTOR
```

`BLOCKED_DATA_CONTRACT` is **also independently true**. The spec reserves that label
for the case where the selector works but the data does not, so the primary label is
`BLOCKED_SELECTOR`. Recording both matters, because fixing only the selector would
not open the pose layer.

## Why BLOCKED_SELECTOR

```text
                  overall   night   min session   coverage
R0                 0.6500  0.6786      0.3030       1.000
R5_PROPOSED        0.5929  0.8214      0.2121       1.000
frozen diagnostic  0.5929  0.4643      0.3333       1.000
required           >=0.95  >=0.90      >=0.85       >=0.95
```

Not one arm clears any of the first three. Full reproduction and the four-measurement
decomposition are in `CURRENT_BLOCKERS.md`; the short version is that geometry never
eliminates a hypothesis (138/138 both valid), the score's own margin predicts
correctness at AUC 0.556, and no fail-closed threshold converts coverage into
accuracy.

## Why BLOCKED_DATA_CONTRACT is also true

```text
real frames with axis_assignment_confirmed == True     0    of 2,004 pose-bearing
SELECTOR_DEV with selector labels                      0    (121 frames exist, unlabelled)
POSE_FINAL inventory                                   0    all five manifests empty
wood symmetry_status                          UNREVIEWED
wood canonical migration                         BLOCKED
```

## What was established, and what it cost

```text
GPU training runs        0
inference runs           0
evaluator reruns         0
existing artifacts modified   0
```

Everything came from reading stored artifacts and code.

### Findings that were not in the brief

Four things surfaced from reading the implementation rather than the documents:

```text
1  a fourth selector gate exists — per-session minimum 0.85
   (pnp_selector.py:26-28). It is the gate the arms are furthest from.

2  IoU3D has no implementation in challenge/evaluation_v2/. It exists only in
   scripts/stage0/real_eval/re_metrics.py:171.

3  two pose_auc implementations exist with different discretisation
   (1001 steps vs 100). One must be declared canonical.

4  the current selector genuinely does not consume GT. Verified at the function
   signature, not from the contract's own claim. So 0.5929 / 0.6500 are honest
   numbers, not values that would fall further once a leak is removed.
```

### A correction made during the work

The first reading of "both hypotheses valid in 138/138" suggested the geometric
signal was structurally absent. That was an overstatement. `eval_cad` scores 18/18
with both classes present (long 3, short 15), where a constant predictor caps at
15/18. The signal exists and is condition-dependent; it collapses exactly where it
is needed. `eval_noapril`'s 12/12 is not evidence of anything — every frame there
has the same answer.

### Wood is blocked by a decision, not by geometry

```text
geometry     canonical <-> camera-facing roundtrip max 2.3e-13 px over 125 frames;
             residual against stored clicks is annotation noise (1.94 px median)
symmetry     {I, Ry(180)} is supportable on measurement — rectangular 0.80 x 0.59
             excludes 90 and 270; deck mirror NCC 0.957; slat residual <= 6 mm
blocker      freezing wood requires editing OBJECT_GEOMETRY_REGISTRY.json, whose
             sha256 is pinned in 62 files across 198 places
```

## Two decisions that need a human

### Decision 1 — wood

```text
A  re-issue the registry with wood frozen and update all 198 pins in one pass
B  exclude wood from the pose table; report wood for detection and 2D only
```

B costs nothing and matches what the current paper already does. A is only worth it
if wood pose is wanted, and that depends on a wood selector that has never been run
and that faces the gate plastic already failed. Wood's aspect ratio is 1.356 against
plastic's 1.182, so it should be more separable — a prediction, not a result.

### Decision 2 — the synthetic semantic-axis head

The only remaining single-frame route. Before approving it, three facts:

```text
already done   this selector family has been trained on the 60k synthetic source.
               Synthetic dev balanced accuracy 0.904, AUROC 0.966.
               The same family scores 0.59-0.65 on real. The gap is sim-to-real,
               not data volume.
precedent      this repository has a learned polarity predictor that went from
               0.95 synthetic to 0.023 real while its oracle stayed healthy.
               Same shape of task.
gate location  the gate demands night >= 0.90, and night is where the selector is
               weakest (eval_night08 at 0.333).
```

An honest reading is that priority B has already been attempted in substance and did
not transfer. What has not been tried is a head that consumes **appearance** rather
than dimensions — the ablation showing accuracy dropping to 0.745 under dims-shuffle
says the existing head learned dimensions, which is exactly the signal that does not
survive the 18 percent aspect ratio in real imagery.

That is a real distinction, and it is also a thin thread. No training starts without
approval.

## What would actually be needed

```text
selector       an appearance-based cue, or multi-frame observation. Single-frame
               geometry is exhausted.
GT             an axis-confirmation procedure. Without one, new annotations save as
               UNCONFIRMED_SIGNED_AXIS — as the most recent 146 and 402 frames did.
population     new capture for daytime plastic and wood. Session-independent stock
               today is nighttime plastic only (1,581 frames).
evaluator      an IoU3D implementation and one canonical pose_auc.
```

## What the paper does in the meantime

Nothing changes. `_docs/paper/final/` already states `POSE_METRICS_STATUS = BLOCKED`,
removes the pose columns rather than dashing them, and explains why in
`LIMITATIONS.md` §3. This track supplies the detailed reason, which strengthens that
limitation rather than altering any claim.

No placeholder pose number was written anywhere.

## If this stays blocked, it is still a result

A monocular pose estimator cannot resolve the footprint axis of a near-square
industrial pallet from a single frame, and the failure is a clean 90-degree swap
costing 85 degrees of rotation. That is a specific, measured, transferable finding
about deploying keypoint-based 6D pose on this class of object — and it is more
useful than a pose table produced by an evaluator that quietly resolves the
ambiguity on the model's behalf.
