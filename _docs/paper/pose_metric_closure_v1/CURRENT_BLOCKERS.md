# Current blockers, reproduced from code and artifacts

Reproduced at HEAD `1664131`. Nothing here is quoted from an earlier document; every
number was recomputed from the stored artifacts. No model was run.

```text
POSE_METRICS_STATUS = BLOCKED
```

Source of the declaration: `data/pallet/results/paper_eval_v1/EVALUATOR_CONTRACT.json`,
key `POSE_METRICS_STATUS`, with per-population reasons under `populations.*.pose_blocked_reasons`.

## The four separated blockers, as the contract states them

```text
plastic_axis_selector    BLOCKED — prediction-only W/D hypothesis selector FAIL
                                   (83/140, NIGHT 13/28)
wood_symmetry            BLOCKED — registry symmetry_status UNREVIEWED; selector NOT_RUN
wood_canonical_migration BLOCKED — CANONICAL_MIGRATION_NOT_PASS
new_146_signed_axis      BLOCKED — pose_status UNCONFIRMED_SIGNED_AXIS 146/146,
                                   axis_assignment_confirmed False
```

A fifth, from the population contract rather than the evaluator:

```text
final population         PAPER_EVAL role = DEV, held_out_final = false,
                         already consumed by V1-V5 and every diagnostic track
```

## Blocker 1 — the plastic W/D selector

### What was reproduced

Population `DEV_POS140` (manifest `challenge/real_gt_v2/manifests/DEV_POS140.json`,
membership sha256 `b0be8173…`), 140 frames across the seven canonical eval sessions.
Two of the 140 have no detection, leaving 138 scored.

```text
run                       status   overall   night   min session   n     n_night
────────────────────────────────────────────────────────────────────────────────
frozen selector diagnostic  FAIL    0.5929   0.4643      0.3333    140      28
R0 predictions              FAIL    0.6500   0.6786      0.3030    140      28
R5_PROPOSED predictions     FAIL    0.5929   0.8214      0.2121    140      28

gate                                >=0.95   >=0.90
```

Sessions split sharply rather than degrading uniformly:

```text
eval_cad        18/18   1.000
eval_noapril    12/12   1.000
eval_outside    14/22   0.636
eval_night09     9/16   0.563
eval_pallet07   12/27   0.444
eval_pallet09   14/33   0.424
eval_night08     4/12   0.333
```

### Why it fails — four measurements

**a. Geometry never eliminates a hypothesis.**

```text
both hypotheses geometrically valid   138 / 138   (100.0%)
exactly one valid                       0 / 138   (0.0%)
accuracy when both valid                          0.601
```

There is not a single frame in which the projective fit rules one hypothesis out.
Whatever the selector is doing, it is choosing between two solutions that both
reproject acceptably.

**b. The selector's own margin does not know when it is wrong.**

```text
|score_margin| AUC for predicting correctness   0.5562     (0.5 = no information)

               n     |margin| median
correct       83          3.739
incorrect     55          3.539
```

The score separates right from wrong barely above chance, and wrong decisions are
made with the same confidence as right ones.

**c. No fail-closed operating point exists.**

Accepting only frames above a margin threshold:

```text
threshold   coverage   accuracy
     0.0      1.000      0.601
     1.0      0.790      0.624
     2.0      0.652      0.633
     3.0      0.551      0.605
     5.0      0.384      0.604
     8.0      0.268      0.676
    12.0      0.145      0.750
```

Accuracy never approaches 0.95 at any usable coverage. Trading coverage away does
not buy correctness, which is the signature of a score with no signal rather than a
badly calibrated one.

**c-2. But the signal is not absent everywhere — one session shows real discrimination.**

Session-level breakdown with the ground-truth class mix:

```text
session          n    acc     expected mix          selected mix
──────────────────────────────────────────────────────────────────────
eval_cad        18  1.000   long 3 / short 15     long 3 / short 15   <- genuine
eval_noapril    12  1.000   long 12               long 12             <- constant guess
eval_outside    22  0.636   long 9 / short 13     long 11 / short 11
eval_night09    16  0.562   long 9 / short 7      long 4 / short 12
eval_pallet09   31  0.452   long 6 / short 25     long 21 / short 10
eval_pallet07   27  0.444   long 9 / short 18     long 16 / short 11
eval_night08    12  0.333   long 11 / short 1     long 3 / short 9
```

`eval_noapril` is 12/12 only because every frame there has the same answer — a
constant predictor scores the same. It is not evidence of anything.

`eval_cad` is different. Both classes are present, and the selector recovers the
exact mix, 18 out of 18. A constant "always short" predictor caps at 15/18 there.
So the geometric score **does** carry real information in at least one session.

This corrects an overstatement that would otherwise be natural from measurement (a):
geometry never *eliminates* a hypothesis, but the scoring function is not uniformly
uninformative. It discriminates in some conditions and collapses in others, and the
pooled 0.6014 is an average across that split.

**d. Pooled, it barely beats a constant guess.**

```text
always "short-face-front"    0.5725
current selector             0.6014
                             ------
gain                        +0.0289
```

Two point nine percentage points over ignoring the image entirely.

### What it costs when wrong

```text
                        correct    incorrect
rotation error [deg]      2.490       85.300
yaw error [deg]           1.956       85.274
translation error [m]     0.065        0.219
```

A wrong choice is not a small perturbation. It is the full 90-degree axis swap, and
it moves translation by a factor of three as the solver compensates.

### The structural reason

```text
plastic_standard_110x130x11    footprint 1.10 m x 1.30 m    ratio 1.182
```

The two hypotheses differ by exchanging 1.10 and 1.30. An eighteen percent
difference in footprint aspect is largely absorbed by the unknown yaw and depth, so
both assignments produce a consistent reprojection. This is a property of the object
and the viewing geometry, not of the scoring function — which is why measurement (a)
returns 138 out of 138.

## Blocker 2 and 3 — wood

```text
registry  wood_small_80x59x14   footprint 0.80 m x 0.59 m   ratio 1.356
          geometry_status       FROZEN
          symmetry_status       UNREVIEWED       <- blocks ADD-S
          symmetry_contract     null
```

For contrast, plastic already has `symmetry_status = FROZEN` with a contract file.
Wood has neither.

The delegated audit is complete (`WOOD_SYMMETRY_REVIEW.md`,
`WOOD_CANONICAL_CLOSURE.md`) and its result is not what the blocker names suggest:

```text
geometry            fine. canonical <-> camera-facing roundtrip max 2.3e-13 px
                    over 125 frames; residual against stored clicks is annotation
                    noise (1.94 px median on DEV45), not a convention error
symmetry set        {I, Ry(180)} is supportable on measurement — rectangular
                    0.80 x 0.59 footprint excludes 90 and 270; deck mirror NCC
                    0.957, slat residual <= 6 mm
what blocks it      freezing wood requires editing OBJECT_GEOMETRY_REGISTRY.json,
                    whose sha256 is pinned in 62 files across 198 places
```

So wood is blocked by a **policy decision about re-issuing a pinned registry**, not
by geometry. And even if that were resolved, the wood selector has never been run and
faces the same gate plastic already failed.

## Blocker 4 — unconfirmed signed axis on the newer annotations

```text
new_146_signed_axis    pose_status UNCONFIRMED_SIGNED_AXIS   146 / 146
                       axis_assignment_confirmed             False
```

These frames cannot serve as pose ground truth until the axis assignment is
confirmed. This is evaluation-GT work, not selector work, and the confirmed labels
must never become selector inputs or training targets.

## Blocker 5 — no untouched final population

```text
PAPER_EVAL   role = DEV   held_out_final = false
             consumed by V1, V2, V3, V4, V5, FILTER_SEPARABILITY,
             STRONG_TEACHER, FAST_TEACHER
```

`DEV_POS140` — the selector diagnostic population — is a subset of the same seven
canonical eval sessions that PAPER_EVAL draws from. Tuning a selector against it
until it reaches 0.95 and then reporting pose numbers on PAPER_EVAL would be
selection on the evaluation set. A separate inventory audit is running to establish
whether a genuinely untouched population exists.

## Leakage status of the existing selector

The frozen diagnostic carries its own contract, and it is the right one:

```text
selector_inputs   predicted 9 keypoints, camera intrinsics,
                  fixed physical dimensions, frozen selector config
forbidden         GT dimensions_m, GT pose, GT axis assignment,
                  GT keypoint error, session prior
comparison_phase  GT parity read only after all selection decisions complete
```

So the current failure is **not** a leakage artefact being removed. It is what
honest prediction-only selection actually scores.

Worth recording alongside this: the earlier `5cm5deg` figure of 30.4 percent fell to
19.3 percent when the evaluator stopped receiving the per-frame GT axis assignment.
That gap is the size of the problem the selector must now solve on its own.

## What this implies for the closure plan

```text
priority A   deterministic selector from existing outputs
             contradicted by measurement (a): geometry never eliminates a
             hypothesis in any of 138 frames, and no threshold reaches the gate

priority B   synthetic-trained semantic-axis head using RGB appearance
             not yet tested. It is the only remaining single-frame route, because
             appearance is the one deployment-available signal the geometric
             selector does not consume

priority C   temporal / multi-frame
             deferred; a moving vehicle sees the same pallet from angles that break
             the ambiguity, but this is future work
```

No conclusion about B is drawn here. Its cost and data contract are reported before
anything is trained.
