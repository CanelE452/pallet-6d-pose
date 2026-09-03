# Metric definitions and dependency graph

All seven blocked metrics sit downstream of one decision. That is why they are not
seven separate problems.

## Dependency graph

```text
                    2D predictions (bbox + 9 keypoints)
                                  |
                                  v
                  prediction-only W/D hypothesis selection      <-- THE BLOCKER
                                  |
                                  v
                            PnP  ->  R_pred, t_pred
                                  |
        +---------+---------+-----+-----+---------+---------+
        |         |         |           |         |         |
      R med    yaw med    t med      IoU3D      ADD      ADD-S
                                                  |         |
                                                  +----+----+
                                                       |
                                                  ADD / ADD-S AUC
```

Every leaf requires a valid `R_pred`. If the hypothesis is wrong the rotation is off
by 90 degrees, and each leaf then reports a number that is not a measurement of the
model's pose accuracy but of the selector's failure.

Measured, on the frames where the selector chose wrong:

```text
rotation error    median 85.300 deg
yaw error         median 85.274 deg
translation error median  0.219 m   (versus 0.065 m when correct)
```

## Definitions, as implemented

Implementation of record: `challenge/evaluation_v2/pose_metrics.py`, which is what
`paper_real_eval.py` calls.

```text
R med       geodesic angle between R_pred and R_gt
            rotation_error_degrees(), pose_metrics.py:201

yaw med     absolute relative yaw about the pallet's canonical local Y axis
            yaw_error_degrees(), pose_metrics.py:211
            This is the fork-alignment component, not a full rotation.

t med       || t_pred - t_gt ||
            translation_error_m(), pose_metrics.py:223

ADD         mean distance between model points transformed by the predicted pose
            and by the GT pose
            add_error_m(), pose_metrics.py:251

ADD-S       nearest-neighbour (symmetry-aware) variant of ADD
            adds_error_m(), pose_metrics.py:263
            The docstring marks the unrestricted version as NOT the paper symmetry
            policy; the paper variant must use the frozen symmetry contract.

pose AUC    area under the accuracy-threshold curve over [0, 0.1 x diameter]
            pose_auc(), pose_metrics.py:297
            Threshold-free by construction, which is why it replaced 5cm5deg.

IoU3D       oriented 3D box overlap
            NOT IMPLEMENTED in challenge/evaluation_v2/ — see below
```

## Two implementation gaps that must be closed before any pose number is produced

These are not in the five blockers listed for this track. They were found by reading
the code.

### Gap 1 — IoU3D has no implementation in the paper evaluator

```text
challenge/evaluation_v2/pose_metrics.py     no IoU3D
scripts/stage0/real_eval/re_metrics.py:171  iou_3d(...)  exists here
```

`metric_split_lock.md` §2.3 specifies *exact* oriented-box IoU (12 half-spaces +
Chebyshev LP + ConvexHull, explicitly not an approximation). The paper evaluator
does not have it. Either it is imported from `re_metrics` under a frozen hash, or it
is reimplemented and unit-tested. Until then IoU3D stays out of the table even if
every other blocker clears.

### Gap 2 — two different `pose_auc` implementations exist

```text
pose_metrics.py:297   input = normalised errors, 1001 thresholds, area / max_fraction
re_metrics.py:73      input = raw errors + diameter, 100 steps
```

Same intent, different discretisation, so they will not return identical numbers.
One must be declared canonical in the evaluator lock and the other must not be used
for paper values. `metric_split_lock.md` names `re_metrics.py::pose_auc` as the
implementation that passed tests T15-T21, while the paper evaluator calls the other
one. This has to be resolved explicitly, not silently.

## The selector gate, as the code enforces it

`challenge/evaluation_v2/pnp_selector.py`:

```text
OVERALL_AXIS_ACCURACY_MIN   0.95
NIGHT_AXIS_ACCURACY_MIN     0.90
SESSION_AXIS_ACCURACY_MIN   0.85     <- a third gate, per-session minimum
```

`build_pose_metric_gate()` additionally requires:

```text
population_validated is True
sample_count == expected_sample_count
session_count == len(expected_sessions)
tail_dominance_assessed is True and tail_dominance_passed is True
```

and separately:

```text
canonical_migration_status == "PASS"
symmetry_status == "FROZEN"
final_manifest_frozen is True
```

Current state against those:

```text
                       R0        R5      required
overall              0.6500    0.5929      >= 0.95
night                0.6786    0.8214      >= 0.90
minimum session      0.3030    0.2121      >= 0.85
tail dominance         FAIL      FAIL       must pass
```

The per-session minimum is the furthest from its gate, which matches the session
breakdown: two sessions score 1.000 while `eval_night08` scores 0.333.

## Coverage is part of the gate, by design

A selector that returns `POSE_UNRESOLVED` on hard frames and is accurate on the rest
has not solved the problem. This track therefore pre-registers a coverage floor
alongside accuracy, and every pose table must report:

```text
pose coverage              fraction of frames with a resolved pose
conditional pose error     computed on resolved frames only
failure-aware score        computed over all frames
```

Reporting only the conditional numbers is prohibited.

## What is not blocked

For completeness, so that the closure is not overstated:

```text
plastic  geometry_status FROZEN, symmetry_status FROZEN, symmetry_contract present
wood     geometry_status FROZEN, symmetry_status UNREVIEWED, contract null
```

Plastic's symmetry side is already closed. Plastic's only blocker is the selector.
