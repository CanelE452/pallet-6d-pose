# Monocular temporal refinement pilot

Every self-training arm so far chose which teacher pseudo-labels to keep and then
used the teacher's own coordinates as supervision. This refines the coordinates
themselves from neighbouring real RGB frames, then constrains them with the known
pallet cuboid. No student was trained. No depth was read. The refinement code cannot
read ground truth — an AST test asserts it takes no such argument and opens no
annotation.

```text
TEMPORAL_METHOD_PILOT = FAILED_TO_IMPROVE
```

## Population

```text
centres                    109
recordings                 8   capturenight01, capturenight02, capturenight03, capturenight04, capturenight05, capturenight06, capturenight07, forklift_raw_20260528_163408
composition                night and forklift, plastic only
PAPER_EVAL frame overlap   0
```

Every recording that feeds PAPER_EVAL was excluded whole, so a confirmation
population for this method is not spent in advance. The lock counted 88 centres
beforehand; the run found 109 because `forklift_raw_20260528_163408` meets every
frozen criterion and had not been in the pre-count. It is kept and reported
separately rather than dropped after the fact.

An implementation bug was caught before measurement: the first population build
required centres to belong to PAPER_EVAL_POSITIVE while also excluding every
recording that feeds it, which is empty by construction. The lock never asked for
that condition; removing it restored the basis the lock's own pre-count used.

## Teacher parity

```text
checkpoint sha    970a0913b38ed4c9...
box max delta     0.00e+00 px
kp max delta      0.00e+00 px
conf max delta    0.00e+00
verdict           PASS
```

## Temporal flow

```text
tracklet                        7 frames
observations per corner median  7.0
corners reaching the minimum    0.938
forward-backward rejection      0.0809
geometry coverage               0.945
```

The correspondence worked. Seven observations per corner, an eight percent
forward-backward rejection rate — flow coverage is not why this failed.

## Results

```text
method                    cov   med px   p90 px   gross20   IoU3D   ADDsym    t cm
──────────────────────────────────────────────────────────────────────────────────
RAW_TEACHER             1.000     8.49    19.06     0.261   0.467    0.198   17.96
TEMPORAL_ONLY           0.945     8.63    16.17     0.247   0.389    0.167   18.13
TEMPORAL_GEOMETRY       0.945     9.14    17.72     0.269   0.389    0.167   18.13
```

## Paired differences

```text
contrast                                   diff          frame 95% CI          cluster 95% CI
─────────────────────────────────────────────────────────────────────────────────────────────
TEMPORAL_ONLY - RAW_TEACHER median_px    +0.810      [-2.900, +1.656]        [-2.324, +1.371]
TEMPORAL_ONLY - RAW_TEACHER p90_px       -1.538      [-6.647, +1.211]        [-8.171, +2.242]
TEMPORAL_ONLY - RAW_TEACHER gross20      -0.006      [-0.036, +0.024]        [-0.040, +0.029]
TEMPORAL_ONLY - RAW_TEACHER iou3d        -0.078      [-0.132, -0.018]        [-0.125, +0.002]
TEMPORAL_ONLY - RAW_TEACHER add_sym_auc   -0.027      [-0.050, -0.005]        [-0.059, -0.006]  excludes 0

TEMPORAL_GEOMETRY - RAW_TEACHER median_px   +1.317      [-1.848, +2.528]        [-1.806, +2.314]
TEMPORAL_GEOMETRY - RAW_TEACHER p90_px   +0.013      [-5.208, +2.865]        [-5.747, +6.102]
TEMPORAL_GEOMETRY - RAW_TEACHER gross20   +0.016      [-0.016, +0.049]        [-0.010, +0.042]
TEMPORAL_GEOMETRY - RAW_TEACHER iou3d    -0.078      [-0.132, -0.019]        [-0.125, +0.001]
TEMPORAL_GEOMETRY - RAW_TEACHER add_sym_auc   -0.027      [-0.050, -0.006]        [-0.058, -0.006]  excludes 0

TEMPORAL_GEOMETRY - TEMPORAL_ONLY median_px   +0.507      [-0.725, +2.378]        [-0.815, +2.141]
TEMPORAL_GEOMETRY - TEMPORAL_ONLY p90_px   +1.551      [-1.010, +4.938]       [-2.989, +10.343]
TEMPORAL_GEOMETRY - TEMPORAL_ONLY gross20   +0.022      [+0.003, +0.041]        [-0.002, +0.055]
TEMPORAL_GEOMETRY - TEMPORAL_ONLY iou3d   +0.000      [+0.000, +0.000]        [-0.000, +0.000]
TEMPORAL_GEOMETRY - TEMPORAL_ONLY add_sym_auc   +0.000      [+0.000, +0.000]        [+0.000, +0.000]

```
clusters 8

## Why it failed

The correspondence was not the problem, and neither was coverage. The consensus
moves each coordinate a median of 1.81 px while the raw teacher sits 7.90 px from
ground truth — a ratio of 0.23 — and the direction of that movement is a coin flip:
exactly half the corners end up closer to the truth.

That is the signature of neighbouring teacher predictions already sharing the same
error. A median across seven observations can only help if those observations
disagree. Here the teacher makes the same mistake on frames a tenth of a second
apart, so there is nothing for the median to average away, and it nudges the point
in a direction uncorrelated with the truth.

One structural note. `TEMPORAL_GEOMETRY` and `TEMPORAL_ONLY` return identical 6D
numbers, to the last digit. That is not a bug: the geometry step is a PnP fit to the
consensus followed by reprojection, so re-fitting PnP during evaluation recovers the
same pose. The geometry step can change the 2D coordinates and can never change the
6D pose derived from them. It changed the 2D for the worse.

## Verdict

```text
TEMPORAL_ONLY_2D              MIXED
TEMPORAL_GEOMETRY_2D          NO_IMPROVEMENT
DOWNSTREAM_6D                 NO_IMPROVEMENT
COVERAGE                      ACCEPTABLE

TEMPORAL_METHOD_PILOT = FAILED_TO_IMPROVE
```

The one thing that moved in the right direction was the p90 of the temporal-only
coordinates, 19.06 px down to 16.17, and its cluster interval spans zero comfortably.
Everything else is flat or worse, and the symmetry-aware ADD is resolvably worse with
a cluster interval that excludes zero in both contrasts against the raw teacher.

Per the lock this ends here. The window is not widened, the tracklet is not
lengthened, the forward-backward gate is not relaxed, the median is not replaced, and
no learned flow network is substituted. No further method search is opened.

## Scope

Night and forklift recordings, plastic only, 109 centres across 8 clusters. Strict
exclusion of every PAPER_EVAL-feeding recording leaves no daytime centre at all, so
this measures one slice and says nothing directly about daytime behaviour. That
limitation was written into the lock before the measurement, not discovered here.

`NEXT_ACTION = USER_REVIEW_TEMPORAL_PILOT`

