# The V2 runner was reproducible and still wrong

`7009c10` stands unedited.  What it claimed is true:

```
RUNNER_REPRODUCIBLE   YES     main / train / evaluate live in Git
O0_REPRODUCED         YES     0.0228 deg / 0.0674 cell from the committed path
REAL_FEATURE_OPENED   NO
```

What it did not say is that four things in that runner were wrong, and every one
of them would have decided the screen:

```
COVERAGE_FULL_SPLIT          NO    256 of 16,011 frames, reported as the split
O1B_SEGMENT_MASK_VALID       NO    the mask was true by construction
TRAIN_ROLE_MASK_VALID        NO    the loss averaged over roles the metric dropped
CHECKPOINTING_IMPLEMENTED    NO    documented, never written
```

No feature had been opened, so all four are geometry-and-wiring fixes made
before any result existed to tune them against.

## The four

**The coverage audit sampled the split it was auditing.**  `dev_ids[:256]` and
`train_ids[:256]` against 2,393 and 13,618.  The gate that authorises the whole
screen was reading 3% of the training split.  It now runs every frame and every
role, geometry only -- `load_geometry` reads `camera_data.width/height` from the
JSON, so 16,011 PNGs are not decoded to learn a number the file already states.

**The population included edge that is not in the image.**  Coverage scored the
full `p0 -> p1` segment.  A cuboid edge that projects entirely outside the frame
has no local image evidence by construction, so demanding that a *local* refiner
recover it is not a measurement of features.  Segments are now Liang-Barsky
clipped to the canonical rectangle and classified:

```
IN_FRAME_FULL      the whole edge is visible
IN_FRAME_PARTIAL   coverage is computed on the clipped part only
OFF_FRAME_FULL     excluded from loss and metric, counted and reported
```

Excluding those roles is not hiding a failure.  Whether a *global* predictor can
place an edge it cannot see is a real question, and it is the coarse/structure
question -- this gate does not answer it and no longer pretends to.

The rectangle used is `[0, 49]`, the pixel centres `grid_sample` can actually
read with `align_corners=True`, not a nominal `[0, 50)`.  That is the
conservative direction: it classifies marginally more edge as off-frame, never
less.

**O1B was O1A.**  The old mask built sample positions as
`span = low + (high - low) * alpha` and then asked whether `low <= span <= high`,
which is true for every `alpha in [0, 1]`.  The oracle arm and the control arm
received identical inputs, so the taxonomy branch that separates "orientation
evidence exists" from "the along-line support can be located" could not have
fired.  `sample_strip` now returns `t`, the along-line coordinate of each
longitudinal sample, and the clipped GT endpoints project onto that same axis --
two independently derived quantities, so the comparison has content.  Measured
on real dev frames the support fraction is about 0.20 of the chord, median 0.14
per role, never 1.

The GT segment still supplies no feature value.  O1A and O1B read byte-identical
Scharr evidence; the oracle says only *which longitudinal portion* is the target.

**Training and the metric used different populations.**  `budget_losses`
reduced to a scalar with `F.smooth_l1_loss(..., reduction="mean")` before any
mask existed, while the reported metric masked to `length > 1e-4 & valid`.  So
the optimiser was being asked to correct roles the screen would not score --
including the off-frame ones above.  `budget_losses(reduce=False)` now returns
per-role terms and `masked_mean` averages over exactly
`edge_supported & coarse_supported & finite`.  `step_batch` lost its `train`
flag: one path, one mask.

Coarse validity is evaluated on the canonical grid rather than on each arm's own
feature rectangle, so F50 at 50x50 and F100 at 100x100 score the identical
frame-role population.  `evaluate` hashes that population and the decision step
refuses to compare arms whose hashes differ.

**Checkpoints.**  Epochs 1, 3 and 5 are written under
`line_capacity_v2/checkpoints/<ARM>/`, carrying arm, epoch, model, stem,
optimiser, seed, runner sha, split sha, radius, jitter and gate.  The epoch-5
decision is read back off disk and re-evaluated before it is acted on; a
mismatch raises `CHECKPOINT_RELOAD_PARITY_FAIL`.

**`--seed`.**  It existed on the command line and reached nothing.  Removed;
`SEED = 1` is the declared constant.

## Radius, re-derived on the corrected population

Geometry only, before any feature was judged.  A pair is covered when
`POINT_QUORUM = 0.90` of its *visible* points lie inside the strip.

```
                r=6       r=8       r=10      r=12      r=14
dev    pair     0.92136   0.98486   0.99830   0.99996   1.00000
       p1       0.000     0.516     1.000     1.000     1.000
train  pair     0.92235   0.98380   0.99856   0.99998   1.00000
       p1       0.000     0.453     1.000     1.000     1.000
```

Smallest radius clearing the 0.995 gate: **10**, unchanged from the configured
value, so the sampler did not move.  O0 was rerun anyway, because its population
did change -- off-frame roles are now excluded there too.

```
LINE_TRAIN   13,618 frames   788,790 pairs (5 epochs of jitter)   off-frame 5,658
LINE_DEV      2,393 frames    27,684 pairs (1 epoch of jitter)    off-frame 1,032
required radius   p50 2.45   p90 5.51   p95 6.63   p99 8.19
```

## Standing

O1 results before this fix: **none**.  Feature selection was unopened, and it
still is at the moment this document is written.  `bf9d8ab`, the V1 addendum and
the known-dimension formulation are untouched.  No PnP, no dimensions, no
`validation512`, no sealed set.
