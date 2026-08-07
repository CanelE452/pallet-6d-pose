# What the V2 gate actually established

`406ecf8` stands unedited.  Its numbers are real and its arms genuinely failed.
What it could not separate is *why*, and this narrows the claim to what the run
supports.

```
original decision        LOCAL_EDGE_EVIDENCE_PRECISION_FAIL
scientific qualification SEARCH2K_LINE_REFINEMENT_FAIL   = CONFIRMED
                         DATA_SCALE_UNRESOLVED           = TRUE
                         LOCAL_EDGE_EVIDENCE_PRECISION_FAIL = PROVISIONAL
                             until the data x step diagnostic closes
```

Every arm drove a 32-frame overfit set to roughly zero and then generalised to
about a 30% error reduction.  That is the signature of a data-limited fit as
much as an evidence-limited one, and the arms saw `line_search2k` -- 2,000 of
13,618 frames.

## Counts, stated correctly

The earlier documents called 788,790 a number of *pairs*, which reads as a
sample size.  It is not: it is the same roles seen under five epochs of jitter.

```
LINE_TRAIN   frames                     13,618
             total roles               163,416   = 13,618 x 12
             off-frame roles             5,658   no local evidence by construction
             unique supported roles    157,758
             5-epoch role exposures    788,790

LINE_DEV     unique supported roles     27,684
```

`coverage_fullsplit.json` now reports `unique_supported_roles` and
`role_exposures` separately.  The word "pairs" is retired.

## The three candidate causes

```
feature   the representation cannot carry 1.0 deg / 0.5 cell
data      2,000 unique frames are too few
steps     835 optimizer updates are too few
```

Running the full split for five epochs would move data *and* steps at once and
settle none of them.  Two diagnostics follow.

## O1C, the hard-support oracle

O1B handed the refiner a support channel but left the feature values outside the
target segment intact, so the arm still had to select among clutter.  O1C
multiplies the Scharr magnitude and both orientation components by that support,
so everything off the physical edge is literally zero.  Ground-truth orientation
and offset remain forbidden as input; only the along-line support is oracular.

```
O1C PASS   the pixels do carry the precision; the bottleneck is selecting the
           edge out of clutter
O1C FAIL   even the target edge's own gradient does not carry it in this
           representation
```

Diagnostic only.  It is not a candidate architecture.

## The data x step factorial

Conditions, with the step counts derived from the runner's real chunking
(batch 12, a tail chunk of fewer than 2 dropped) rather than assumed:

```
S_SHORT   835   = 167 steps/pass x 5   (line_search2k, 2,000 frames)
S_LONG   5,675  = 1,135 steps/pass x 5 (LINE_TRAIN, 13,618 frames)

A_2K_SHORT     2,000 frames    835 steps    reused, not retrained
B_FULL_SHORT  13,618 frames    835 steps
C_2K_LONG      2,000 frames  5,675 steps
D_FULL_LONG   13,618 frames  5,675 steps
```

Two trajectories cover all four.  The schedule, the seed and the jitter are
deterministic, so the 2k trajectory's state at step 835 *is* the fresh-init
835-step run, and likewise for the full pool.  That also buys a check: 2k at
S_SHORT must reproduce the recorded search2k epoch-5 numbers exactly, and
`scale-decide` raises `CONDITION_A_NOT_REPRODUCED` if it does not.

Jitter is keyed on `(frame, role, visit, purpose)`, where visit counts passes
over the pool.  In the original run one pass was one epoch, so visit equals
epoch there and condition A is reproduced bit-for-bit.  Cycling 2,000 frames for
5,675 steps therefore draws fresh jitter on each of its 34 visits, and data
diversity is not confounded with jitter diversity.

Arms: `C0_F50` (primary), `C2_MULTI`, `C3_RGB_STEM`.  `C1_F100` lost clearly to
F50 at 2k and is dropped.  O1A is not a candidate.

## Thresholds, fixed before the run

The gate is unchanged and is the only thing that decides PASS:

```
angle median <= 1.0 deg    angle p90 <= 2.0 deg
offset median <= 0.5 cell  offset p90 <= 1.0 cell
```

The scaling threshold is diagnostic and does not replace it:

```
SCALING_SIGNAL_PRESENT   D vs A: angle reduction >= 40% AND offset >= 40%
APPROACHES_GATE          D: angle <= 1.5 deg AND offset <= 0.75 cell
```

Attribution:

```
B and C no better than A, all fail   DATA_SCALE_NOT_THE_BOTTLENECK
B < A or D < C                       DATA_DIVERSITY_HELPS
C < A                                OPTIMIZATION_STEPS_HELP
D PASS                               DATA_SCALE_RESCUES_LINE_REFINEMENT
D fails, scaling signal present       SCALING_SIGNAL_PRESENT_BUT_INSUFFICIENT
otherwise                            LOCAL_EDGE_REPRESENTATION_PRECISION_FAIL
```

SLQ is built only if some arm reaches D PASS or `APPROACHES_GATE`.  Otherwise the
next direction is an explicit line/edge representation or a line-supervised
encoder, not this refiner with more data.

No PnP, no dimensions, no `validation512` tuning.  `untouched`, `eval56`,
`wood45` and final-test remain unopened.
