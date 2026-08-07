# LOCAL_EDGE_EVIDENCE_PRECISION_FAIL

The question, stated before any arm was run:

> given a coarse line within +/-8 degrees and +/-4 cells, and a physical cuboid
> edge that intersects the image, can local image evidence refine that line to
> 1.0 degree and 0.5 cell?

Answer, on the corrected population: **no**, for every feature tested.  Nothing
came within a factor of three of the budget.

## The ladder

`n = 5,889` frame-roles, identical for all six arms (`population_sha
46378206fe5f5708`).  Identity means the coarse line left alone.

```
arm            stage      angle med   angle p90   offset med   offset p90
────────────────────────────────────────────────────────────────────────
identity       --             3.938       7.234        1.974        3.584
gate                        <=1.000      <=2.000      <=0.500      <=1.000

O1A            overfit32      0.002       0.004        0.001        0.001
               epoch 5        3.651       7.176        1.882        3.527
O1B            overfit32      0.012       0.017        0.011        0.016
               epoch 5        3.596       7.081        1.695        3.438
C0_F50         overfit32      0.000       0.001        0.000        0.000
               epoch 5        2.703       6.434        1.310        3.234
C1_F100        overfit32      0.000       0.000        0.000        0.000
               epoch 5        3.002       6.720        1.503        3.249
C2_MULTI       overfit32      0.004       0.008        0.005        0.011
               epoch 5        2.727       6.487        1.365        3.204
C3_RGB_STEM    overfit32      0.018       0.046        0.013        0.025
               epoch 5        3.335       7.075        1.889        3.570
```

Reduction against identity at epoch 5:

```
arm            angle    offset
──────────────────────────────
O1A             -7.3%    -4.7%
O1B             -8.7%   -14.2%
C0_F50         -31.3%   -33.7%
C1_F100        -23.7%   -23.9%
C2_MULTI       -30.8%   -30.9%
C3_RGB_STEM    -15.3%    -4.3%
```

Epochs 1 and 3 are in `line_capacity_v2_arms.json`; every arm improves
monotonically and flattens by epoch 3, so epoch 5 is not a truncation artefact.
The decision reads epoch 5 only.

## What each result rules out

**The refiner is not the bottleneck.**  O0, the perfect-evidence oracle, reaches
0.0283 degree and 0.0158 cell -- two orders of magnitude inside the gate.  Every
arm also drives its 32-frame overfit set to essentially zero.  Representation and
optimisation are both fine; what fails is reading an unseen image.

**Raw gradient carries almost nothing here.**  O1A ends 7% below identity.  A
Scharr magnitude and orientation field, sampled over the full visible chord, is
close to no information at this precision.

**Knowing where the edge lies along the chord does not unlock it.**  That was the
point of O1B, and it is why the no-op mask mattered: with the oracle actually
connected, the support fraction is 0.151 of the chord and the arm gains 14% on
offset against O1A's 5%, but stays 3.6 degrees from a 1.0 degree gate.  So
`ALONG_LINE_SUPPORT_LOCALIZATION_REQUIRED` does not fire.  Along-line
localisation is not the missing ingredient.

**The A1 features carry roughly four times more than raw gradient, and it is
still not enough.**  F50 and multi-scale reach about -31%, F100 -24%, the shallow
RGB stem -15%.  Ordering is stable across both metrics: a learned mid-level
representation genuinely beats an edge filter for this task, by a margin that is
not close to the gate.

Taxonomy: O0 PASS, O1B FAIL, all learned arms FAIL -> **LOCAL_EDGE_EVIDENCE_PRECISION_FAIL**.

## The confound this run does not exclude

Every arm fits 32 frames perfectly and then generalises to about a 30% error
reduction.  That gap is the signature of a data-limited fit as much as an
evidence-limited one, and the arms trained on `line_search2k` -- 2,000 frames of
a 13,618-frame `LINE_TRAIN`.  The 2,000-frame budget was fixed in advance by the
nested-search design, so this is the result the declared protocol produces, but
"no feature carries this precision" and "no feature carries it from 2,000
frames" are not the same statement and this run cannot separate them.

Two observations bear on it without settling it.  The gap between arms is large
and ordered (F50 -31% against O1A -7%), which is a feature effect rather than a
uniform data ceiling.  And every arm has flattened by epoch 3, which is what a
data limit looks like rather than an unfinished optimisation.  If this line is
pursued, the cheap next measurement is the same six arms on the full
`LINE_TRAIN`, changing nothing else.

## Population

The corrected definition, from `coverage_fullsplit.json`:

```
LINE_TRAIN   13,618 frames   788,790 pairs   off-frame 5,658   partial 21,075
LINE_DEV      2,393 frames    27,684 pairs   off-frame 1,032   partial  3,741
strip radius 10 cell   pair coverage dev 0.99830 / train 0.99856   gate 0.995
```

Per dev evaluation: 6,144 roles, 223 off-frame, 32 coarse-invalid, 0 degenerate,
0 non-finite, **5,889 used**.  Roles whose edge never enters the image are
excluded from loss and metric alike -- they have no local evidence by
construction.  Whether a global predictor can place an edge it cannot see is a
real question and a different one; this gate does not answer it.

## Provenance

```
runner       scripts/stage0/line_feature_capacity_v2.py  (0efcfd1)
split sha    70ba7f1e8832bb0c...        seed 1 (locked, no CLI override)
checkpoints  line_capacity_v2/checkpoints/<ARM>/epoch_{001,003,005}.pth
reload       epoch-5 re-evaluated from disk, max delta 0.0 for all six arms
```

No PnP, no dimensions, no `validation512`.  `untouched`, `eval56`, `wood45` and
final-test remain unopened.
