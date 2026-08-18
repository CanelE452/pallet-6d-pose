# Pre-registration — A0 / A1 / A2 quick screen

Fixed before any arm was trained.  Gate wording is copied from the brief; the
numbers below it are the ones this repository can actually supply.

## Arms

```
A0_LINE_ONLY          vgg[19:27] + DirectHoughModel
A1_CORNER_LINE        + belief stages 4-6 trained, L_corner added
A2_CORNER_LINE_MASK   + seg stages trained,        L_mask added
```

Everything except the loss is held: same checkpoint, same init, same freeze
boundary (`vgg[19]`), AdamW lr 1e-3 / wd 1e-4, batch 8, seed 1, same data order,
same steps, same marks, same dev population, same checkpoint-selection rule
(the decision mark, never the best mark).

`mh_wiring.T3` proves the hold rather than asserting it: with lambda = 0 the A1
and A2 line outputs are bit-identical to A0's at step 0 **and after 20 optimiser
steps**, max abs diff 0.0, with the run's self-repeat also 0.0 under strict
deterministic kernels.

## Loss

```
A0   L = L_line
A1   L = L_line + lambda_corner * L_corner
A2   L = L_line + lambda_corner * L_corner + lambda_mask * L_mask
```

`L_line`   support-masked cross-entropy over the 27,000-hypothesis joint lattice
           (unchanged from the line stage).
`L_corner` channel-masked MSE on belief stages 4-6.  A corner whose centre falls
           outside the 50-grid is dropped from the mean, never supervised as
           background and never clamped to the border.
`L_mask`   BCE-with-logits over both seg stages, per-frame valid weighted --
           the formula already in `train.py:284-303`, not a new one.

## Lambda, measured not chosen

Gradient norms on the shared late-A1 block, 4 batches, before any optimiser step:

```
                    ||g|| on vgg[19:27]      ratio to line
line                7.530e-04                1.0
corner              1.595e-02               21.32
mask                1.043e+00             1391.05

lambda_corner = 0.03518    targets ratio 0.75   (brief: 0.5 - 1.0)
lambda_mask   = 5.392e-05  targets ratio 0.075  (brief: 0.05 - 0.10)
ramp          = 500 steps, linear, applied to both auxiliary terms
```

The brief marks those target ratios `[미검증]` and they stay marked.  Two things
about this measurement are recorded now so they are not discovered later: the
ratio is taken at step 0 with heads that are warm-started but partly wrong on
this data, so it will drift as they fit, and AdamW normalises per parameter, so
lambda governs how the heads mix into the *shared* block and not how fast each
head itself learns.

## Populations

```
MH_TRAIN        33,758   pool
D2_MH_DEV512       512   the decision population, and the only one
D0_MH_SEEN512      512   drawn from train; diagnostic, enters no gate
```

Group-disjoint on `background | scene_preset | floor_mode | floor_texture`,
overlap 0.  Stratum shares match train to within 0.003 on all eight strata.
`floor_mode` is the one axis that does not match (train native 0.215, dev 0.124)
because native floors exist as only eight large groups; recorded, not fixed.

## Schedule, fixed before the screen ran

```
pool     MH_TRAIN, all 33,758 frames, deterministic order
marks    500, 1000, 2000, 3000, 4000, 5000, 6000
decision 6,000 and only there -- the last mark, never the best one
```

Choosing the best mark afterwards would turn a seven-point ladder into a
seven-way search over the dev set, which is the reason the line stage writes its
decision step into the plan.  The 32-frame overfit showed all three arms
non-monotone at their last mark, so the ladder is reported in full and the
decision still comes from the pre-registered step.

## Step 0, before any optimiser step

Recorded because the line head starts at chance and its gradient reshapes the
shared block within a few hundred steps, dragging the warm-started heads down
before they recover -- the 32-frame overfit showed A1's corner decode going
2.9 -> 16.0 -> 1.7 cell.  Without this row there is nothing to read "A1's corner
head reached X" against.

```
D2_MH_DEV512   corner (direct)  2.9453 cell   in-grid only 1.9227
               corner (CIGM)   51.5711 cell   line head untrained
               mask IoU          0.605
D0_MH_SEEN512  corner (direct)  4.1358 cell   in-grid only 3.0124
               mask IoU          0.5408
```

## Gates

A1 keeps its head if **either** of:

```
1. line primary (angle median or offset median) improves >= 5% over A0
2. pose success improves >= 5 percentage points
3. truncation subset (V<8) improves >= 10% relative
```

and simultaneously:

```
non-truncation (V=8) primary regression <= 5%
```

or, failing all of that, if a clear representation complementarity is measured,
in which case the finding is headroom for fusion and not a win.

A2 keeps its head if, against A1:

```
corner or line primary improves >= 5%
or pose success improves >= 3 percentage points
```

Mask IoU alone never passes A2.  If mask improves and the main metrics do not,
the verdict is `MASK_AUX_VISUALISATION_ONLY`.

## Predicted before running

Recorded so it cannot be written after the fact.

`PATH-L` is expected to lose to `PATH-C` on full-view frames.  The line stage's
own noise budget puts CIGM->PnP at 1.0 degree and 0.5 cell, the best line arm
ever recorded reached 2.07 degree and 1.08 cell, and the response is roughly
linear at 4.3 px per degree and 9.9 px per cell, so CIGM corner error around
13-19 px is expected against about 8.5 px for a fitted corner head `[추정]`.
Complementarity, if it exists, is expected on V<8 and off-frame corners, where a
heatmap structurally cannot place a corner outside its own grid and CIGM can
place it by intersection.

## Out of scope

Real transfer, negatives, occlusion learnability, fusion, and any comparison to
the pre-2026-08-14 line-stage numbers.  See `PURPOSE.md`.
