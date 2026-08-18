# CORNER_LINE_MULTIHEAD_SCREEN — no arm earned promotion, and the reason is measurable

Three arms, one shared backbone, one factor each.  Two seeds.  The screen answers
its three questions, and the answer to the first one is that the screen cannot
answer it at this budget -- which is itself the most actionable finding here.

```
A0_LINE_ONLY          vgg[19:27] + DirectHoughModel
A1_CORNER_LINE        + belief stages 4-6 trained, L_corner added
A2_CORNER_LINE_MASK   + seg stages trained,        L_mask added
```

Everything else held: same checkpoint, same freeze boundary, AdamW 1e-3 / 1e-4,
batch 8, same marks, same dev population, same 500-step ramp.  `mh_wiring.T3`
proves the hold rather than asserting it -- with lambda = 0 the A1 and A2 line
outputs are bit-identical to A0's at step 0 and after 20 optimiser steps, max abs
diff 0.0, self-repeat 0.0 under strict deterministic kernels.

## The table

`D2_MH_DEV512`, n = 512, step 6,000, both seeds.  Group-disjoint from train.

```
arm                    seed    angle    offset   CIGM cor  direct cor    R_L      R_C   maskIoU
------------------------------------------------------------------------------------------------
(step 0 reference)        -        -         -    51.5711      2.9453       -        -    0.605
A0_LINE_ONLY              1   3.1676    1.8233     3.1289           -   21.15        -        -
A0_LINE_ONLY              2   3.5961    1.7877     2.9384           -   19.26        -        -
A1_CORNER_LINE            1   2.9756    1.5860     2.6695      1.0833   18.95    14.01        -
A1_CORNER_LINE            2   3.6126    1.8758     3.1059      1.0775   22.79    13.69        -
A2_CORNER_LINE_MASK       1   3.2488    1.6797     2.8165      0.9794   19.15    12.65    0.737
A2_CORNER_LINE_MASK       2   3.7099    2.1229     3.5328      1.1297   21.63    14.75    0.732
```

angle in degrees, offset and corner in canonical 50-grid cells, R in degrees.
`5cm5deg` is 0.000 for PATH-L in every arm and every seed, and 0.037-0.039 for
PATH-C.  Nothing here is a working pose.

## Q1 -- is corner + line better than line only?

**NOT_ESTABLISHED.**  Not "no"; the screen cannot resolve it.

```
A1 vs A0 (positive = A1 better)   angle     offset   CIGM corner
seed 1                           +6.06%   +13.02%      +14.68%
seed 2                           -0.46%    -4.92%       -5.70%
mean                             +2.80%    +4.05%       +4.49%    gate needs >= +5%
```

The sign flips.  On one seed A1 clears the gate comfortably on both line axes; on
the other it is behind on both.  The mean is under the gate on all three.

The reason is not speculation, it was measured:

```
arm                  metric    seed1 / seed2      spread
A0_LINE_ONLY         angle    3.1676 / 3.5961     12.7%
A0_LINE_ONLY         offset   1.8233 / 1.7877      2.0%
A1_CORNER_LINE       angle    2.9756 / 3.6126     19.3%
A1_CORNER_LINE       offset   1.5860 / 1.8758     16.7%
A1_CORNER_LINE       cornerL  2.6695 / 3.1059     15.1%
A1_CORNER_LINE       cornerC  1.0833 / 1.0775      0.5%
```

**The line branch swings 15-19% between seeds and the corner branch swings 0.5%.**
Same backbone, same run, same optimiser -- a fortyfold difference in
reproducibility between two heads.  A 5% effect cannot be read off a single seed
in the line branch, and can be read off one seed in the corner branch.

Scope: this is 6,000 steps on `v2_prod40k_clean_merged` with two seeds that varied
initialisation and data order together.  It does not invalidate the earlier
line-stage screens, which ran 25,545 steps on different data -- F1's +44.58% /
+45.39% sits far outside this band.  What it does is close, for these conditions,
the item `DIRECT_HOUGH_OVERFIT_EXTENSION_RESULT.md` left open as "run-to-run
variability UNMEASURED, and it straddles the gate at 3,000".  Here it is measured,
and it is larger than the effect.

## Q2 -- is 3-head better than 2-head?

**NO.**  `MASK_AUX_NO_MAIN_TASK_GAIN`.

```
A2 vs A1 (positive = A2 better)  angle    offset   CIGM cor  direct cor     R_C
seed 1                          -9.18%   -5.91%     -5.51%     +9.59%   +9.68%
seed 2                          -2.69%  -13.18%    -13.74%     -4.84%   -7.70%
mean                            -5.94%   -9.54%     -9.63%     +2.37%   +0.99%
```

The line branch is worse on **both** seeds, on both axes.  The corner gain that
looked convincing on seed 1 does not reproduce.  The mask head itself works --
IoU 0.605 at step 0 to 0.737 / 0.732, stable across seeds -- so this is not a
head that failed to train.  It trained, and the main task did not improve.

Per the brief the mask head is therefore removed from the main architecture.

One defect in the pre-registered gate is worth recording rather than quietly
working around: the A2 gate reads "corner **or** line primary >= 5%" with no
regression guard, while the A1 gate has one.  On seed 1 alone A2 would have
passed on its corner clause while regressing line by 9%.  The two-seed result
makes the point moot, but a one-seed run would have promoted a trade.

## Q3 -- are the two representations complementary?

**Weakly, and not enough to build fusion on.**  This is the one conclusion that
is stable across seeds.

```
A1              direct   CIGM    tie   direct med  CIGM med  oracle  gain%
seed 1  ALL     73.9%   22.3%   3.7%     1.083     2.669    0.985    9.1%
seed 2  ALL     75.7%   21.5%   2.9%     1.078     3.106    1.000    7.2%

corners whose GT falls outside the belief grid (n = 304, 7.4% of all corners)
seed 1          58.2%   39.1%   2.6%     4.342     5.476    3.734
seed 2          54.3%   43.4%   2.3%     4.673     5.413    3.581

per-corner direct win rate 69.9% - 80.3%; direct leads on all eight
```

Three separate readings, because they do not all say the same thing.

1. **Direct dominates everywhere.**  74-76% of corners, every subset, every corner
   index.  The pre-registered prediction was that PATH-L would lose to PATH-C on
   full-view frames; it loses more broadly than that.

2. **The off-frame hypothesis has the right direction and the wrong conclusion.**
   Off the grid CIGM's win rate roughly doubles, 21% to 39-43%, exactly the
   division of labour predicted.  But its median error there is still worse than
   the heatmap's (5.41-5.48 against 4.34-4.67).  A heatmap does not return nothing
   for an off-grid corner -- it peaks at the grid edge, and for a corner just
   outside that is only a few cells wrong.  Off-grid corners are 7.4% of the
   population and mostly sit just outside.

3. **Oracle-min buys 7.2-9.1%.**  That is a ground-truth chooser, an upper bound
   no confidence rule can reach.  Section 17's own criterion -- one branch
   dominating almost everywhere means fusion is weakly motivated -- is met, so no
   fusion was built.

## The pose path is correct, and the corner residual is not noise

Checked because a wrong PnP would invalidate the whole pose column:

```
input                       R med      R p90   t med (m)
GT pixels (exact)          0.0000     0.0198      0.0000
GT -> 50-grid -> px        0.0000     0.0198      0.0000
GT + 1.0 cell noise        5.5983    22.7868      0.1037
GT + 2.7 cell noise       17.4117    85.6079      0.3705
```

PnP recovers ground truth exactly, and the grid round trip is lossless.  Against
that, the measured arms:

```
A1 PATH-L, CIGM error 2.67 cell    R 18.95    vs 17.41 simulated   as predicted
A1 PATH-C, corner error 1.08 cell  R 14.01    vs  5.60 simulated   2.5x worse
```

PATH-L behaves like isotropic noise of its own magnitude.  PATH-C does not: the
corner head's residual shakes the pose two and a half times harder than random
error of the same size, which means it is **systematic, not noise**.  Halving the
corner error would not buy the pose improvement the noise model promises.

This independently reproduces the STAGE8 oracle-PnP finding -- "the bottleneck is
not the size of the corner error but its systematic structure" -- on different
data, a different dataset generation, and a different model lineage.

## What the corner head did do

Both arms improved the warm-started corner head by about 63% (2.9453 cell at step
0 to 1.08 / 1.05 averaged over seeds), and A1's corner is reproducible to 0.5%.
The corner branch trains well and measures reliably.  It simply did not hand that
reliably on to the line branch.

## Standing

```
A0_LINE_ONLY                        remains the baseline
A1_CORNER_LINE                      NOT_ESTABLISHED, effect inside the seed band
A2_CORNER_LINE_MASK                 REJECT, mask aux gives no main-task gain
DUAL_HEAD_FUSION                    not motivated, not built
CIGM adapter                        BUILT and verified: GT lines -> 0.0027 cell
                                    median, reproducing cigm_oracle.json's 0.0029
CIGM in the historical runners       still BLOCKED, untouched, tests still assert it
LINE_BRANCH_SEED_VARIANCE           15-19% at 6,000 steps, MEASURED
CORNER_BRANCH_SEED_VARIANCE         0.5%, MEASURED
CORNER_RESIDUAL_SYSTEMATIC          2.5x worse than isotropic noise of equal size
full 50k training                   not started, and not recommended yet
real transfer, negatives, occlusion  out of scope, unchanged
```

## What would make the next screen decide something

The blocker is measurement resolution, not architecture.  A 5% gate cannot be read
through a 15-19% band.  Two ways out, and the first is cheaper than it looks:

1. **More steps.**  The band was measured at 6,000 steps, which is 1.4 passes.  The
   historical line runs decided at 25,545 and F1's margin there was 45%, far
   outside any plausible band.  Whether the band narrows with exposure is itself
   unmeasured and is one cheap A0 run at two seeds away from being known.
2. **Three or more seeds per arm**, reporting the paired distribution rather than a
   point.  At the current cost that is roughly an hour per arm per seed.

Either way, A1 deserves to be re-asked rather than dropped: its mean across two
seeds is positive on all three line metrics (+2.80 / +4.05 / +4.49%), it is just
not separated from zero.
