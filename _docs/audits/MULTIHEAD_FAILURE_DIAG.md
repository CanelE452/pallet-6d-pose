# Why sharing a backbone between the corner head and the line head did not pay

A1 at 25,000 steps does not improve the line branch over A0, on either seed, while
its own corner head is excellent and reproducible to 0.8%.  Both halves work; the
combination does not.  This separates three explanations, cheapest first, and the
answer is not the one the experiment was designed around.

```
H1  the corner gradient fights the line gradient in the shared late block
H2  the two tasks can share early features but not late ones
H3  the representations are complementary and CIGM discards the evidence
    between lines, corners and PnP
```

Provenance: repo HEAD `8962368`, `mh_diag_provenance.json`,
`BASELINE_RECONFIRM = PASS`.  A0/A1 25k results and checkpoints were reused as
controls and never recomputed.  No historical runner was modified; the five test
files that assert `CIGM: BLOCKED` still assert it.  No sealed set was touched.

---

## The short answer

**H1 is rejected, and rejected for a reason nobody was looking for.** The two
gradients do not conflict -- they agree weakly, cosine +0.13 to +0.25 at every
checkpoint on both seeds.  But the corner gradient reaching the shared block is
**three orders of magnitude smaller than the line gradient**, and it got that way
during training:

```
||g_corner|| / ||g_line||  on the shared late-A1 block
step 0 (calibration)   21.3
step 6,000              6.5e-04
step 25,000             5.9e-04       a ~35,000x collapse
```

`lambda_corner = 0.03518` was fixed from the step-0 ratio.  Multiply the two and
the corner loss contributes about **2e-05 of the line gradient** to the shared
trunk.  Whatever A1 does differently from A0, it is not corner supervision
reshaping shared features -- there is almost no corner supervision arriving.

To hold the intended 0.75 ratio at 25,000 steps, `lambda_corner` would have to be
roughly **1,300** rather than 0.035.

**H3 is partly confirmed.**  CIGM really does throw line information away: the
same predicted lines that give a *worse* pose than points when intersected into
corners give a *better rotation* than points when used directly as constraints.
The native solver still fails its pre-registered gate, on translation.

**A new cause appears that was not in the hypothesis set.** The corner residual is
strongly structured, and the mode that damages pose is not the mode that carries
the variance.

---

## PHASE 1 -- gradient interference

512 train frames disjoint from both evaluation manifests, 64 fixed batches, the
same frames and order at every checkpoint, `torch.autograd.grad` only, no
optimiser step.

```
seed/step         |g_line|   |g_corner|      ratio   cos med   cos p10   neg frac
seed1 @ 6000     2.761e+00   1.808e-03   6.549e-04   +0.1818   ...        0.08
seed1 @12000     3.746e+00   2.051e-03   5.475e-04   +0.1896              0.12
seed1 @18000     3.157e+00   2.061e-03   6.527e-04   +0.1303              0.14
seed1 @25000     3.672e+00   2.170e-03   5.911e-04   +0.1437              0.12
seed2 @ 6000     2.927e+00   2.262e-03   7.729e-04   +0.2481              0.09
seed2 @12000     3.111e+00   2.310e-03   7.426e-04   +0.1715              0.09
seed2 @18000     3.320e+00   2.510e-03   7.559e-04   +0.1901              0.03
seed2 @25000     3.571e+00   2.026e-03   5.673e-04   +0.1312              0.05

gate      cos median < -0.10  AND  negative fraction >= 0.60
observed  cosine positive everywhere, negative fraction 0.03 - 0.14
GRADIENT_CONFLICT_SUPPORTED = False
```

Per tensor at the decision step, ordered by how much of the line gradient each
carries -- no sub-block hides a conflict either:

```
tensor        cos median   neg frac   share of line gradient
25.weight        +0.1289       0.19        0.525
23.weight        +0.1427       0.12        0.197
21.weight        +0.1536       0.11        0.135
19.weight        +0.1658       0.09        0.131
```

The block gate needed a sub-block with cosine < -0.10, negative fraction >= 0.60
and >= 25% of the line gradient.  The four tensors carrying 98.8% of the line
gradient are all positive.

### The mechanism behind the collapse

The two norms move in opposite directions during training:

```
                 step 0        step 25,000     change
||g_line||       7.53e-04      3.67e+00        x 4,900
||g_corner||     1.60e-02      2.17e-03        / 7.4
```

At step 0 the line head is randomly initialised, so almost nothing propagates
back through it and the corner head -- warm-started and badly wrong on this
dataset -- dominates.  As the line head's weights grow, its gradient to the trunk
grows with them, while the corner head fits and quiets down.  Lambda was measured
at the single moment when the ratio was most favourable to the corner branch, and
`SCREEN_LOCK.md` warned the ratio would drift.  It drifted by 35,000x.

This is a general trap, not a detail of this run: **calibrating a multitask
weight from step-0 gradient norms measures the initialisation, not the training.**

---

## PHASE 2 -- what shape is the corner error?

`SYSTEMATIC_CORNER_BIAS_SUPPORTED = True`, on both seeds, by both criteria.

```
                          seed1     seed2
top-3 PC explained        67.2%     68.1%      (gate >= 50%)

measure                s1 median   rho_R    s2 median   rho_R
front_rear_shift          0.9543  +0.643       0.9981  +0.631
centroid_shift            0.4966  +0.420       0.4957  +0.474
nonaffine_rms             0.6362  +0.398       0.6320  +0.445
affine_scale_isotropic    0.9523  -0.368       0.9631  -0.376
front_area_ratio          0.7545  +0.035       0.8836  +0.032
height_ratio              0.7822  +0.071       0.9098  +0.074
depth_ratio               0.9675  -0.146       0.9602  -0.245
width_ratio               0.9569  -0.241       0.9620  -0.203
affine_rotation_deg      -0.2310  +0.083      -0.6062  +0.031
affine_shear             -0.0119  -0.107      -0.0046  -0.030
```

The dominant mode is `front_rear_shift` -- the predicted displacement between the
front-face centroid and the rear-face centroid is wrong, and that error is what
moves the rotation.  Alongside it the whole box is predicted about 4-5% too small
(`affine_scale_isotropic` 0.952 / 0.963), with the front face 12-25% and the
height 8-22% short.  That is the flatten / depth-compression mode this project has
named before, measured here on new data and a new model lineage.

**The loud modes are not the harmful ones.**  PC1 carries 30% of the residual
variance and correlates with rotation error at only -0.19; `front_rear_shift`
carries no particular share of the variance and correlates at +0.64.  A
correction built on the principal components would aim at the wrong thing.

---

## PHASE 5 -- does CIGM hide the complementarity?

Lambda chosen on `D0_MH_SEEN512` from a grid locked before the sweep
(0.03 / 0.1 / 0.3 / 1.0 / 3.0), Huber scale 5.0 px locked, spent once on
`D2_MH_DEV512`.  Both seeds selected lambda = 1.0.  No confidence weighting.

```
seed 1                R med   R p90    t med   reproj   5cm5deg
ALL  F0 point-only    7.830   41.75   0.2244     8.96    0.0781
ALL  F1 CIGM         11.997   39.80   0.3419    18.16    0.0117
ALL  F2 joint         7.125   26.79   0.2389     9.83    0.0586
V<8  F0               9.721   48.19   0.3192    14.99    0.0351
V<8  F2               7.902   26.77   0.2980    14.16    0.0585

seed 2
ALL  F0               8.067   33.02   0.2397     9.19    0.0938
ALL  F1              11.397   39.41   0.5788    23.31    0.0020
ALL  F2               6.964   29.27   0.2951    11.28    0.0352
V<8  F0               9.761   45.09   0.2608    14.08    0.0702
V<8  F2               7.442   39.87   0.3827    17.24    0.0292

gate   V<8 R >= +10%  AND  V<8 t >= +10%  AND  V=8 R,t degradation <= 5%
       AND overall 5cm5deg must not fall
POINT_LINE_SOLVER_SIGNAL = False
```

The gate fails, and it fails on translation.  What it fails *around* is the
finding:

```
V<8 rotation   seed 1  9.721 -> 7.902   +18.7%
               seed 2  9.761 -> 7.442   +23.8%
ALL rotation   seed 1  7.830 -> 7.125   + 9.0%
               seed 2  8.067 -> 6.964   +13.7%
ALL R p90      seed 1 41.75 -> 26.79    +35.8%
```

The same predicted lines, routed through CIGM into corners, give R = 11.4-12.0
degrees -- far worse than points alone.  Routed directly into the pose objective
as line constraints, they beat points alone on rotation, on both seeds, in every
subset, with a large tail improvement.  **The intersection step, not the line
prediction, is what destroys the information.**

Translation goes the other way (0.2397 -> 0.2951 overall on seed 2), which
PHASE 2 explains: the predicted box is 4-5% too small, so a solver asked to
satisfy line constraints buys angular agreement by moving depth.  Fixing the
scale bias is a precondition for a joint solver, not an afterthought.

---

## PHASE 6 -- does the model know which of its lines are wrong?

`LINE_UNCERTAINTY_SIGNAL = True`.

```
        entropy~angle  entropy~offset  margin~angle  margin~offset  peak~angle     n
seed 1         +0.486          +0.438        -0.331         -0.301      -0.477  5960
seed 2         +0.413          +0.394        -0.290         -0.281      -0.403  5960

roles of 12 meeting the per-role threshold
seed 1   entropy~angle 9   entropy~offset 10   margin~angle 4   margin~offset 4
seed 2   entropy~angle 8   entropy~offset  8   margin~angle 4   margin~offset 2
```

Hough entropy clears +0.35 on both axes and both seeds, with 8-10 of 12 roles
agreeing individually.  Margin does not.  So a confidence-weighted point-line
solver has something real to weight by -- which matters because PHASE 5 ran
deliberately uniform.

---

## PHASE 7 -- which corner and which role does the pose depend on?

Ground-truth geometry perturbed, no model.

```
corner  +-0.5 cell    dR 0.92 - 1.36 deg   (0,1 near-top and 6,7 far-bottom worst)
role    +-0.5 deg / +-0.25 cell, through CIGM
                      dR 0.37 - 0.75 deg   (roles 0 and 11, the width edges, worst)
```

The spread is 1.5x across corners and 2x across roles.  **No single structural
role dominates the pose**, so there is no obvious candidate for a targeted
weight.  Recorded as an audit; nothing here became a training weight.

---

## PHASE 11 -- paired frame bootstrap

10,000 resamples, paired by frame so the enormous frame-to-frame spread cancels
instead of drowning the effect.  Seeds are never pooled: two seeds is n=2 and
stays n=2.  The statistic here is the median over a frame's supported roles, then
the median over frames -- slightly different from the role-pooled median used
elsewhere, which is why the point values differ in the third decimal.

```
comparison                   n    a med    b med   improve            95% CI   P(b better)
A1_vs_A0 seed1 angle       512   2.2093   2.2788    -3.15%   [-12.36, +6.06]        0.229
A1_vs_A0 seed2 angle       512   2.3186   2.3866    -2.93%   [-13.50, +5.34]        0.279
A1_vs_A0 seed1 offset      512   0.9885   0.9676    +2.11%    [-6.24, +8.64]        0.656
A1_vs_A0 seed2 offset      512   1.0299   1.0599    -2.91%   [-10.73, +3.94]        0.213
A1_vs_A0 seed1 cigm        512   1.7115   1.6152    +5.62%   [-2.56, +13.22]        0.891
A1_vs_A0 seed2 cigm        512   1.7487   1.9196    -9.77%   [-20.28, -1.70]        0.009
A1_vs_A0 seed1 R_L         512  12.1080  11.9973    +0.91%   [-11.27, +9.55]        0.564
A1_vs_A0 seed2 R_L         512  12.0436  11.3971    +5.37%   [-7.15, +13.36]        0.785

A2_vs_A1 seed1 offset      512   0.9676   1.0484    -8.35%   [-17.16, -2.43]        0.005
A2_vs_A1 seed1 cigm        512   1.6152   1.7718    -9.69%   [-18.06, -2.68]        0.002
A2_vs_A1 seed2 R_C         512   8.0668   9.0606   -12.32%   [-24.57, -2.37]        0.008
A2_vs_A1 seed1 direct      512   0.7396   0.7286    +1.49%    [-5.25, +5.91]        0.682
A2_vs_A1 seed2 direct      512   0.7396   0.7673    -3.75%    [-9.39, +2.10]        0.131
A2_vs_A1 seed2 angle       512   2.3866   2.3001    +3.63%    [-3.98, +9.90]        0.807
```

Two things this settles that point estimates could not.

**A1 against A0 is unresolvable, and now with intervals rather than a seed
disagreement.**  Every A1-vs-A0 confidence interval straddles zero.  The widths
are roughly +-10%, which is the same order as the seed band measured earlier, so
the frame bootstrap and the seed spread agree about how much resolution this
screen has: not enough for a 5% effect.

The CIGM row is the sharpest illustration.  Seed 1 says A1 is 5.62% *better*
with P(better) = 0.891; seed 2 says 9.77% *worse* with a CI that excludes zero,
P = 0.009.  Two seeds, both near or past significance, pointing opposite ways.
Any single-seed run here would have produced a confident and wrong sentence.

**A2 against A1 is resolved, against A2.**  Three intervals exclude zero --
offset -8.35%, CIGM -9.69% on seed 1, and PATH-C rotation -12.32% on seed 2 --
and every one of them says A2 is worse.  No metric on either seed shows A2 better
with an interval that excludes zero.  The specific axis that reaches significance
differs between seeds, so the claim is "A2 is worse somewhere, consistently", not
"A2 is worse by X% on offset".

## PHASE 3 -- stop-grad causal screen

One artefact must be recorded before the numbers, because it affects how they may
be read.  All three arms restart from A0 @18,000 with a **fresh** optimiser, and
that alone costs accuracy that 3,000 steps does not fully repay:

```
original A0 seed1, continuous optimiser      E0 continuation, fresh optimiser
  @18,000  angle 2.2163  offset 1.0787         +0     2.2163  1.0787   (source matches)
  @25,000  angle 2.2051  offset 0.9693         +250   2.2504  1.0119
                                               +500   2.2918  1.0774
                                               +1000  2.5589  1.1157   (+15% worse)
                                               +2000  2.4505  1.0833
                                               +3000  2.3094  1.0866   (still behind)
```

Re-estimating Adam's moments costs about 15% of the line accuracy and has not
recovered by the end of the budget.  `CONTINUATION_OPTIMIZER = FRESH` is recorded
in every result file.  All three arms pay this equally, so **E0/E1/E2 may be
compared with each other and must not be compared with A0 @25,000.**

### The result

```
arm                      seed   step    angle   offset  ang p90     CIGM   direct   dir@0
E0_CONTINUE_LINE            1   3000   2.3094   1.0866   11.611   1.8233      -       -
E2_STOPGRAD_CORNER          1   3000   2.3094   1.0866   11.611   1.8233   0.9342  21.7668
E1_SHARED_CORNER_LINE       1   3000   2.3675   1.0822   11.188   1.8422   0.9570  21.7668

E0_CONTINUE_LINE            2   3000   2.2994   1.1019   12.895   1.8929      -       -
E2_STOPGRAD_CORNER          2   3000   2.2994   1.1019   12.895   1.8929   1.0173  21.7308
E1_SHARED_CORNER_LINE       2   3000   2.4600   1.1428   12.937   1.8982   0.9987  21.7308
```

**E2 equals E0 to the last decimal, on every metric and both seeds** -- the
paired bootstrap returns +0.00% with a CI of exactly [+0.00, +0.00].  That is not
a close call, it is a structural consequence: with `f50.detach()` the corner loss
reaches neither the shared block nor the line head, so E2's line branch receives
the same gradients as E0's and, under deterministic kernels, walks the same
trajectory.  Read it as a wiring proof rather than a result -- it shows the
stop-grad does exactly what it claims.

So the comparison that carries information is E1 against E0, which is also E1
against E2:

```
E1_vs_E0   seed1 angle   -3.87%   CI [-11.15, +5.58]   P 0.237
           seed2 angle   -5.02%   CI [-11.94, +0.60]   P 0.043
           seed1 offset  +1.45%   CI  [-8.18, +7.95]   P 0.521
           seed2 offset  -5.36%   CI [-11.34, +2.96]   P 0.110
```

Letting the corner gradient into the shared block makes the line worse.  Each
interval individually straddles zero, but this is now the fourth independent
measurement of the same quantity and all four have the same sign:

```
A1 vs A0 @25,000   seed1 angle -3.10%   seed2 -4.93%
E1 vs E0 @ 3,000   seed1 angle -3.87%   seed2 -5.02%
```

Four of four negative, all between -3.1% and -5.0%.  A sign test on four
same-signed observations gives p = 0.0625 one-tailed -- not significance, but not
what chaos looks like either.

### The mechanism is probably not chaos

`||g_corner|| / ||g_line||` is 6e-04 and lambda multiplies it to about 2e-05, so
the obvious explanation for any difference is trajectory divergence in a
non-convex landscape.  **Divergence would give random signs, and the signs are
not random.**

A candidate that is systematic rather than chaotic `[추정]`: AdamW normalises each
parameter by its second moment, so a small extra gradient component on the shared
parameters raises `v` and *lowers* their effective step size.  At a fixed step
budget the line branch would then learn slightly slower, consistently, in the
direction observed.  This is testable -- switch to SGD, or compare at matched
loss instead of matched steps -- and is not claimed here.

### The corner head does not need the shared gradient

```
direct corner @3,000        E1        E2 (stop-grad)     bootstrap
seed 1                    0.9570        0.9342           +3.86%  P 0.927
seed 2                    0.9987        1.0173           -0.50%  P 0.523
both start from                21.7 (A0 never trained the belief stages)
```

Both arms take the corner head from 21.7 cells to about 0.95 in 3,000 steps, and
they are indistinguishable.  **Sharing the trunk costs the line branch and buys
the corner branch nothing.**  Stop-grad dominates: it is free on the line side by
construction and gives the same corner head.

---

## PHASE 4 -- late feature separation (E3_SPLIT_LATE)

The pre-registered condition fired: `CORNER_GRADIENT_HURTS_LINE` was true on
seed 2 in PHASE 3, so E3 ran.  One frozen early trunk, two copies of the late
block -- one for the line branch, one for the corner branch -- both initialised
from the same A0 @18,000 weights so the two branches are bit-identical at step 0.

Seed 1, step 3,000:

```
arm                      angle  ang p90   offset     CIGM   direct   R med  R p90   t med   5cm5
E0_CONTINUE_LINE        2.3094   11.611   1.0866   1.8233       -        -      -       -      -
E1_SHARED_CORNER_LINE   2.3675   11.188   1.0822   1.8422   0.9570   11.191  73.01  0.2869  0.0605
E2_STOPGRAD_CORNER      2.3094   11.611   1.0866   1.8233   0.9342   12.041  73.21  0.3019  0.0625
E3_SPLIT_LATE           2.3094   11.611   1.0866   1.8233   0.7584    9.633  55.89  0.2515  0.0762
```

E3's line column is identical to E0's and E2's, for the same structural reason:
the line branch owns its late copy and the early trunk is frozen, so it receives
exactly E0's gradients.  Line preservation here is not a measurement that came
out well, it is a property of the wiring.

Against E2 -- the fair comparison, since both preserve the line exactly:

```
direct corner            +18.8%
PATH-C rotation median   +20.0%   (12.041 -> 9.633)
PATH-C rotation p90      +23.7%   (73.21  -> 55.89)
PATH-C translation       +16.7%   (0.3019 -> 0.2515)
5cm5deg                  +1.37 pp (0.0625 -> 0.0762)

E3_LOCALIZATION_GAIN   = True
E3_POSE_GEOMETRY_GAIN  = True
```

Both verdicts are reported separately on purpose.  The pose bottleneck is
systematic geometry rather than pixel error, so a corner median that improves
without the pose following would have been a localisation gain only and would
not have justified the extra late block.  Here the pose does follow.

Absolute values are not comparable to the 25k arms: these are 3,000-step
continuations with a fresh optimiser from a checkpoint whose belief stages had
never been trained, so every corner head starts at 21.7 cells.  The comparison is
internally valid and only internally.

One gap, recorded rather than papered over: `front_rear_shift`, isotropic scale
and the non-affine residual **cannot be computed for the E arms**, because those
continuation runners saved metrics but not weights and a median cannot be
un-medianed.  PATH-C rotation and translation stand in for the geometry question
and they answer it in the same direction.  Checkpoint saving has since been added
to both runners.

---

## theta / rho oracle -- which half of the line is the problem?

Same point branch, same solver, same Huber scale, same lambda fixed on D0.  Only
the line parameterisation changes.  Ground truth is a ceiling here and never a
method.

```
seed 1            R med   R p90    t med   5cm5deg  │ seed 2   R med    t med  5cm5deg
point only        7.830   41.75   0.2244   0.0781   │         8.067   0.2397   0.0938
O0 pred t, pred r 7.125   27.42   0.2407   0.0586   │         6.964   0.2951   0.0352
O1 pred t, GT r   6.339   26.45   0.2360   0.0723   │         6.076   0.2350   0.0977
O2 GT t, pred r   7.960   33.87   0.2742   0.0449   │         7.636   0.3146   0.0410
O3 GT t, GT r     4.550   17.91   0.1434   0.1914   │         4.688   0.1574   0.2012
```

The reading is unambiguous even though the gate is not.

* **rho is the problem.**  Replacing only rho with ground truth (O1) improves
  rotation *and* translation over O0 on both seeds: R 7.125 -> 6.339 and
  6.964 -> 6.076, t 0.2407 -> 0.2360 and 0.2951 -> 0.2350.
* **theta is already good.**  Replacing only theta (O2) makes both *worse* than
  O0 on both seeds.  There is nothing to gain from a better orientation.
* The ceiling is high: with both oracle, 5cm5deg roughly doubles against
  point-only (0.191 / 0.201 against 0.078 / 0.094).

The pre-registered flag nevertheless reads:

```
LINE_THETA_USEFUL_RHO_BIASED               False
LINE_THETA_IS_BOTTLENECK                   False
POINT_LINE_SCALE_INCONSISTENCY_SUSPECTED   True
```

`O1_recovers_translation` required O1 to claw back more than 5 points of the
translation that O0 lost.  Seed 2 recovered 25 points and passed; seed 1
recovered 2.1 points and failed -- because seed 1 had only lost 7.25 points to
begin with.  The gate is not adjusted; the mechanism is reported next to it.

## The follow-up that gate authorised: point-configuration scale

`POINT_LINE_SCALE_INCONSISTENCY_SUSPECTED` sends the investigation to the point
branch's isotropic scale, which PHASE 2 had independently measured at 4-5% small.
Rescaling the predicted corners about their own centroid to ground-truth size,
changing nothing else:

```
seed 1                R med    t med   5cm5deg  │ seed 2   R med    t med  5cm5deg
S0 point only         7.830   0.2244   0.0781   │         8.067   0.2397   0.0938
S1 point, GT scale    7.978   0.1475   0.1230   │         8.096   0.1596   0.1270
S2 joint point+line   7.125   0.2407   0.0586   │         6.964   0.2951   0.0352
S3 joint, GT scale    7.171   0.2044   0.0859   │         6.948   0.2246   0.0488
S4 point, const 1.029 7.817   0.2244   0.0859   │         8.089   0.2288   0.0469
S5 joint, const 1.029 7.129   0.2381   0.0723   │         6.943   0.2659   0.0859

SCALE_EXPLAINS_TRANSLATION_LOSS = True
CONSTANT_SCALE_IS_ENOUGH        = False
```

Two findings, and the second is the one that matters.

**Scale is the single largest lever on translation.**  Correcting it alone --
no lines involved -- improves translation by 34% and 33% and lifts 5cm5deg from
0.078 to 0.123 and from 0.094 to 0.127.  That is a bigger effect than anything
the line branch contributes anywhere in this study.  Once scale is corrected the
joint solver also beats point-only on both rotation and translation medians on
both seeds, which it never managed before.

**But a constant will not do it.**  The per-frame oracle ratio has a median of
1.031 / 1.033, which invites a fixed 1.03 correction.  Calibrating exactly that
constant on D0 and spending it once on D2 captures **0% and 13.6%** of the
oracle's translation gain, and on seed 2 it costs 4.7 points of 5cm5deg.  The
shrinkage is a per-frame, view-dependent quantity, not a fixed bias.  Testing it
was the difference between an actionable claim and a wrong one.

## Line uncertainty: magnitude yes, sign no

```
seed 1                          all lines   most-confident quartile
mean signed rho                  +0.9884            +1.0458
mean |rho|                        5.0250             3.6837
entropy vs |rho|                  +0.395
entropy vs signed rho             +0.049
roles with confident rho bias >= 0.10 cell        12 / 12
roles where entropy separates error                9 / 12

seed 2  mean signed rho +0.1151 -> +0.0921, |rho| 5.2494 -> 3.6334,
        entropy vs |rho| +0.360, entropy vs signed rho -0.055, 12/12 and 8/12

UNCERTAINTY_CANNOT_FIX_SYSTEMATIC_RHO_BIAS = True
UNCERTAINTY_WEIGHTING_ELIGIBLE             = True
```

Both are true and they are not in tension.  Entropy predicts how *large* a line
error is (+0.36 to +0.40) and says nothing about its *direction* (+0.05, -0.06).
Restricting to the most confident quartile cuts the mean absolute rho error by
27% and leaves the signed bias untouched -- on seed 1 it grows slightly.  Every
one of the twelve roles carries a signed offset bias that survives at high
confidence.

So confidence weighting is worth having as a tail suppressor and cannot be the
answer to the translation problem.  The per-role structure of the bias is worth
noting separately: the pooled signed bias differs between seeds (+0.99 against
+0.12 cell) while all twelve roles are individually biased on both, which means
the bias is per-role and partially cancels in the pool -- a per-role calibration
is a cheaper idea than a learned selector.

## Long confirmation -- E3 at 25,000 from scratch

The 3,000-step screen was a continuation from A0 @18,000 with a fresh optimiser,
so the candidate was re-run from scratch on the same 25,000-step budget as
A0/A1/A2 before anything is promoted.  Seed 1 is complete; seed 2 is running.

```
arm                     seed     angle    offset   CIGM cor   cornerC     R_C      t_C  5cm5deg_C
A0_LINE_ONLY               1    2.2051    0.9693     1.6364        -        -        -          -
A1_CORNER_LINE             1    2.2735    0.9604     1.5563   0.7418     7.83   0.2244     0.0781
A2_CORNER_LINE_MASK        1    2.3024    1.0399     1.7390   0.7427     8.22   0.2251     0.1074
E3_SPLIT_LATE              1    2.2051    0.9693     1.6364   0.6019     7.23   0.1825     0.1465
```

E3's three line numbers are A0's three line numbers, to four decimals, in a
from-scratch 25,000-step run.  The structural preservation seen in the short
screen is not an artefact of continuing from a shared checkpoint.

Against A1 at the same budget and seed: corner +18.86%, PATH-C rotation +7.63%,
translation +18.67%, and 5cm5deg from 0.0781 to **0.1465** -- nearly doubled, on
a metric that was 0.000 for every arm at 6,000 steps.

### E3 reduces the systematic geometry, not just the pixel error

This is the check the short screen could not run, because its runners saved no
weights.  Same residual audit as PHASE 2, same population, A1 against E3 at
25,000 (seed 1):

```
measure                    A1 @25k    E3 @25k     direction
front_area_ratio            0.7545     0.8433     toward 1.0
rear_area_ratio             0.8785     0.9449     toward 1.0
height_ratio                0.7822     0.9528     22% short -> 5% short
depth_ratio                 0.9675     0.9777     toward 1.0
width_ratio                 0.9569     0.9812     toward 1.0
affine_scale_isotropic      0.9523     0.9703     shrinkage roughly halved
centroid_shift              0.4966     0.4062     -18.2%
front_rear_shift            0.9543     0.8355     -12.5%
nonaffine_rms               0.6362     0.5594     -12.1%
PATH-C R median             7.83       7.23
```

All nine move the right way.  The two that matter most are `front_rear_shift`,
the dominant pose correlate, and `affine_scale_isotropic`, which the scale oracle
identified as the largest lever on translation -- E3 closes roughly half of that
shrinkage without any oracle.

**Mitigated, not solved.**  `front_rear_shift` still correlates with rotation
error at rho 0.647 and `SYSTEMATIC_CORNER_BIAS_SUPPORTED` is still True.  Giving
the corner branch its own late features reduces the systematic geometry; it does
not remove it, and the scale term remains the open lever.

Seed 1 only.  Two single-seed verdicts have already reversed in this study, so
nothing is promoted until seed 2 lands.

## Preserved, not rewritten

The following stand exactly as recorded and are not adjusted to fit anything
above:

```
A1 @25k          shared corner+line does not improve line-only, both seeds
A1 corner        own localisation excellent and stable, 0.8% seed spread
CIGM oracle      complementarity headroom 7-9%
off-grid         CIGM relative win rate rises, absolute median still worse
corner residual  shakes pose 2.5x harder than isotropic noise of equal size
A2 @25k          REJECT: offset -7.87%, CIGM -6.02%, corner -2.99% mean vs A1,
                 with offset worse on both seeds
```

One earlier statement of mine is corrected here: I reported A2's corner at 25k as
0.6877 cell.  That is the `D0_MH_SEEN512` value.  On the decision population
`D2_MH_DEV512` it is 0.7427 (seed 1) and 0.7793 (seed 2), and A2 is behind A1 on
average.
