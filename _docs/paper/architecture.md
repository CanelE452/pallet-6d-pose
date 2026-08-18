# Architecture — why the corner branch needs a private pre-bottleneck pathway

Paper-facing write-up of the two-head design and the evidence for every choice in
it.  Numbers here are copied from result JSONs under
`data/pallet/results/paper_s2_multihead/`; the diagnostic narrative lives in
`_docs/audits/MULTIHEAD_FAILURE_DIAG.md` and is pointed at rather than repeated.

This file is canonical (see `_docs/paper/README.md`).  The older architecture
notes under `_docs/method/` and `_docs/models/` are from the v8 generation and
contradict this document; do not merge them in.

## The claim, in one sentence

> Corner estimation benefits from a private late pathway rooted **before** the
> line-specific 128-channel bottleneck; capacity added **after** that bottleneck
> does not reproduce the gain.

Two things are bundled in "private pathway rooted before the bottleneck", and the
experiments separate them from a third:

```
(1) corner and line want different late representations
(2) the corner branch needs the 256-channel early features, not the line
    branch's 128-channel output
(3) the corner branch simply needs more parameters
```

**(3) is measured and rejected.**  (1) and (2) are supported jointly and are *not*
separated from each other by the experiments run so far; the write-up says so
wherever the claim appears.

## The architecture

```
RGB 400x400
 │
 ├─ VGG19 features[0:19]                     FROZEN, 2,325,568 params
 │     └─ early feature   (B, 256, 50, 50)
 │
 ├─ late copy L  = vgg[19:27]                trainable, 5,014,912
 │     └─ F50_line       (B, 128, 50, 50)
 │           └─ 12 fixed-role queries -> Direct Hough -> 12 joint P(theta, rho)
 │
 └─ late copy C  = vgg[19:27]                trainable, 5,014,912
       └─ F50_corner     (B, 128, 50, 50)
             └─ DOPE belief stages 4-6 -> 9 channels (8 corners + centroid)

total trainable 22,661,532
```

Both late copies are initialised from the same weights, so at step 0 the two
branches are bit-identical and the line output equals the line-only baseline's.
No head is randomly initialised: the belief stages come from the checkpoint the
whole line stage already builds on (`weights/paper_s2/paper_s2_pdg/A1/epoch_003.pth`,
a `DopeNetwork(numSeg=1)` that already contains belief, affinity and seg heads).

### Line preservation is structural, not empirical

Because the line branch owns its late copy and the early trunk is frozen, the
line branch receives exactly the gradients it would receive with no corner head
at all.  Measured over 25,000 steps from scratch, two seeds:

```
                    angle     offset      CIGM
A0 line-only  s1   2.2051     0.9693    1.6364
E3 two-head   s1   2.2051     0.9693    1.6364
A0 line-only  s2   2.3360     1.0331    1.6680
E3 two-head   s2   2.3360     1.0331    1.6680
```

Identical to four decimals.  This is a property of the wiring, so it should be
stated as a guarantee rather than as a result that happened to come out well.

## Evidence for each design decision

### Why not fully shared (one late block, both losses)

A fully shared two-head model does **not** improve the line branch and slightly
degrades it, on every budget and seed measured:

```
A1 vs A0 line angle    25,000 steps   -3.10% (s1)   -4.93% (s2)
E1 vs E0 line angle     3,000 steps   -3.87% (s1)   -5.02% (s2)
```

Four of four negative.  The mechanism is **not** a directional gradient conflict:
cosine between the line and corner gradients on the shared block is positive at
every checkpoint on both seeds (+0.13 to +0.25, negative-batch fraction 0.03 to
0.14).  What the audit found instead is that the corner gradient reaching the
shared block collapses during training:

```
||g_corner|| / ||g_line||   step 0:  21.3      step 25,000:  5.9e-04
```

`lambda_corner` was fixed from the step-0 ratio, so by the end the corner loss
contributes about 2e-05 of the line gradient to the shared trunk.  A static
multitask weight calibrated on step-0 gradient norms measures the initialisation,
not the training.

### Why not stop-grad (shared late block, corner reads it detached)

Stop-grad preserves the line exactly -- and it is free, so it is the natural
baseline -- but its corner branch is clearly worse than a private late block:

```
E3 vs E2, 3,000 steps, paired frame bootstrap, 10,000 resamples
corner    +19.54%  CI [+15.67, +25.90]  P 1.000   (seed 1)
          +15.54%  CI [ +9.68, +19.29]  P 1.000   (seed 2)
PATH-C R  +20.00%  CI [+12.07, +27.32]  P 1.000
          +23.07%  CI [+11.45, +30.14]  P 0.999
line        0.00%  CI [ +0.00,  +0.00]
```

### Why the gain is not just capacity — the E4 control

E3 adds 5,014,912 trainable parameters over stop-grad.  E4 grants the same budget
but places it **after** the line bottleneck:

```
frozen early -> late L -> F50 -+-> line head              (identical to A0)
                               |
                               +-> detach -> capacity block -> corner head
```

The block is matched to 0.005% (5,015,168 against 5,014,912) and is a
zero-initialised residual, so at step 0 the corner head sees exactly what it sees
under stop-grad.  Verified before training: line output bit-identical to the
line-only arm; `L_line` gradient into the corner side exactly 0.000e+00 and
`L_corner` into the line side exactly 0.000e+00; twenty-step replay identical.

```
3,000 steps, two seeds, paired frame bootstrap

E4 vs E2 (capacity only)   corner   -1.24%  CI [-7.54, +3.83]  P 0.301
                                    -0.14%  CI [-7.06, +5.18]  P 0.419
E3 vs E4 (same budget,     corner  +20.53%  CI [+15.67, +25.90]  P 1.000
          different place)         +15.66%  CI [ +9.40, +21.57]  P 1.000
                           PATH-C R +13.83%  CI [ +7.08, +23.26]  P 1.000
                                    +14.73%  CI [ +5.35, +22.10]  P 0.999
```

Five million parameters after the bottleneck buy nothing.  The same five million
before it buy 16-21% corner accuracy and 14-15% rotation.  That is the sentence
the architecture is chosen on.

The geometry moves the same way -- E4 is worse than the fully-shared baseline on
the dominant pose-driving mode, while E3 is better than it:

```
front_rear_shift        A1 @25k  0.9543 / 0.9981
                        E3 @25k  0.8355 / 0.8544     better
                        E4 @ 3k  1.3865 / 1.4721     worse
affine_scale_isotropic  A1 @25k  0.9523 / 0.9631
                        E3 @25k  0.9703 / 0.9691     shrinkage roughly halved
                        E4 @ 3k  0.9462 / 0.9106     shrinkage deepened
```

(E4 is a 3,000-step arm and E3 a 25,000-step arm, so this comparison is
directional; the matched-budget comparison is in `CAPACITY_CONTROL_RESULT.md`.)

### What E3 actually improves

Against the fully-shared model at the same 25,000-step budget, two seeds:

```
corner localisation   +18.86% / +15.84%
PATH-C rotation       + 7.63% / + 6.55%
PATH-C translation    +18.67% / +19.02%
5cm5deg               + 6.84pp / + 4.29pp   (0.0781 -> 0.1465, 0.0938 -> 0.1367)
line                    0.000% (structural)
```

And the improvement is in the *systematic* geometry, not only the pixel error:

```
measure                  A1 @25k        E3 @25k
height_ratio          0.7822/0.9098   0.9528/0.8775
front_area_ratio      0.7545/0.8836   0.8433/0.8674
affine_scale_isotropic 0.9523/0.9631  0.9703/0.9691
centroid_shift        0.4966/0.4957   0.4062/0.4170
front_rear_shift      0.9543/0.9981   0.8355/0.8544
nonaffine_rms         0.6362/0.6320   0.5594/0.5265
```

This matters because the corner residual damages pose far more than its magnitude
suggests: a 1.0-cell corner error moves rotation by 14.0 degrees where isotropic
noise of the same size moves it by 5.6.  The bottleneck is the structure of the
error, not its size, so an architecture that reduces `front_rear_shift` and the
isotropic scale bias is doing something a smaller MSE would not.

## Limitations to state in the paper

1. **The mechanism is not fully separated.**  "Private pathway rooted before the
   bottleneck" bundles a task-specific representation with access to 256 early
   channels instead of 128.  E4 rules out capacity; it does not tell those two
   apart.  Do not write "late specialization" as if it were isolated.
2. **The 5.0M duplication is real cost.**  22.66M trainable against 17.65M.
   Justified here by +17% corner and +7% rotation at zero line cost, but it is a
   cost.
3. **Rotation p90 is the one unstable number.**  E3 against E4 gives +20.19% on
   seed 1 and -22.39% on seed 2.  Medians agree; the tail does not.
4. **Seed variance is large in the line branch and small in the corner branch**
   (15-19% against 0.5% at 6,000 steps).  Any line-side claim below about 20% at
   short budgets needs seed averaging; corner-side claims do not.
5. **Synthetic only.**  Everything above is `v2_prod40k_clean_merged`, whose
   elevation distribution (8% below 8 degrees) is nearly the inverse of the real
   captures (94%).  No real-transfer claim is made from these numbers.

## What was tried and rejected

Recorded so the paper can say the design space was searched rather than guessed.

```
fully-shared two-head (A1/E1)     line degrades, corner gains nothing
+ visible-mask auxiliary (A2)     REJECT: offset -7.87%, CIGM -6.02% vs A1,
                                  worse on both seeds; the mask head trains fine
                                  (IoU 0.605 -> 0.74) and simply does not help
stop-grad two-head (E2)           line free, corner clearly worse than E3
capacity after bottleneck (E4)    no corner gain at all
CIGM as the fusion path           direct corners beat CIGM corners 74-76% of the
                                  time; oracle-min headroom only 7-9%
native joint point+line solver    Huber-robust least squares over corner
                                  reprojection and line incidence together,
                                  lambda calibrated on the dev split.  Re-run on
                                  E3's own predictions rather than the shared
                                  model's, it improves rotation on both seeds
                                  (+8.2% / +5.3%) but only seed 1 keeps
                                  translation and 5cm5deg; seed 2 loses 9.9% and
                                  6.8pp.  Two-of-two was the pre-registered bar,
                                  so this is not adopted.  The defect is
                                  identified: lambda is selected on rotation
                                  median alone, which is the axis the line term
                                  is good at, so the selection over-weights the
                                  lines and translation pays.  Recorded rather
                                  than patched, because the threshold must not be
                                  changed after seeing the result.
dense vector voting, sparse edge, affinity association, centre-conditioned
offset, side-face anchors         all previously measured negative, see
                                  `pvnet-dense-vector-voting-negative-result`
```

## What the line branch contributes to pose: orientation, and only orientation

The two heads are justified above by corner accuracy and by the pose the corners
drive.  The reverse direction -- feeding the line predictions back into the pose
-- was tested three times, and the three attempts separate cleanly.

The full `(theta, rho)` constraint qualified neither time it was tried:

```
                                     seed 1                 seed 2
native joint point+line             R +8.2  t +5.0  pass   R +5.3  t  -9.9  fail
same, on scale-corrected corners    R +5.7  t +6.1  pass   R +4.3  t  -9.3  fail
```

The third attempt removed `rho` from the pose objective and kept the line's
orientation.  This is not a new head or a new prediction -- the existing joint
residual puts both projected endpoints of an edge on the predicted line, and the
half-difference of those two values is algebraically free of the offset:

```
(da + db)/2  = offset of the edge midpoint      carries rho
(da - db)/2  = (L/2) * sin(delta)               rho cancels exactly
```

Rotation then improves everywhere, by two to four times the full-line margin:

```
                        ALL rotation              V<8 rotation
seed 1  dev            +23.98%  CI [+16.1,+30.2]  +35.90%  CI [+26.2,+47.6]
seed 1  confirmation   +24.56%  CI [+14.5,+33.1]  +39.33%  CI [+29.2,+49.0]
seed 2  dev            +15.76%  CI [+10.5,+21.4]  +22.34%  CI [+14.3,+31.6]
seed 2  confirmation   +16.30%  CI [+11.0,+21.5]  +18.90%  CI [+12.0,+28.1]
```

Twenty subset-by-seed-by-population combinations, and the confidence interval
excludes zero in all of them.  Translation is a different story: its interval
contains zero in seven of eight, so the orientation term neither helps nor
provably harms it.  And `rho` is confirmed as what was doing the damage -- on
seed 2 the full-line solver halves 5cm5deg (0.1367 to 0.0684) while the
orientation-only solver raises it (0.1367 to 0.1504), with the confirmation
population reproducing both.

The pre-registered two-seed gate still fails, because seed 1 loses 3.9% of
translation and 1.4pp of 5cm5deg.  That failure is traced, not mysterious: the
selection rule was given a safety filter on translation but still picked the
*smallest rotation* among the survivors, and rotation falls monotonically with
the weight, so seed 1 was pushed to the edge of the locked grid.  Seed 2's
grid-edge candidate was caught by the filter and it passed on both populations.
Re-choosing the weight after seeing that would no longer be a pre-registered
test, so `THETA_ONLY_LINE_USEFUL` stands at false and re-selection needs its own
registration.

So the honest claim for the paper is narrower than "the line branch improves
pose" and much narrower than "the line branch is useless": **the line branch
carries orientation information that transfers to rotation robustly, largest
where corners are truncated, and its offset channel is what has repeatedly
damaged translation.**  Full numbers:
`data/pallet/results/paper_s2_multihead/THETA_ONLY_SOLVER_RESULT.md`.

## Open axis, not an architecture question

The largest remaining lever on translation is the corner configuration's
**per-frame isotropic scale**: restoring it to ground truth improves translation
by 31-33% and adds 3.3-3.5pp of 5cm5deg, more than any line formulation
contributes anywhere in this study.

That lever is now known to be **not recoverable from the network's own outputs**.
Ridge regression of the per-frame factor, fitted on the seen split and read once
on dev, reaches R^2 0.13-0.17 against a pre-registered bar of 0.30, and which
feature block wins flips between seeds.  Applying the prediction makes pose worse:
every corrected arm loses 5cm5deg against no correction at all (-7.0pp, -4.3pp),
and a constant factor loses too.

The mechanism is simple and worth stating in the paper.  Translation from PnP is
nearly proportional to the corner configuration's scale, so a multiplicative
correction trades bias for variance.  Two heads' worth of training has already
pushed the median bias down to about 2% (E3 halves A1's 4-5% shrinkage), so an
R^2 = 0.13 estimate injects more variance than the bias it removes.  The oracle
works because it is exact, not because correction is the right shape of fix.

So the next step is not a better scale predictor and not another backbone.  It is
a representation and loss that keep the scale from drifting in the first place.
Full numbers: `data/pallet/results/paper_s2_multihead/FINAL_2HEAD_POSE_QUALIFICATION.md`.
