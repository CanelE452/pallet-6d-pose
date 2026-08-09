# DIRECT_HOUGH_TOKEN_XY_FULL_FAIL

The line-native representation memorises 32 frames better than anything before
it and generalises worse than the image map it replaced.

```
DIRECT_HOUGH_TOKEN_XY_V0, FULL LINE_TRAIN, decision at 8,515 on D2

                 angle med   offset med   angle p90   offset p90   overall
@8,515 D2         4.429565     2.445153   30.717606     8.864645    FAIL 0/4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
```

Every gate fails.  Against the Q1 image-map baseline at full precision the
result is **worse on both metrics**:

```
                    angle median   offset median
Q1_ROLE_QUERY_GLOBAL    4.179304       1.878792
DIRECT_HOUGH_TOKEN_XY   4.429565       2.445153
change                    +5.99%        +30.14%

40% threshold        <= 2.507582    <= 1.127275
```

```
ABSOLUTE_PASS   False
REDUCTION_40    False
DECISION        DIRECT_HOUGH_ROLE_HEATMAP_FAIL  (branch A: below 40%)
                -> DIRECT_HOUGH_TOKEN_XY_FULL_FAIL
```

Per the locked protocol: no replicate, no role shuffle, no whole-LINE_DEV
promotion, **CIGM blocked**.  The runner skipped the shuffle on its own -- the
result file carries no `shuffle` key.

## Trajectories

```
step     D0_SEEN512                       D2_LINE_DEV512                    CE
1,250   11.1139 / 5.4098  p90 61.35/15.59  11.3232 / 5.5618 p90 61.46/16.81  8.43693
2,500    7.7093 / 4.4669  p90 48.51/13.19   8.0217 / 4.7464 p90 50.98/13.99  7.97747
5,000    5.2020 / 2.7727  p90 39.31/ 9.73   5.4482 / 2.8880 p90 33.24/ 9.47  7.22405
8,515    4.2185 / 2.3396  p90 31.36/ 8.73   4.4296 / 2.4452 p90 30.72/ 8.86  6.84112
```

D0 and D2 differ by 5.0% on angle at the decision step, so this is not a
generalisation gap -- it is underfitting at FULL scale, with the cross-entropy
still falling (8.44 to 6.84).  The same architecture reached 0.338 degree on 32
frames at 6,000 steps.

## The tail is the story

```
@8,515 D2      frac angle > 5 deg   frac angle > 10 deg   frac offset > 2 cell
                     0.4614               0.2586                0.5736
```

Nearly half the roles are beyond 5 degrees and a quarter beyond 10.  The p90 is
30.7 degrees against the image map's 36.6 -- both are far outside anything
usable, and the median advantage the map held is not the interesting part.  What
this representation did not do is produce a confident, correct peak on unseen
frames.

## Per role

```
role   angle med   offset med      role   angle med   offset med
  0      4.0423      2.6108          6      7.0766      2.4773
  1      3.4910      3.2304          7      7.9182      2.4739
  2      8.2468      2.4649          8      3.4431      1.9814
  3      2.7224      2.4952          9      2.4243      2.5948
  4      7.0217      2.4093         10      2.4713      2.2316
  5      4.2392      2.5438         11      3.5885      1.9271
```

The offset is nearly uniform at 1.93-3.23 across all twelve, which is a different
shape from the map family where roles 1 and 3 stood out.  Angle splits the roles
in two: 2, 4, 6, 7 sit at 7-8 degrees while 3, 9, 10 sit at 2.4-2.7.  That
pattern is recorded, not explained.

## What this does and does not close

```
established   with token XY conditioning, a line-native fixed-role Hough output
              does not beat the image-space map it replaced on LINE_DEV
not established  that line-native representation is the wrong idea -- this arm
              never carried the map family's additive AbsoluteXY branch
```

`DirectHoughModel` builds `AbsoluteXY` and never calls it, recorded in
`5040aab`.  Position still reaches the model through the role-query tokens, so
the arm is not position-blind, but the specific mechanism worth 20.1% in the map
family was absent from every number above.  That is why this is scoped as
`DIRECT_HOUGH_TOKEN_XY_V0` and not as "direct Hough with the map's XY factor".

The four oracles remain valid and unaffected: `O_DOMAIN`, `O_GRID`, `O_TARGET`
and `O_SCORER` all passed, and none of them uses `DirectHoughModel`.  So the
lattice, the target, the wiring and the bilinear scorer are all cleared, and
what failed is the encoder-to-descriptor path at FULL scale.

## Standing

```
DIRECT_HOUGH_TOKEN_XY_FULL_FAIL        0/4 gates, worse than Q1 on both
replicate / shuffle / promotion        not run, per protocol
CIGM                                   BLOCKED
overfit extension                      DIRECT_HOUGH_OVERFIT_EXTENDED_PASS stands
next question                          FROZEN_A1_FEATURE_OR_ENCODER_LIMIT
```

The overfit result and the FULL result together point one way: the head can fit
line space to 0.34 degree when it is allowed to memorise, and cannot get near
that from the frozen A1 descriptor on unseen frames.  Nothing is built here.

No PnP, no CIGM, no dimensions, no MAP200.  `untouched`, `eval56`, `wood45` and
final-test remain unopened.
