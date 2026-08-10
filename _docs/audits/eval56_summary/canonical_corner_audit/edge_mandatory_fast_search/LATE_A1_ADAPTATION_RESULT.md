# LATE_A1_FEATURE_ADAPTATION_SIGNAL

Letting line supervision reach `net.vgg[19:27]` halves every dev statistic and
still fails every gate.

```
D2_LINE_DEV512 @25,545, the decision step and the only one

                 angle med   offset med   angle p90   offset p90   overall
F1                2.070244     1.077348    9.702057     4.125696    FAIL 0/4
F0                3.735687     1.972937   27.843582     8.279793    FAIL 0/4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
F1 vs F0           +44.58%      +45.39%     +65.16%      +50.17%
```

Recomputed from the raw JSON.  `F0` is the Phase A run reused after its code
path was proven bit-identical under deterministic kernels.

## The label

```
ABSOLUTE_PASS   False    0 of 4 gates
REDUCTION_40    True     both medians clear the Q1 threshold
                         angle  2.070244 <= 2.507582   (Q1 4.179304, -50.46%)
                         offset 1.077348 <= 1.127275   (Q1 1.878792, -42.66%)
OVERFITS        False    axis 1 holds, axis 2 does not
SIMILAR_TO_F0   False
->  LATE_A1_FEATURE_ADAPTATION_SIGNAL
```

This is the first arm in the entire line-stage program to clear the 40%
threshold on both axes.  Every earlier screen -- the strip refiner, the weighted
moment decoders, the image maps, direct Hough with a frozen encoder -- failed it.

It is still 2.1x and 2.2x from the task gate and 4.9x and 4.1x from safety.

## The ladder

```
step     D0_SEEN512 (diagnostic)        D2_LINE_DEV512 (primary)         CE       slope/step
 1,703   4.8801/4.0854 p90 32.60/12.86  5.1868/4.2923 p90 36.36/13.76  7.590148  -9.730e-04
 5,000   2.2855/1.4539 p90 10.45/ 4.71  2.5746/1.6321 p90 12.88/ 5.53  5.913029  -3.214e-04
 8,515   1.8171/1.0490 p90  6.94/ 3.56  2.1699/1.2540 p90 11.08/ 4.92  5.259740  -1.619e-04
17,030   1.5262/0.7777 p90  5.43/ 2.56  2.0885/1.0079 p90  9.64/ 4.15  4.720402  -5.215e-05
25,545   1.4523/0.8291 p90  4.50/ 2.36  2.0702/1.0773 p90  9.70/ 4.13  4.696517  +4.478e-05
```

F1 passed F0's *final* D2 at somewhere before 5,000 steps.  Three passes of
adaptation beat fifteen passes of frozen training on both medians and both p90s.
That is the finding.

## The run turned over before the decision step

```
F1, 17,030 -> 25,545        best mark
angle median   2.088455 -> 2.070244   +0.87%
offset median  1.007887 -> 1.077348   -6.89%   <- best offset was at 17,030
angle p90      9.638733 -> 9.702057   -0.66%
offset p90     4.147220 -> 4.125696   +0.52%
CE             4.720402 -> 4.696517   drop 0.023885, slope now +4.478e-05
```

The cross-entropy slope crossed zero and three of the four D2 statistics moved
the wrong way in the last five passes.  D0's offset degraded too, 0.7777 to
0.8291, so this is the whole run settling into a noise floor under a constant
learning rate, not a train/dev divergence.

**This does not change the label, and it was not allowed to.**  The
pre-registered `OVERFITS` test compares F1 against F0 -- "F1 train CE below F0's
while either D2 median is above F0's" -- and by that test axis 1 holds and axis 2
does not, because F1's D2 is far better than F0's on both axes.  So the
condition is false and the label stands.

But the pre-registration was written to catch adaptation overfitting *relative to
the frozen arm*, and it does not see a run that peaks against itself.  That is a
gap in the screen, stated as one rather than patched after the fact.  The 25,545
decision was fixed in advance precisely so that "the 17,030 checkpoint was
better" could not become a result, and it does not become one here.  It is
recorded as an observation for whoever designs the next screen.

## Seen against unseen

```
step     D0 angle   D2 angle   gap        F0's gap at the same step
 1,703    4.8801     5.1868     6.3%      1.0%
 5,000    2.2855     2.5746    12.6%      3.4%
 8,515    1.8171     2.1699    19.4%      2.6%
17,030    1.5262     2.0885    36.8%      4.7%
25,545    1.4523     2.0702    42.5%      6.9%
```

Frozen training held the gap at 1-7% because only the head could move.  Unfreezing
5,014,912 parameters opens it to 42.5%.  Both curves still fall, so this is not
the overfitting the pre-registration was looking for, but it is the clearest
signal in the run about where the next constraint sits.

## Tails

```
@25,545 D2      angle > 5 deg   angle > 10 deg   offset > 2 cell
F1                 0.2108           0.0974           0.2631
F0                 0.4077           0.2304           0.4938
```

The tail halves.  Roughly one role in ten is still beyond 10 degrees.

## Per role

```
role   n     F0 angle   F1 angle      F0 offset   F1 offset
  0   508      3.766      2.065          1.730       0.861
  1   468      2.783      1.848          2.564       1.473
  2   511      6.438      2.490          2.010       1.273
  3   449      2.670      2.039          2.386       1.356
  4   505      5.527      2.938          1.928       1.015
  5   488      3.915      2.243          1.772       0.941
  6   504      5.553      2.785          2.030       0.951
  7   505      5.938      2.449          2.043       1.324
  8   504      3.067      1.887          1.831       0.807
  9   490      2.095      1.482          2.273       1.396
 10   481      2.235      1.533          1.765       1.150
 11   508      3.054      1.896          1.719       0.771
```

Every role improves on both axes.  The angle spread collapses from 2.095-6.438
to 1.482-2.938: the roles that were worst under a frozen encoder -- 2, 4, 6, 7 --
gain the most, which is what a feature-side explanation of that split would
predict, though this run does not test that explanation.  Roles 9 and 10 remain
best.  Recorded, entering no selection.

## What this settles

```
established     line supervision reaching net.vgg[19:27] improves every D2
                statistic by 45-65% over the identical frozen arm, at equal
                budget, with one factor moved
established     the frozen A1 feature was a real constraint on this
                architecture -- not the only one
not established that it is the primary remaining constraint: F1 still fails
                all four gates and its own last five passes went backwards
not established anything about A1 adaptation in general -- this is
                LATE_A1_BLOCK19_26, 68.3% of net.vgg and 9.15% of A1
not established anything about other A1 learning rates: one value, CAP.LR x 0.1,
                pre-registered and not swept, which is a limitation of the
                screen and is not being chased
```

Phase A said exposure was not the primary limit.  Phase B says the frozen
feature was *a* limit worth 45%, and that removing it is not sufficient.

## Standing

```
DECISION                LATE_A1_FEATURE_ADAPTATION_SIGNAL
NEXT                    ROLE_ENCODER_CAPACITY_SCREEN, not implemented
architecture status     NOT_LOCKED
CIGM                    BLOCKED
widen the unfreeze      not licensed by this result
```

`LINE_STAGE_ARCHITECTURE_LOCKED` needs four things and this screen delivered
none of them, because it did not pass:

```
1  line-stage D2 task and safety PASS        not met
2  same-protocol replicate, same class       not run
3  fixed role-shuffle causal PASS            not run
4  whole LINE_DEV, 2,393 frames              not run
```

Nothing here is called `ARCHITECTURE_COMPLETE`.  The integration screen -- twelve
lines to CIGM to eight corners to known-dimension PnP to 6D pose -- stays closed.

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.  No PnP, no
CIGM, no dimensions, no MAP200.
