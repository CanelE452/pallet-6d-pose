# LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL

Three times the optimizer exposure moved every number in the right direction and
did not come close to the gate.  The loss was still falling when it stopped.

```
DIRECT_HOUGH_TOKEN_XY_V0, 13,618 frames, fresh 0 -> 25,545 (15 passes)
decision at 25,545 on D2_LINE_DEV512, and only there

                 angle med   offset med   angle p90   offset p90   overall
@25,545 D2        3.735687     1.972937   27.843582     8.279793    FAIL 0/4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
```

Recomputed from the raw JSON, not read off a `PASS` field.

## The ladder

```
step     D0_SEEN512 (diagnostic)        D2_LINE_DEV512 (primary)          CE        slope/step
 1,703   9.7019/4.8385 p90 56.44/15.05  9.7985/5.3051 p90 58.52/15.93   8.252227   -4.882e-04
 5,000   5.3469/2.7777 p90 40.59/ 9.97  5.5276/2.9119 p90 34.12/ 9.39   7.244919   -2.015e-04
 8,515   4.3194/2.3224 p90 30.10/ 8.67  4.4309/2.4293 p90 30.90/ 8.70   6.845638   -1.013e-04
17,030   3.8075/2.0109 p90 29.33/ 7.95  3.9864/2.0692 p90 29.00/ 8.23   6.576459   -6.253e-05
25,545   3.4941/1.9265 p90 27.99/ 7.81  3.7357/1.9729 p90 27.84/ 8.28   6.436799   -4.942e-05
```

`D0` is diagnostic and entered no selection.  The seen/unseen gap stays at 5-6%
throughout, so this run never became a generalisation story.

## Why A4 and not A3

Both A3 and A4 are absolute failures.  What separates them is whether the run
had stopped learning, and it had not.

```
condition            pre-registered            observed              verdict
WEAK_IMPROVEMENT     both < 20%, 8515->25545   15.69% / 18.78%       true
GEOMETRY_PLATEAU     both < 5%, 17030->25545    6.29% /  4.65%       false
CE_PLATEAU           drop<0.02 and slope>=-1e-5 0.139660, -4.94e-05  false
CE_STRONG_DROP       drop >= 0.10               0.139660             true
```

A3 needs all three of weak, CE plateau and geometry plateau.  Two of them fail,
so the branch is A4: `LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL`.

Both near-misses are worth naming rather than rounding away.  Offset improved
4.65% in the last interval, just inside the 5% plateau line, while angle
improved 6.29%, just outside -- so by the pre-registered criterion the offset
axis is flattening and the angle axis is not.  And the last-pass CE slope of
-4.94e-05 is five times the plateau threshold but half of what it was at 8,515.
The trend across the run is unambiguous:

```
slope/step at   1,703      5,000      8,515     17,030     25,545
              -4.88e-04  -2.02e-04  -1.01e-04  -6.25e-05  -4.94e-05
```

Every doubling of exposure roughly halves the slope.  It is still descending,
and it is descending in a way that will not reach 1.0 degree by descending.

## Against the Q1 baseline

```
                        angle median   offset median
Q1_ROLE_QUERY_GLOBAL        4.179304       1.878792
TOKEN_XY @8,515             4.429565       2.445153     +5.99%  +30.14%  (worse)
TOKEN_XY @25,545            3.735687       1.972937    -10.61%   +5.01%
40% threshold               2.507582       1.127275
```

At 8,515 the line-native arm was worse than the image map on both axes.  At
25,545 it is 10.61% better on angle and still 5.01% worse on offset.  So part of
`DIRECT_HOUGH_TOKEN_XY_FULL_FAIL` -- the part that said this representation is
worse than the one it replaced -- **was an exposure artefact on the angle axis**.
The offset axis was not: three times the exposure did not recover the 30% deficit
to parity, it only closed it to 5%.

`REDUCTION_40` is false and was never close.

## Tails

```
@25,545 D2     angle > 5 deg    angle > 10 deg    offset > 2 cell     n
                  0.4077            0.2304            0.4938        5,921
@8,515 (this run's own start point, recorded run: 0.4614/0.2586/0.5736)
```

A quarter of all roles are still beyond 10 degrees after fifteen passes.  The
tail is where this fails, and exposure barely touched it: `angle_p90` moved 9.88%
across the whole span while the median moved 15.69%.

## Per role

```
role   n     angle 8,515 -> 17,030 -> 25,545      offset 8,515 -> 17,030 -> 25,545
  0   508      3.814     3.709     3.766            2.489     2.007     1.730
  1   468      3.491     3.036     2.783            3.404     2.730     2.564
  2   511      7.947     6.724     6.438            2.352     1.836     2.010
  3   449      2.745     2.649     2.670            2.601     2.603     2.386
  4   505      6.945     6.233     5.527            2.454     2.111     1.928
  5   488      4.556     3.998     3.915            2.385     2.020     1.772
  6   504      7.044     5.995     5.553            2.353     2.037     2.030
  7   505      7.964     6.810     5.938            2.315     1.881     2.043
  8   504      3.349     3.338     3.067            2.049     1.822     1.831
  9   490      2.380     2.340     2.095            2.605     2.325     2.273
 10   481      2.460     2.277     2.235            2.214     2.030     1.765
 11   508      3.489     3.122     3.054            2.160     1.871     1.719
```

The angle split recorded at 8,515 survives the whole schedule: roles 2, 4, 6 and
7 remain the worst and roles 9 and 10 the best, and exposure compressed the gap
without changing the ordering.  Roles 0 and 3 barely moved on angle at all
(3.814 to 3.766, 2.745 to 2.670).  This is recorded, not explained, and it
entered no selection.

## Reproduction at 8,515

The fresh trajectory passes through the recorded decision point.  Diagnostic
only -- no tolerance was invented and no block attached.

```
D2 @8,515          recorded      fresh        relative
angle median       4.4295654     4.4308929    +0.03%
angle p90         30.7176056    30.8966675    +0.58%
offset median      2.4451532     2.4292507    -0.65%
offset p90         8.8646450     8.6972351    -1.89%
train CE           6.8411159     6.8456376    +0.07%
qualification      FAIL          FAIL         same class

D0 @8,515          angle median +2.39%, offset median -0.74%, both same class
```

That is a closer agreement than the run's own parameters manage.  Under the
default kernels the locked runner differs from itself by 1.435e-03 across 21 of
28 tensors after twenty steps, yet at 8,515 the aggregate medians land within
0.03-1.89%.  The nondeterminism is real at the parameter level and does not
propagate to these statistics.  It says nothing about whether individual
components agree, which is a different question and is still open.

Structural parity of the re-composed loop was verified with the nondeterminism
removed: deterministic control 0.000e+00, locked against re-composed 0.000e+00.

## Standing

```
DIRECT_HOUGH_TOKEN_XY_FULL_FAIL @8,515      unchanged, 9e1ad5e
LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL  this run, 0/4 gates
FURTHER_STEP_EXTENSION                       FORBIDDEN
FEATURE_LIMIT_CONFIRMED                      False
FROZEN_A1_FEATURE_LIMIT_CONFIRMED            still not claimed
ROLE_ENCODER_LIMIT_CONFIRMED                 still not claimed
LINE_NATIVE_REPRESENTATION_FAIL              still not claimed
architecture status                          NOT_LOCKED
CIGM                                         BLOCKED
```

The exposure question is closed and the answer is no: exposure was not the
primary limit.  Fifteen passes buy 15.69% and 18.78% off the medians and leave
the model 3.7x and 3.9x away from the gate, with a slope that halves every time
the budget doubles.

But A4 is deliberately not A3.  The run was still learning when it stopped, so
this does not license "the frozen A1 feature is the limit" -- it only says the
step axis is no longer where the answer is.  Phase B may now open as an
architecture screen, and it opens as a question, not as a confirmation.

One claim from `9e1ad5e` needs narrowing.  "Worse than the image map it
replaced" held at 8,515 and does not hold at 25,545 on the angle axis, where the
line-native arm is now 10.61% ahead.  The offset axis is still behind.  The
recorded FULL result is not revised -- it was true at its own decision step --
but its comparison should not be carried forward as a property of the
representation.

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.  No PnP, no
CIGM, no dimensions, no MAP200.
