# What Phase A and Phase B are allowed to be called

Neither result changes.  `ddc77e3` and `58379ca` stand exactly as committed.
This narrows how they may be cited, because two sentences that are easy to write
from them are not supported by them.

## Standing

```
Phase A   LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL
          LOCKED_3X_EXPOSURE_DID_NOT_RESCUE
          OPTIMIZATION_REMAINS_ACTIVE

Phase B   LATE_A1_FEATURE_ADAPTATION_SIGNAL
```

## Withdrawn and forbidden

```
"optimizer exposure is not a bottleneck"
"frozen A1 explains 45% of the error"
FROZEN_A1_FEATURE_LIMIT_CONFIRMED
ROLE_ENCODER_CAPACITY_LIMIT_CONFIRMED
```

The first overreaches Phase A.  What Phase A established is that a *locked* 3x
exposure -- 8,515 to 25,545 steps -- did not rescue the task, while the
cross-entropy was still falling at the decision step and the last-pass slope was
five times the plateau threshold.  Optimization remained active.  "Exposure is
not a bottleneck" is a claim about the whole axis; the measurement covers one
tripling.

The second overreaches Phase B.  What Phase B established is that late-A1
adaptation bought roughly 45% off both D2 medians relative to F0 under one
protocol, one block and one learning rate.  A measured improvement from removing
a constraint is not an error decomposition, and 45% of the *remaining* error is
not attributable to the frozen feature by that arithmetic.

The two `_CONFIRMED` labels were never earned; Phase A's verdict is A4 precisely
because it is not A3, and no role-encoder screen has been run at all.

## What Phase B actually recorded

```
                    F0 frozen     F1 late-A1     change
angle median         3.735687      2.070244      +44.58%
offset median        1.972937      1.077348      +45.39%
angle p90           27.843582      9.702057      +65.16%
offset p90           8.279793      4.125696      +50.17%
gates                0 of 4        0 of 4
REDUCTION_40         false         true
```

Against that gain, three things:

```
D0/D2 gap        1-7% frozen  ->  42.5% adapted
final interval   offset median 1.007887 -> 1.077348, worse; CE slope crossed to
                 +4.478e-05 at 17,030 -> 25,545
safety           p90 9.702057 and 4.125696 against 2.0 and 1.0 -- 4.9x and 4.1x
```

So the gain is real, is the first in the line-stage program to clear the 40%
threshold on both axes, and arrives together with a large specialization and a
run that peaked before its own decision step.

## The next question

```
CAN_CONSTRAINED_LINE_SPECIFIC_FEATURE_ADAPTATION
KEEP_THE_GAIN_WITHOUT_UNFREEZING_A1 ?
```

Not a capacity question.  Before asking whether the role encoder is too small,
ask whether 5,014,912 unconstrained backbone parameters were more adaptation
than the task needed.  A1 goes back to fully frozen and a single zero-init
residual adapter sits on F50.  One factor: `CONSTRAINED_F50_LINE_ADAPTER`.

`ROLE_ENCODER_DEPTH_SCREEN` is recorded as the next candidate after this one and
is not implemented, not scaffolded, and not filed.  Feature adaptation and
encoder depth do not move in the same run.
