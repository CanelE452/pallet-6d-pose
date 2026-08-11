# F50_ADAPTER_INSUFFICIENT

A 17,601-parameter adapter on a frozen backbone gets 17-21% where unfreezing
5,014,912 backbone parameters got 45%.  It also does not specialize.

```
D2_LINE_DEV512 @25,545, the decision step and the only one

                 angle med   offset med   angle p90   offset p90   gates
F2 adapter        3.084991     1.554134   17.584229     6.242823   0 of 4
F0 frozen         3.735687     1.972937   27.843582     8.279793   0 of 4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
F2 vs F0           +17.42%      +21.23%     +36.85%      +24.60%
40% required on both medians                              not met
```

Recomputed from the raw JSON.  `F0` is Phase A's frozen run at full precision
and is the only baseline that decides anything.

```
ABSOLUTE_PASS   False
REDUCTION_40    False    17.42% and 21.23% against 40%
->  F50_ADAPTER_INSUFFICIENT   (C3)
```

## The ladder

```
step     D0_SEEN512                    D2_LINE_DEV512                CE        alpha    relL2    cos
 1,703   9.4527/5.1153 p90 62.96/14.55 10.2486/5.2875 p90 61.41/15.78 8.274255 +0.00685 0.02104 0.999773
 5,000   4.9934/2.5619 p90 33.98/ 9.01  4.8882/2.5544 p90 28.69/ 8.54 7.065881 +0.06212 0.19488 0.980748
 8,515   3.6443/1.8893 p90 25.58/ 7.33  3.6727/1.9238 p90 24.64/ 7.44 6.540333 +0.07340 0.23244 0.972703
17,030   3.1683/1.5437 p90 18.88/ 6.10  3.3022/1.5929 p90 19.33/ 6.38 6.190232 +0.07495 0.26676 0.964224
25,545   2.9535/1.4435 p90 16.56/ 5.71  3.0850/1.5541 p90 17.58/ 6.24 6.006698 +0.07404 0.29731 0.955988
```

At 1,703 the adapter is still nearly off and F2 sits slightly *behind* F0
(10.2486 against 9.7985), which is what zero-init predicts -- alpha starts at
zero and has to grow.  By 8,515 F2 has passed F0's *final* numbers, five passes
against fifteen.  Then it flattens: the last five passes buy 6.6% and 2.4% while
the cross-entropy keeps falling.  No instability, no NaN; `finite` is true at
every mark.

## The adapter was used, and it saturated in the gate but not in the body

```
alpha    0.00685 -> 0.06212 -> 0.07340 -> 0.07495 -> 0.07404
relL2    0.02104 -> 0.19488 -> 0.23244 -> 0.26676 -> 0.29731
cosine   0.999773 -> 0.980748 -> 0.972703 -> 0.964224 -> 0.955988
```

The scalar gate stops moving after 8,515 -- it even ticks down at the end -- while
the relative change to F50 keeps climbing to 29.7% and the cosine keeps falling.
Reading alpha alone would have said "saturated at 8,515"; what actually happened
is that the body's convolutions kept growing under a fixed gate.  The adapter
parameter delta norm at 25,545 is 21.63.

So this is not a case of a component that failed to engage.  It engaged, changed
F50 by nearly a third, and that bought 17-21%.

## It did not specialize

```
D2/D0 ratio at 25,545      angle    offset
F2 adapter                 1.0445   1.0766
F0 frozen                  1.0691   1.0241
F1 late-A1 unfreeze        1.4255   1.2994
```

F2 sits with F0, not with F1.  The 42.5% seen/unseen gap that came with
unfreezing the backbone does not appear when the same supervision goes through a
small adapter, at any mark -- the F2 ratio stays between 0.98 and 1.08 the whole
way.  `SPECIALIZES` is false: D0 did not improve while D2 stalled; both moved
together.

This is a diagnostic and is reported as one.  A small gap here is not evidence
that generalization is solved -- F2's absolute numbers are worse than F1's on
every statistic, and a model that has learned less has less to specialize with.

## Against F1, for context only

```
                  F1 late-A1     F2 adapter     F2 vs F1
angle median        2.070244       3.084991      -49.02%
offset median       1.077348       1.554134      -44.26%
angle p90           9.702057      17.584229      -81.24%
offset p90          4.125696       6.242823      -51.32%
F50_ADAPTER_PARETO_BETTER_THAN_LATE_UNFREEZE   False
```

F1 dominates on all four.  F1 selected nothing here -- the reduction test reads
F0 and an AST test enforces that -- but the comparison is the point of the
screen and it is one-sided.

## Tails

```
@25,545 D2      angle > 5 deg   angle > 10 deg   offset > 2 cell
F2                 0.3420           0.1689           0.4042
F0                 0.4077           0.2304           0.4938
F1                 0.2108           0.0974           0.2631
```

## Per role

```
role   n      F0 ang  F2 ang  F1 ang      F0 off  F2 off  F1 off
  0   508      3.766   2.993   2.065       1.730   1.491   0.861
  1   468      2.783   2.477   1.848       2.564   2.043   1.473
  2   511      6.438   4.524   2.490       2.010   1.526   1.273
  3   449      2.670   2.595   2.039       2.386   1.901   1.356
  4   505      5.527   4.419   2.938       1.928   1.530   1.015
  5   488      3.915   3.224   2.243       1.772   1.489   0.941
  6   504      5.553   4.068   2.785       2.030   1.570   0.951
  7   505      5.938   4.808   2.449       2.043   1.500   1.324
  8   504      3.067   2.721   1.887       1.831   1.184   0.807
  9   490      2.095   1.617   1.482       2.273   1.511   1.396
 10   481      2.235   2.046   1.533       1.765   1.774   1.150
 11   508      3.054   2.839   1.896       1.719   1.258   0.771
```

```
group                    F0 mean   F2 mean   F1 mean
high-angle 2/4/6/7         5.864     4.455     2.665
lower-angle 3/9/10         2.334     2.086     1.684
```

The split that has survived every screen survives this one.  The adapter
compresses it about a third as much as the unfreeze does: the high-angle group
falls 24% under F2 and 55% under F1, while the low group falls 11% and 28%.
Role 10's offset is the single case that got slightly worse than F0 (1.765 to
1.774).  Recorded, entering no selection.

## What this settles

```
established     a constrained frozen-backbone adapter improves every D2
                statistic over F0 -- 17.42%, 21.23%, 36.85%, 24.60% -- and does
                not reproduce the specialization that came with unfreezing
established     it does not reach 40% on either median, so it does not preserve
                the late-A1 signal at this size and shape
not established that a broader unfreeze is necessary.  One adapter shape, one
                bottleneck, one learning rate, no sweep of any of them
not established that adapter capacity is the binding constraint: alpha stopped
                growing at 8,515 while the body kept growing, which is not the
                signature of a component out of room
```

The late-A1 signal stands as recorded.  What this adds is that the gain does not
transfer to 17,601 parameters at this shape, and that the specialization which
accompanied the gain is not intrinsic to line supervision -- it came with the
5M-parameter unfreeze.

## Standing

```
DECISION                          F50_ADAPTER_INSUFFICIENT
SPECIALIZES                       False
PARETO_BETTER_THAN_LATE_UNFREEZE  False
architecture status               NOT_LOCKED
CIGM                              BLOCKED
NEXT                              ROLE_ENCODER_DEPTH_SCREEN, not implemented
```

`LINE_STAGE_ARCHITECTURE_LOCKED` requires a task-and-safety pass, a
same-protocol replicate in the same class, a fixed role-derangement causal pass,
and the whole 2,393-frame LINE_DEV.  None is met.  `ARCHITECTURE_COMPLETE` is
not written.  CIGM, PnP, known dimensions and K-based pose evaluation stay
blocked.

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.
