# REGULARIZED_LATE_A1_OVERCONSTRAINED

Anchoring closes 74% and 48% of F1's seen/unseen gap and costs 26% and 13% of
its accuracy.  The gap condition passes, the accuracy condition does not, so the
branch is G4.

```
D2_LINE_DEV512 @25,545, the decision step and the only one

                 angle med   offset med   angle p90   offset p90   gates
S1 F1+L2-SP       2.615707     1.213144   12.913452     5.090273   0 of 4
F1 historical     2.070244     1.077348    9.702057     4.125696   0 of 4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
```

Recomputed from the raw JSON.  `lambda_sp = 150.3369063067523`, locked from a
train-only gradient balance and never touched after.

```
G3 needs BOTH                              observed
  degradation vs F1  <= +10%               angle +26.35%   offset +12.60%   MISSED
  gap closure        >= 20%                angle  73.96%   offset  48.39%   met
->  REGULARIZED_LATE_A1_OVERCONSTRAINED   (G4)
```

## The ladder

```
step     D0            D2            p90            task      R_SP      l*R_SP   dW/W    F50    cos     D2/D0
 1,703  6.2304/4.4017 6.6519/4.7551 47.33/14.33   7.912153  0.001460  0.219440  0.0385  0.7116 0.8054 1.068/1.080
 5,000  3.2730/1.9144 3.4913/1.9643 18.16/ 6.97   6.432412  0.002737  0.411480  0.0517  0.9187 0.7013 1.067/1.026
 8,515  2.7871/1.4737 2.9888/1.5381 14.30/ 5.91   6.050255  0.003497  0.525677  0.0574  0.9217 0.6798 1.072/1.044
17,030  2.5009/1.2045 2.7447/1.3356 12.92/ 5.30   5.713382  0.004291  0.645090  0.0635  0.9234 0.6387 1.098/1.109
25,545  2.3548/1.0508 2.6157/1.2131 12.91/ 5.09   5.581613  0.004659  0.700482  0.0666  0.9303 0.6168 1.111/1.155
```

Finite throughout, no instability.

## The mechanism did what it was built to do

```
weight drift ||W-W0||/||W0||   0.0385 -> 0.0666    F1 is unconstrained
F50 drift                      0.7116 -> 0.9303    flat from 5,000 onward
F50 cosine                     0.8054 -> 0.6168    still falling
SP share of the total loss     2.7% -> 10.1%
```

The penalty grows as the weights move and pulls back, which is why the drift
curve flattens while the cosine keeps turning: the weights keep rotating at
roughly fixed distance.  Late weights ended 6.66% from their pretrained values.

For contrast, the low-rank arm at the same step reached an F50 drift of 3.9068
with cosine 0.3724 -- four times the feature movement.  These arms differ in how
far the features are allowed to travel, and the ordering of their seen/unseen
gaps follows that: 1.2336 for the loosest internal arm, 1.1108 here, 1.4255 for
free adaptation.

## Where it sits among the arms

```
                  angle med   offset med   angle p90   offset p90   D2/D0 angle
F1 late unfreeze   2.070244     1.077348    9.702057     4.125696      1.4255
L1 low-rank        2.450502     1.114699   11.359642     4.747742      1.2336
S1 F1 + L2-SP      2.615707     1.213144   12.913452     5.090273      1.1108
R1 role depth      2.909729     1.230865   18.133209     5.730148      1.0461
F2 adapter         3.084991     1.554134   17.584229     6.242823      1.0445
F0 frozen          3.735687     1.972937   27.843582     8.279793      1.0691
```

Read down the accuracy column and up the ratio column: every arm that gets
closer to F1's accuracy carries more of F1's specialization.  S1 lands between
L1 and the outside-the-extractor arms on both.  That is an observation about six
runs, not a law, and nothing here isolates a cause.

## Tails

```
@25,545 D2      angle > 5 deg   angle > 10 deg   offset > 2 cell
S1                 0.2799           0.1321           0.3143
F1                 0.2108           0.0974           0.2631
L1                 0.2699           0.1172           0.2944
```

## Per role

```
role   n      F1 ang  S1 ang  L1 ang      F1 off  S1 off  L1 off
  0   508      2.065   2.586   2.535       0.861   1.243   1.122
  1   468      1.848   2.047   2.179       1.473   1.641   1.427
  2   511      2.490   3.818   3.231       1.273   1.297   1.060
  3   449      2.039   2.325   2.087       1.356   1.397   1.438
  4   505      2.938   3.612   3.522       1.015   1.154   1.109
  5   488      2.243   2.654   2.711       0.941   1.244   1.172
  6   504      2.785   3.369   3.297       0.951   1.160   0.969
  7   505      2.449   3.539   3.045       1.324   1.106   1.038
  8   504      1.887   2.358   2.017       0.807   1.025   0.997
  9   490      1.482   1.592   1.482       1.396   1.168   1.121
 10   481      1.533   1.685   1.701       1.150   1.182   1.221
 11   508      1.896   2.318   2.350       0.771   1.046   0.975
```

```
group             F1      S1      L1
high 2/4/6/7     2.665   3.584   3.274
lower 3/9/10     1.684   1.867   1.757
```

No role beats F1 on angle; roles 7 and 9 beat it on offset, and both are roles
where F1's offset was unusually poor.  The cost is concentrated in the
high-angle group, which loses 34.5% against the low group's 10.9% -- the same
four roles that every screen in this program has struggled with.

## What this settles

```
established     under this coefficient, explicit anchoring closes 73.96% and
                48.39% of F1's seen/unseen gap
established     it costs 26.35% and 12.60% of F1's median accuracy, so the
                accuracy condition fails and the branch is G4
established     the penalty engaged and held: weight drift ended at 6.66% and
                the F50 drift curve flattened after 5,000
not established that lambda is too large in general.  One coefficient, fixed by
                a gradient balance at one state, never swept
not established that weight drift causes specialization.  What is shown is that
                explicit anchoring moves the accuracy/generalization tradeoff
not established anything about other anchoring formulations, other penalty
                shapes, or other layers
```

The phrase "lambda was too large" is not written.  The screen fixed a unit and
measured what that unit buys; it did not search for a better one, and a G4 under
one calibration is not evidence about the family.

## Standing

```
DECISION                          REGULARIZED_LATE_A1_OVERCONSTRAINED
lambda_sp                         150.3369063067523, locked, not swept
conditions                        GAP_CLOSED true, ACCURACY_PRESERVED false
architecture status               NOT_LOCKED
replicate                         BLOCKED (G1 only)
role shuffle                      BLOCKED
whole LINE_DEV                    BLOCKED
CIGM / PnP / dimensions / K-pose  BLOCKED
```

`LINE_STAGE_ARCHITECTURE_LOCKED` still requires a task-and-safety pass first,
and no arm in this program has produced one.  Nothing is called
`ARCHITECTURE_COMPLETE`.

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.
