# LATE_A1_LOW_RANK_INSUFFICIENT

113,664 parameters get 34.40% and 43.50% where 5,014,912 got 44.58% and 45.39%.
One axis clears the threshold and the other does not, so the verdict is E3 --
and the run specializes on the way there.

```
D2_LINE_DEV512 @25,545, the decision step and the only one

                 angle med   offset med   angle p90   offset p90   gates
L1 low-rank       2.450502     1.114699   11.359642     4.747742   0 of 4
F0 frozen         3.735687     1.972937   27.843582     8.279793   0 of 4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
L1 vs F0          +34.40%      +43.50%     +59.20%      +42.66%
40% threshold     2.241412     1.183762    -            -
                     MISSED       cleared
```

Recomputed from the raw JSON.  F0 is Phase A at full precision and is the only
baseline that decides anything.

```
ABSOLUTE_PASS   False
REDUCTION_40    False    both medians are required; angle missed by 4.4 points
->  LATE_A1_LOW_RANK_INSUFFICIENT   (E3)
LOW_RANK_A1_SPECIALIZES   True      (E4 diagnostic, decision unchanged)
```

## What this does not say

```
"low-rank adaptation cannot work"      forbidden, and not written
"full unfreeze is required"            forbidden, and not written
LOW_RANK_ADAPTATION_REFUTED            False, recorded in the JSON
FULL_UNFREEZE_PROVEN_REQUIRED          False, recorded in the JSON
F1_SIGNAL                              unchanged
scope                                  rank 8, one formulation, one LR
```

## The ladder

```
step     D0_SEEN512              D2_LINE_DEV512           CE        drift   cos     dW/W range
 1,703   7.5026/4.8296           7.9907/5.0584          8.032967  1.3867  0.5788  0.294-0.439
 5,000   3.1850/2.0492           3.4564/2.1463          6.479651  2.0716  0.4640  0.528-0.690
 8,515   2.4158/1.3190           2.7881/1.5146          5.888951  2.8137  0.4202  0.710-0.889
17,030   2.0361/1.0172           2.4362/1.2210          5.394779  3.2360  0.3846  1.001-1.187
25,545   1.9865/0.9074           2.4505/1.1147          5.165383  3.9068  0.3724  1.163-1.352
```

Finite at every mark, no instability.  L1 passed F0's final numbers before 8,515
and passed F2's and R1's finals too.

## Low rank does not mean small

This is the correction the run forces.

```
effective kernel change, ||dW|| / ||W0||, at 25,545
[19] 1.2539   [21] 1.1628   [23] 1.3521   [25] 1.1916
F50 relative L2 drift 3.9068, cosine against the frozen F50 0.3724
```

The delta kernels end up **larger than the frozen kernels they correct**, and
F50 moves nearly four times its own norm.  Rank 8 constrains `dW` to a rank-8
subspace; it does not constrain its magnitude.  So this arm did not test "a
small amount of adaptation" -- it tested adaptation restricted to a low-rank
structure, with 2.2665% of F1's parameters and more than F1's freedom of
magnitude.

That matters for reading everything below.  The smaller D0/D2 gap cannot be
explained by the delta being gentle, and a null on the medians cannot be
explained by the branch being starved.

The branch stayed foldable throughout, which was checked before the run at
2.980e-06 against a 1e-5 tolerance: a 1x1 followed by a k x k with nothing
between is a k x k, so inference could merge these into the frozen kernels.
Diagnostic; it selected nothing.

## It specializes

```
D0 angle   F0 3.494102 -> L1 1.986481   +43.15%   (>= 40%)
D2 angle   F0 3.735687 -> L1 2.450502   +34.40%   (<  40%)
->  LOW_RANK_A1_SPECIALIZES = True
```

The seen population cleared the threshold and the unseen one did not.  That is
the E4 diagnostic exactly, and the hard decision stays E3.

```
D2/D0 ratio at 25,545      angle    offset
L1 low-rank                1.2336   1.2284
F1 full late unfreeze      1.4255   1.2994
F2 post-F50 adapter        1.0445   1.0766
R1 F2 + deeper decoder     1.0461   1.0283
```

L1 sits between the two families, closer to F1.  Its gap opened steadily -- 6.5%,
8.5%, 15.4%, 19.6%, 23.4% on angle -- tracking F1's shape at roughly half the
magnitude.  The arms that adapt *inside* the late convolutions specialize; the
arms that adapt outside them do not.  Two points on each side is not a law, and
nothing here isolates why.

## Against the other arms, context only

```
                 angle med   offset med   angle p90   offset p90   D2/D0 angle
F1 late unfreeze  2.070244     1.077348    9.702057     4.125696      1.4255
L1 low-rank       2.450502     1.114699   11.359642     4.747742      1.2336
R1 role depth     2.909729     1.230865   18.133209     5.730148      1.0461
F2 adapter        3.084991     1.554134   17.584229     6.242823      1.0445
```

```
L1 vs F1   angle -18.37%   offset -3.47%   p90 -17.08% / -15.08%
PARETO_BETTER_THAN_FULL_UNFREEZE   False
```

F1 is still ahead on all four, so there is no Pareto claim.  But the offset
median is within 3.47% of F1 at 2.27% of the parameters, and L1 beats both
outside-the-extractor arms on every statistic.  These selected nothing; an AST
test bars them from the reduction test and every decision branch.

## The last five passes

```
17,030 -> 25,545     angle median  2.436192 -> 2.450502   -0.59%   worse
                     offset median 1.220999 -> 1.114699   +8.71%
                     angle p90    11.911049 -> 11.359642  +4.63%
                     offset p90    5.281483 ->  4.747742  +10.11%
```

The angle median crossed and came back: 2.4362 at 17,030 against 2.4505 at
25,545.  It was 8.7% short of the threshold at the earlier mark and 9.3% short
at the decision step, so the miss does not hinge on which of the two was taken --
but the decision step was fixed in advance and is what is reported.  Meanwhile
the cross-entropy was still falling at -6.949e-05.

## Tails

```
@25,545 D2      angle > 5 deg   angle > 10 deg   offset > 2 cell
L1                 0.2699           0.1172           0.2944
F1                 0.2108           0.0974           0.2631
R1                 0.3285           0.1670           0.3383
F2                 0.3420           0.1689           0.4042
F0                 0.4077           0.2304           0.4938
```

## Per role

```
role   n      F0 ang  L1 ang  F1 ang      F0 off  L1 off  F1 off
  0   508      3.766   2.535   2.065       1.730   1.122   0.861
  1   468      2.783   2.179   1.848       2.564   1.427   1.473
  2   511      6.438   3.231   2.490       2.010   1.060   1.273
  3   449      2.670   2.087   2.039       2.386   1.438   1.356
  4   505      5.527   3.522   2.938       1.928   1.109   1.015
  5   488      3.915   2.711   2.243       1.772   1.172   0.941
  6   504      5.553   3.297   2.785       2.030   0.969   0.951
  7   505      5.938   3.045   2.449       2.043   1.038   1.324
  8   504      3.067   2.017   1.887       1.831   0.997   0.807
  9   490      2.095   1.482   1.482       2.273   1.121   1.396
 10   481      2.235   1.701   1.533       1.765   1.221   1.150
 11   508      3.054   2.350   1.896       1.719   0.975   0.771
```

```
group             F0      L1      F1
high 2/4/6/7     5.864   3.274   2.665
lower 3/9/10     2.334   1.757   1.684
```

All twelve improve on both axes -- unlike R1, where two high-angle roles got
worse.  The high group falls 44.2% against F1's 54.6%; the low group falls 24.7%
against 27.9%.  Roles 9 and 2 and 7 beat F1 on at least one axis.  The split
still survives, and this arm compresses it more than either outside-the-extractor
arm did.

## What this settles

```
established     low-rank adaptation inside the late convolutions reaches 34.40%
                and 43.50% against F0, beating both outside arms on every
                statistic, at 2.2665% of F1's trainable parameters
established     it misses the 40% threshold on the angle median
established     it specializes: D0 cleared 40% while D2 did not
established     low rank did not mean small: ||dW||/||W0|| ended above 1 on all
                four convolutions
not established that low-rank adaptation cannot work.  Rank 8, one formulation,
                one learning rate, no sweep of any of them
not established that a full unfreeze is required.  F1's signal is unchanged and
                this run does not bear on whether it is necessary
not established that the branch ran out of capacity or that it had too much
```

## Standing

```
DECISION                            LATE_A1_LOW_RANK_INSUFFICIENT
LOW_RANK_A1_SPECIALIZES             True (diagnostic)
LOW_RANK_CONV_MERGEABLE             True (diagnostic)
PARETO_BETTER_THAN_FULL_UNFREEZE    False
architecture status                 NOT_LOCKED
replicate                           BLOCKED (E1 only)
role shuffle                        BLOCKED
whole LINE_DEV                      BLOCKED
CIGM / PnP / dimensions / K-pose    BLOCKED
NEXT                                consider REGULARIZED_LATE_A1_FULL_ADAPTATION
                                    in a separate lock
```

Not implemented, not scaffolded, not filed here.

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.
