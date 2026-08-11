# ROLE_ENCODER_DEPTH_INSUFFICIENT

One more role-conditioned block rewrites the descriptors more than completely
and buys 5.68% of angle.

```
D2_LINE_DEV512 @25,545, the decision step and the only one

                 angle med   offset med   angle p90   offset p90   gates
R1 +one block     2.909729     1.230865   18.133209     5.730148   0 of 4
R0 = F2           3.084991     1.554134   17.584229     6.242823   0 of 4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
R1 vs R0           +5.68%      +20.80%      -3.12%       +8.21%
40% required on both medians                              not met
```

Recomputed from the raw JSON.  R0 is F2 at full precision and is the only
baseline that decides anything.

```
ABSOLUTE_PASS   False
REDUCTION_40    False    5.68% and 20.80% against 40%
->  ROLE_ENCODER_DEPTH_INSUFFICIENT   (D3)
```

The angle p90 is 3.12% **worse** than R0 -- the one statistic that moved
backwards.

## What this does not say

```
"role encoder capacity is not the bottleneck"     forbidden, and not written
ROLE_ENCODER_CAPACITY_EXONERATED                  False, recorded in the JSON
scope                                             one extra block at this width
                                                  and head count only
```

One recipe failed: a single refinement block at dim 64, four heads, a 128-wide
FFN, at the shared learning rate, with no sweep of any of those.  That is what
was tested and that is the extent of the claim.

## The ladder

```
step     D0_SEEN512                   D2_LINE_DEV512               CE        beta      dchg     cos
 1,703   9.4257/4.3904 p90 56.1/13.0  9.9299/4.5180 p90 55.3/13.4  8.142158  -0.17844  0.69409  0.8899
 5,000   4.5625/2.1673 p90 33.2/ 7.9  4.5287/2.1558 p90 32.2/ 8.1  6.868146  -0.35489  0.99279  0.8081
 8,515   3.5804/1.7974 p90 25.0/ 6.8  3.6834/1.8090 p90 23.3/ 7.2  6.384469  -0.41546  1.09381  0.7776
17,030   3.1510/1.4122 p90 21.3/ 6.3  3.1970/1.4386 p90 19.0/ 6.4  6.004488  -0.43162  1.38556  0.7095
25,545   2.7814/1.1970 p90 16.7/ 5.4  2.9097/1.2309 p90 18.1/ 5.7  5.792634  -0.42030  1.59989  0.6623
```

Finite at every mark, no instability.

## The block was used, hard, and it did not convert

```
beta     -0.17844 -> -0.35489 -> -0.41546 -> -0.43162 -> -0.42030
dchg      0.69409 ->  0.99279 ->  1.09381 ->  1.38556 ->  1.59989
cosine     0.8899 ->   0.8081 ->   0.7776 ->   0.7095 ->   0.6623
```

By the end the block's contribution is 1.6x the norm of the descriptor it
refines, and the cosine between the descriptor entering and leaving is 0.66.
This is not a component that failed to engage.  It rewrote the representation
and returned 5.68% on angle.

`beta`'s sign is not interpretable -- the gate and the block's output can absorb
a sign between them -- so only its magnitude is read, and that magnitude stops
moving after 8,515 (-0.41546 to -0.42030, and it ticks back at the end) while
`dchg` keeps climbing from 1.09 to 1.60.  Exactly the shape the F50 adapter
showed: the scalar gate plateaus while the body keeps growing.  Reading the gate
alone would have called both components saturated when neither was.

Attention entropy sat between 5.76 and 6.24 against ln(2500) = 7.82 for a
uniform distribution over the tokens, and the FFN output norm ratio between 3.28
and 4.26.  Both are recorded.  Nothing here supports a statement about where
attention looks, and none is made.

## Seen against unseen

```
D2/D0 at 25,545      angle    offset
R1 +one block        1.0461   1.0283
R0 = F2              1.0445   1.0766
F1 late-A1           1.4255   1.2994
```

R1 stays with R0, well away from the broad-unfreeze arm, at every mark.
`SPECIALIZES` is false: D0 and D2 moved together.  Diagnostic only -- a ratio
near one between two populations is not a generalization result, and R1 is worse
than F1 on all four statistics.

## Against F1, for context only

```
                  F1 late-A1     R1            R1 vs F1
angle median        2.070244      2.909729      -40.55%
offset median       1.077348      1.230865      -14.25%
angle p90           9.702057     18.133209      -86.90%
offset p90          4.125696      5.730148      -38.89%
PARETO_BETTER_THAN_LATE_UNFREEZE   False
```

F1 still dominates on all four.  It selected nothing here; an AST test bars it
from the reduction test and from every decision branch.

## Tails

```
@25,545 D2      angle > 5 deg   angle > 10 deg   offset > 2 cell
R1                 0.3285           0.1670           0.3383
R0 = F2            0.3420           0.1689           0.4042
F1 late-A1         0.2108           0.0974           0.2631
```

The offset tail is the one that moved, 0.4042 to 0.3383.  The angle tails barely
move, which is the same story the medians tell.

## Per role

```
role   n      R0 ang  R1 ang  F1 ang      R0 off  R1 off  F1 off
  0   508      2.993   2.736   2.065       1.491   1.177   0.861
  1   468      2.477   2.107   1.848       2.043   1.678   1.473
  2   511      4.524   4.664   2.490       1.526   1.285   1.273
  3   449      2.595   2.267   2.039       1.901   1.674   1.356
  4   505      4.419   4.240   2.938       1.530   1.333   1.015
  5   488      3.224   3.046   2.243       1.489   1.196   0.941
  6   504      4.068   4.234   2.785       1.570   1.230   0.951
  7   505      4.808   4.435   2.449       1.500   1.164   1.324
  8   504      2.721   2.529   1.887       1.184   1.037   0.807
  9   490      1.617   1.512   1.482       1.511   1.237   1.396
 10   481      2.046   1.833   1.533       1.774   1.180   1.150
 11   508      2.839   2.564   1.896       1.258   1.024   0.771
```

```
group             R0      R1      F1
high 2/4/6/7     4.455   4.393   2.665
lower 3/9/10     2.086   1.871   1.684
```

Offset improves for all twelve.  Angle does not: roles 2 and 4.664 against 4.524
and 6 at 4.234 against 4.068 got worse, and both are in the high-angle group.
The high group as a whole moves 1.4% while the low group moves 10.3%, so the
extra block helped where the problem already was smallest.  The split that has
survived every screen in this program survives this one too, and the arm that
compresses it remains the broad unfreeze.

## What this settles

```
established     an extra role-conditioned nonlocal block engages fully -- 1.6x
                descriptor rewrite, cosine 0.66 -- and returns 5.68% angle and
                20.80% offset against a 40% threshold
established     it does not specialize; D2/D0 stays with R0 throughout
established     the gain is almost entirely on the offset axis, and two
                high-angle roles got worse
not established that role-encoder capacity is not the bottleneck.  One block at
                one width, one head count, one learning rate
not established that the block ran out of room: beta plateaued while the body
                kept growing, which is not the signature of exhausted capacity
```

Three screens have now shown the same shape.  A component is added, it engages
strongly by its own diagnostics -- the adapter changed F50 by 30%, this block
rewrites descriptors by 160% -- the training cross-entropy improves, and the dev
medians move by well under half of what is needed.  That pattern is recorded
here; explaining it is not this screen's job.

## Standing

```
DECISION                          ROLE_ENCODER_DEPTH_INSUFFICIENT
SPECIALIZES                       False
PARETO_BETTER_THAN_LATE_UNFREEZE  False
architecture status               NOT_LOCKED
replicate                         BLOCKED (D1 only)
role shuffle                      BLOCKED
whole LINE_DEV                    BLOCKED
CIGM / PnP / dimensions / K-pose  BLOCKED
NEXT                              a separate lock decides between broader
                                  feature adaptation, joint regularization, or
                                  a different structural decoder
```

None of the three is implemented, scaffolded or filed here.

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.
