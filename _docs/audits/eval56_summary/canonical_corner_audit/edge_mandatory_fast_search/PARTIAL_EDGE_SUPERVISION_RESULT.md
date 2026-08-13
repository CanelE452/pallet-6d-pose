# PARTIAL_EDGE_SUPERVISION_NOT_USEFUL

The emitted decision string is `PARTIAL_EDGE_SUPERVISION_NOT_USEFUL`, and it does
not describe what happened.  That is the first thing to say, because the primary
condition passed and passed widely.

```
T1_PARTIAL @25,545, n = 3,737, the decision step and the only one

                    angle med   offset med   angle p90   offset p90   both-gate
B1  T1 supervised    2.391045     1.544678   14.063293     6.083636      5.62%
B0  T1 excluded      3.182896     1.891941   17.356171     7.708471      3.72%
B1 vs B0              +24.88%      +18.35%     +18.97%      +21.08%     +1.90pp
```

Recomputed from `partial_supervision_result.json`.  Both arms sit at
25,545 steps, seed 1, sigma 1.5, split sha `70ba7f1e`.

## The pre-registered gates, and the branch that had no rule

```
A  both T1 medians >= 15% better      +24.88% / +18.35%     pass
B  both-gate >= +10 points            +1.90pp               fail
C  T0 degradation <= 10%              angle -8.29%, offset +11.20%   fail
   both medians worse                 no
```

The rule set registered four branches: A+B+C is USEFUL, A+B with C failing is
TRADEOFF, A failing is NOT_USEFUL, both medians worse is HURTS.  The state
observed -- A passing, B failing -- matches none of them, and the runner's `else`
fell through to NOT_USEFUL.  The label stands, because gates are not edited after
results are read.  But the label is the fallback, not a registered verdict on
this state, and reading it as "partial-edge supervision did nothing" would be
wrong.

That gap is mine.  Condition B asked for +10 points of both-gate on a population
whose absolute both-gate is 3.72% under one arm and 5.62% under the other; the
task gate is 1.0 deg AND 0.5 cell together, which nothing in this line stage has
ever come close to.  A +10pp bar on a 3.7% base was not reachable by any outcome
short of solving the stage, so B was never going to discriminate here.  It is
recorded as a defect in the pre-registration rather than repaired after the fact.

## What supervising the partial edges actually did

Every direction agrees, which is what makes it worth stating despite the label.

```
T1_PARTIAL tails      B0        B1
angle > 5 deg       34.44%    26.84%
angle > 10 deg      16.72%    13.09%
offset > 2 cell     47.82%    39.23%
angle <= 1 deg      18.30%    24.27%
offset <= 0.5 cell  14.91%    17.21%
```

Per role on T1, B1 is better on angle in 12 of 12 and on offset in 11 of 12.
The one loss is role 7's offset, -6.62%.

```
role     n     B0 ang  B1 ang   d ang     B0 off  B1 off   d off
   0    520    2.7481  1.8829  +31.48     2.0589  1.4102  +31.51
   1    134    1.5017  1.3838   +7.85     2.2328  1.9802  +11.31
   2    408    4.0147  3.1699  +21.04     2.0802  1.8829   +9.49
   3    128    1.8959  1.5319  +19.20     2.6361  2.0942  +20.56
   4    437    4.6870  3.5161  +24.98     1.8215  1.6723   +8.19
   5    538    3.0902  2.2186  +28.21     2.2816  1.5715  +31.12
   6    449    4.2176  3.3637  +20.25     1.8895  1.5078  +20.20
   7    406    3.8392  2.9016  +24.42     1.7383  1.8534   -6.62
   8    300    2.8179  2.0300  +27.96     1.4748  1.1994  +18.68
   9    102    1.4760  1.3876   +5.99     1.7320  1.1207  +35.30
  10     84    1.2922  1.0122  +21.66     1.5048  1.1672  +22.43
  11    231    2.7986  2.0850  +25.50     1.4041  1.1862  +15.51
```

## Span bins, fixed before the numbers were read

```
chord/span    n      B0 ang  B1 ang    B0 off  B1 off    B0 gate  B1 gate
0.00-0.25    302     4.4951  3.7457    3.4834  2.0500      3.64%    4.97%
0.25-0.50    520     3.7202  2.6742    2.5898  1.5052      4.42%    7.69%
0.50-0.75    967     3.5677  2.4727    2.0783  1.5369      2.28%    4.24%
0.75-1.00   1948     2.8244  2.1782    1.5783  1.4815      4.26%    5.85%
```

The offset gain grows as less of the edge remains in frame -- 41.1% in the
0.00-0.25 bin against 6.1% in 0.75-1.00 -- which is the ordering the mechanism
would predict, since the bins where most of the edge is visible are the ones B0
already almost had.  These bins are diagnostic and were not allowed to move a
gate.

```
endpoints out    n      B0 ang  B1 ang    B0 off  B1 off    B0 gate  B1 gate
one            3689     3.1536  2.3880    1.8736  1.5453      3.77%    5.64%
both             48     4.8530  2.7461    4.4905  1.2822      0.00%    4.17%
```

The both-out row is n = 48 and is reported for completeness, not as evidence.

## The cost, which is real and is on offset

```
T0_FULL @25,545, n = 23,936      B0        B1        B1 vs B0
angle median                   2.013660  1.846683    +8.29%
offset median                  0.861287  0.957695   -11.20%
angle p90                      8.974348  8.542320    +4.81%
offset p90                     3.509748  3.581718    -2.05%
both-gate                        11.51%    10.94%    -0.57pp
```

Per role on T0, B1 is better on angle in 9 of 12 and worse on offset in 11 of 12.
So this is not noise on a couple of roles: supervising the truncated edges moves
the whole model's offset calibration on the full edges by roughly a tenth, and
buys angle with it.  Condition C is failed by 1.20 points on one axis of two.

The two are not separable here.  One factor moved, and it changed both
populations; nothing in this screen says which part of the T0 offset shift is the
extra supervision and which is the different loss population.

## What the arms were

```
B1   P0_AUG_ONLY reused, support = seg["hit"]
     sha b37dfc2617a7e0f0fb13f1c549f2377459ba9b20477caacedc2d68a6fec2e43c
B0   trained fresh, support = seg["in_frame_full"]
     sha 3eab4f750dffae6a551b9842f992804a240433c822db4581c5d744514dd51138
```

Reuse was qualified by measurement before it was used.  Initialisation matched at
0.000e+00 on the decoder and the late A1 with identical optimizer groups; frames,
both views, geometry, support and category matched exactly on probe batches with
the supervised loss agreeing at 0.000e+00; and under deterministic kernels a
control run of the original screen against itself was exact, with this runner's
B1 also exact against it over twenty steps.

The mask was proven to isolate the factor it claims.  On a batch of 71 T0, 20 T1
and 5 T2 roles: T0 loss identical at 0.000e+00, T1 contributing 0.000e+00 under
B0 against 202.170258 under B1, T2 zero in both, the T1-only late-A1 gradient
0.000e+00 under B0 against 7.169e-04 under B1, and the T0-only gradient matching
to 0.000e+00 relative.

No rescaling was applied to compensate B0's smaller supervised set, because that
would have been a second factor.  Supervised-role exposure is recorded instead:
B0 accumulated 2,050,245 supported roles over the run.

## B0's ladder

```
step      D2 angle  D2 offset   p90      L_sup      roles sup
 1,703      6.0385    5.0483   51.252   7.899219      136,683
 5,000      3.2516    1.9807   18.500   6.487561      401,440
 8,515      2.7610    1.4116   14.497   5.987545      683,415
17,030      2.2081    1.0271   10.437   5.401453    1,366,830
25,545      2.1401    0.9869   10.124   5.140960    2,050,245
```

Finite throughout, monotone, no best-step selection anywhere.

## What this settles

```
established     supervising partially in-frame structural edges improves
                prediction on geometrically truncated edges substantially:
                +24.88% angle median, +18.35% offset median, better on angle in
                12 of 12 roles, all three tails shorter
established     the gain grows as less of the edge remains in frame, 41.1%
                offset in the 0.00-0.25 span bin against 6.1% in 0.75-1.00
established     it costs 11.20% of the T0 offset median while improving the T0
                angle median by 8.29%, and this shows in 11 of 12 roles
established     it does not move the strict both-gate meaningfully, +1.90pp
not established anything about occlusion.  Every edge here is truncated by the
                image border and the categories come from segment clipping
not established that the T0 offset shift is caused by the added supervision
                rather than by the changed loss population.  One factor, two
                effects, not separated
```

## Standing

The pre-registered decision label is `PARTIAL_EDGE_SUPERVISION_NOT_USEFUL`.  The
observed A-pass / B-fail state means this label must not be interpreted as "no
effect": the causal result is that supervising the partial edges improves the
T1 medians by 24.88% on angle and 18.35% on offset, while the absolute line gate
still fails and a T0 offset trade-off is present.

```
TRUNCATION_SIGNAL_PRESENT                       True
PARTIAL_EDGE_SUPERVISION_CAUSALLY_IMPROVES_T1_MEDIANS   True
ABSOLUTE_LINE_GATE_FAIL                         True
T0_OFFSET_TRADEOFF_PRESENT                      True
OCCLUSION_NOT_EVALUATED                         True
ARCHITECTURE_NOT_LOCKED                         True
CIGM_BLOCKED                                    True
SEALED_UNCHANGED                                True
```

```
DECISION                          PARTIAL_EDGE_SUPERVISION_NOT_USEFUL (fallback branch)
A_both_medians_15pct_better       True
B_gate_plus_10_points             False
C_T0_safety                       False
PRE_REGISTRATION_BRANCH_GAP       True  (A pass / B fail had no registered rule)
SEMANTICS                         PARTIAL_IN_FRAME_STRUCTURAL_EDGE
EXTERNAL_OCCLUSION_LEARNABILITY   False
SELF_OCCLUSION_LEARNABILITY       False
HIDDEN_CORNER_RECOVERY            False
CIGM_RECOVERY                     False
POSE_IMPROVEMENT                  False
OCCLUSION_LABEL_SOURCE            MISSING
OCCLUSION_LEARNABILITY            NOT_EVALUATED
architecture status               NOT_LOCKED
replicate / role shuffle          BLOCKED
whole LINE_DEV                    BLOCKED
CIGM / PnP / dimensions / K-pose  BLOCKED
```

## Sealed

`validation512`, `untouched`, `eval56`, `wood45` and final-test remain unopened.
