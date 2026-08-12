# APPEARANCE_CONSISTENCY_OVERREGULARIZES

The consistency term does what it was built to do -- two photometric views of a
frame disagree 65% less -- and costs 15.73% of the angle median to do it.

```
D2_LINE_DEV512 @25,545, the decision step and the only one

                 angle med   offset med   angle p90   offset p90   gates
P1 +consistency   2.195404     1.128651    9.187943     4.310326   0 of 4
P0 aug only       1.896942     1.065037    8.956047     4.029266   0 of 4
gate              <= 1.0       <= 0.5      <= 2.0       <= 1.0
P1 vs P0           -15.73%      -5.97%      -2.59%       -6.98%
```

Recomputed from the raw JSON.  `lambda_cons = 11.10883257173405`, locked at
`fed1aff` from a train-only gradient balance and untouched after.

```
H2  both medians >= +20%                 -15.73% / -5.97%     no
H3  accuracy within +-10% AND closure >= 20%
      accuracy    angle -15.73%                                no
      closure     angle 42.91%, offset 41.71%                  yes
H4  accuracy worse than 10%              angle -15.73%         yes
->  APPEARANCE_CONSISTENCY_OVERREGULARIZES
```

H3 fails on one leg only.  The gap closure it asks for is met on both axes and
comfortably -- 42.91% and 41.71% against a 20% bar -- and the accuracy band is
missed on angle by 5.7 points.  That is the shape of this result: the
generalization half of the hypothesis held and the accuracy half did not.

## The ladder

```
P0_AUG_ONLY
step     D0            D2            p90           L_sup     view JS   top-bin  D2/D0
 1,703  5.2039/4.4816 5.4846/4.5703 45.40/14.52  7.905792  0.053509  0.2899  1.0539/1.0198
 5,000  2.9668/1.9859 3.1396/2.0944 20.17/ 7.08  6.552926  0.107352  0.1276  1.0582/1.0546
 8,515  2.2132/1.4300 2.4002/1.4470 11.53/ 5.05  5.998059  0.117540  0.1552  1.0845/1.0119
17,030  1.7732/1.1265 1.9220/1.2041  8.99/ 3.97  5.443563  0.131326  0.1504  1.0839/1.0689
25,545  1.5505/0.9277 1.8969/1.0650  8.96/ 4.03  5.173691  0.134904  0.1572  1.2234/1.1480

P1_AUG_CONSISTENCY                                          lambda*L_cons
 1,703  5.7915/4.7503 6.4356/4.9317 50.34/14.36  8.041548  0.132417  0.023648  0.3669  1.1112/1.0382
 5,000  3.4237/2.3245 3.5185/2.3745 18.90/ 8.37  6.752628  0.352273  0.035199  0.2672  1.0277/1.0215
 8,515  2.5821/1.5550 2.7430/1.6132 13.31/ 5.48  6.254851  0.408353  0.044789  0.2434  1.0623/1.0374
17,030  2.1362/1.1606 2.3412/1.2170 10.14/ 4.67  5.770994  0.421069  0.050723  0.2311  1.0959/1.0486
25,545  1.9470/1.0390 2.1954/1.1287  9.19/ 4.31  5.569079  0.443327  0.047329  0.2631  1.1276/1.0863
```

Finite throughout.

## The term engaged, and stayed engaged

```
                    P0        P1        change
view JS             0.134904  0.047329   -65%
top-bin agreement   15.72%    26.31%     1.67x
decoded angle delta 2.1953    1.1719     -47%
->  CONSISTENCY_ACTUALLY_REDUCED_VIEW_SENSITIVITY = True
```

The suppression is stable rather than transient: measured at every mark it runs
-56%, -67%, -62%, -61%, -65%.  And `lambda * L_cons` settles at 0.443 against an
`L_sup` of 5.569, which is 8.0% -- close to the 5.6% the coefficient was
calibrated at, so it neither vanished nor ran away.

What it cost is visible in the same table: `L_sup` ends 7.6% above P0's, and
that is what the D2 medians are made of.

## Specialization

```
D2/D0 at 25,545      angle    offset
P1 +consistency      1.1276   1.0863
P0 aug only          1.2234   1.1480
closure              42.91%   41.71%
F1 late-A1 (context) 1.4255   1.2994
S1 L2-SP  (context)  1.1108   1.1545
```

P0's own gap opened sharply in its last stretch, 1.0839 to 1.2234 on angle
between 17,030 and 25,545.  P1's opened too but less, 1.0959 to 1.1276.  So the
consistency term did hold back the late widening, which is the mechanism the
screen proposed, and it did not buy accuracy with it.

## Tails

```
@25,545 D2      angle > 5 deg   angle > 10 deg   offset > 2 cell
P1                 0.2331           0.0912           0.2923
P0                 0.1929           0.0902           0.2594
```

P1 is worse on all three.  Lower view sensitivity did not translate into a
shorter error tail.

## Per role

```
role   n      P0 ang  P1 ang  F1 ang      P0 off  P1 off  F1 off
  0   508      1.938   1.978   2.065       1.204   1.283   0.861
  1   468      1.779   1.780   1.848       1.260   1.470   1.473
  2   511      2.242   2.918   2.490       1.065   1.067   1.273
  3   449      1.843   2.026   2.039       1.379   1.415   1.356
  4   505      2.437   3.030   2.938       1.020   0.967   1.015
  5   488      1.950   2.225   2.243       1.176   1.308   0.941
  6   504      2.272   3.053   2.785       1.006   1.036   0.951
  7   505      2.110   2.887   2.449       1.067   1.098   1.324
  8   504      1.580   1.713   1.887       0.926   0.881   0.807
  9   490      1.336   1.430   1.482       1.048   1.232   1.396
 10   481      1.541   1.464   1.533       0.934   1.105   1.150
 11   508      1.625   1.850   1.896       0.909   1.002   0.771
```

```
group             P0      P1      F1
high 2/4/6/7     2.265   2.972   2.665
lower 3/9/10     1.574   1.640   1.684
```

Role 10 is the only one where P1 beats P0 on angle; roles 4 and 8 are the only
ones on offset.  The damage concentrates in the high-angle group, which loses
31.2% against the low group's 4.2% -- the four roles that have been the hard
ones in every screen here are the four the consistency term hurts most.

## P0 against historical F1, context only

```
                 angle med   offset med   D2/D0 angle
P0 aug only       1.896942     1.065037     1.2234
F1 late-A1        2.070244     1.077348     1.4255
P0 vs F1           +8.37%       +1.14%
->  P0_NOT_MATERIALLY_DIFFERENT_FROM_F1
```

P0 sees two views per optimizer step, so its image exposure is twice F1's, and
that difference is not attributed.  Under the pre-registered thresholds the
label is "not materially different", which is what gets recorded -- the +8.37%
on angle is not called a gain.

Worth noting without claiming: P0's seen/unseen gap is 1.2234 against F1's
1.4255 at comparable medians.  Two arms, differing in exposure as well as in
augmentation, so nothing is isolated.

## What this settles

```
established     the consistency term reduces view sensitivity substantially and
                stably: JS -65%, top-bin agreement 1.67x, decoded angle -47%
established     it closes 42.91% and 41.71% of P0's seen/unseen gap
established     it costs 15.73% of the angle median and 5.97% of the offset
                median, and worsens all three tails
not established that appearance causes specialization.  The factor changed
                held-out geometry; that is the whole of the causal claim
not established anything about other coefficients, other photometric policies or
                other consistency formulations.  One lambda, fixed before the
                run by a gradient balance, never swept
```

## A defect in this run, and how it was handled

Both arms completed all 25,545 steps and every mark was evaluated, and then the
process raised `FileNotFoundError` while assembling its report: `locked_lambda`
had been pointed at the new coefficient file while the report assembly still
read the old path.  Nothing was written.

Rather than repeat six and a half hours, the report was rebuilt from the
per-mark checkpoints, and the restoration was checked against what the run had
already printed.  All ten marks across both arms match the logged medians and
p90s exactly, so the recomputation is faithful and is recorded as such: the
evaluation metrics come from the checkpoints, and the training scalars -- running
means no checkpoint carries -- were read back from the run log.

## Standing

```
DECISION                          APPEARANCE_CONSISTENCY_OVERREGULARIZES
CONSISTENCY_ACTUALLY_REDUCED_VIEW_SENSITIVITY   True (diagnostic)
GAP_CLOSED_20                     True
ACCURACY_WITHIN_10                False
architecture status               NOT_LOCKED
replicate                         BLOCKED (H1 only)
role shuffle                      BLOCKED
whole LINE_DEV                    BLOCKED
CIGM / PnP / dimensions / K-pose  BLOCKED
lambda                            not re-run, not swept, not adjusted
```

## Sealed

`untouched`, `eval56`, `wood45` and final-test remain unopened.
