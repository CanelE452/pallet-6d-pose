# ROLE_CONDITIONED_GLOBAL_MAP_FAIL

Role queries with genuine nonlocal access to a position-carrying F50 buy 6.5% on
angle and 4.6% on offset.  The pre-registered bar was 40% on both.

```
D2_LINE_DEV512, 8,515 steps       angle med   offset med   angle p90   train loss
Q0_LOCAL_XY  (= C_G0P1)              4.4705      1.9697      35.084      0.07269
Q1_ROLE_QUERY_GLOBAL                 4.1793      1.8788      36.605      0.07164
reduction                            -6.51%      -4.61%

40% threshold                     <= 2.682305  <= 1.181795
absolute budget                   <= 1.0       <= 0.5      p90 <= 2.0
Q0 baseline drift                  0.000e+00   (all four marks, both metrics)
```

## Trajectories

```
arm                    step   D0_SEEN512               D2_LINE_DEV512           loss
Q0_LOCAL_XY            1,250  6.2349/2.8074 p90 47.9   6.1131/2.7862 p90 46.7  0.07791
Q0_LOCAL_XY            2,500  5.4155/2.5244 p90 45.7   5.3864/2.4973 p90 46.5  0.07622
Q0_LOCAL_XY            5,000  4.9706/2.1567 p90 40.8   4.8973/2.1344 p90 39.2  0.07445
Q0_LOCAL_XY            8,515  4.3724/1.8359 p90 35.8   4.4705/1.9697 p90 35.1  0.07269
Q1_ROLE_QUERY_GLOBAL   1,250  6.0472/2.8207 p90 48.2   6.0215/2.6990 p90 46.8  0.07786
Q1_ROLE_QUERY_GLOBAL   2,500  5.5027/2.5763 p90 44.7   5.4071/2.4980 p90 43.6  0.07622
Q1_ROLE_QUERY_GLOBAL   5,000  4.8389/2.0972 p90 39.2   4.7269/2.0930 p90 38.5  0.07423
Q1_ROLE_QUERY_GLOBAL   8,515  4.3756/1.8800 p90 35.9   4.1793/1.8788 p90 36.6  0.07164
```

Q0 reproduces C_G0P1 at every mark to `0.000e+00`, so the difference is the
factor.  The per-mark deltas are -1.5%, +0.4%, -3.5%, -6.5%: never large, and not
monotone until the end.

## The branch was used

This is not a dead-branch result.  The gate opens and keeps opening:

```
mark    alpha      descriptor norm   attention entropy
1,250   0.04902    52.31             7.090
2,500   0.04738    44.54             6.878
5,000   0.05376    29.71             6.452
8,515   0.10164    38.66             7.004
```

Alpha doubles over the run and the attention entropy falls from 7.09 to 6.45
before rising again -- the queries do attend somewhere and the residual does
enter the logits.  The wiring test had already established a live gradient
(alpha grad 2.408e-03 at step 0, attention grad 6.636e-06 at step 2).  So the
factor was exercised and its effect is simply small.

Attention entropy is reported as a diagnostic.  It is not evidence about *what*
the queries look at, and no such claim is made.

## Per role

```
role   Q0 angle   Q1 angle      Q0 offset  Q1 offset
0        3.0661     2.9199        1.4401     1.3625
1        5.8999     5.3444        4.4340     3.9984
2        3.8285     3.9223        1.0437     1.1736
3        6.6417     5.8982        4.0174     3.4798
4        4.4423     4.1062        1.8170     1.7617
5        3.7479     3.7214        1.5155     1.6236
6        4.5915     4.3120        1.8003     1.7177
7        4.1921     4.1918        1.3689     1.2304
8        3.2589     3.6364        1.4035     1.4304
9        4.5082     3.8789        2.6620     2.3332
10       4.5248     3.7110        2.2604     1.8639
11       4.7068     4.5032        2.2053     2.0958
```

Nine of twelve roles improve on angle, three worsen, and the largest gains are on
the roles that were already worst -- role 3 from 6.64 to 5.90, role 10 from 4.52
to 3.71, role 1 from 5.90 to 5.34.  The effect is broad rather than a rescue of
one role, which is the shape a genuine global signal would have.  It is still
6.5%.

Roles 1 and 3 remain the worst on offset by a wide margin (3.48 and 4.00 against
about 1.2 for roles 2 and 7).  That ordering is stable across both arms and is
not explained by this screen.

## Hough coherence

```
              margin median    peak entropy
Q0            2.689e-04        11.48
Q1            2.599e-04        11.48
```

Unchanged.  The decoder is no more certain about Q1's maps than Q0's -- the
improvement is in where the ridge sits, not in how peaked the accumulator is.

## Verdict

```
ROLE_CONDITIONED_GLOBAL_MAP_FAIL
next: DIRECT_HOUGH_SPACE_ROLE_HEATMAP
```

Per the locked branch, the image-space map architecture is not explored deeper or
wider from here.  Three shapes of nonlocal or positional conditioning have now
been measured on top of the same output representation:

```
GAP-FiLM global content        -3.4% angle
absolute XY                   -20.1% angle
role-query nonlocal global     -6.5% angle on top of XY
```

Absolute position remains the only factor worth more than a few percent, and the
combined best is 4.18 degrees -- 4.2x the median budget and 18x the p90 safety
line.

What the next screen may not do is re-open this one.  The recorded direction is
twelve fixed-role heatmaps in Hough space `(theta_bin, rho_bin)` predicted
directly from role queries over global F50 tokens -- still a spatial prediction,
not scalar regression -- and it is a *representation* change, which is why it is
locked separately rather than added here.

Full LINE_DEV promotion and the role-shuffle causality check did not run: both
are gated on qualification.  `ROLE_QUERY_SEMANTICS` is therefore unevaluated for
this arm.

No MAP200, no deeper local head, no RGB stem, no data filtering, no loss or sigma
change.  No PnP, no CIGM, no dimensions.  `untouched`, `eval56`, `wood45` and
final-test remain unopened.
