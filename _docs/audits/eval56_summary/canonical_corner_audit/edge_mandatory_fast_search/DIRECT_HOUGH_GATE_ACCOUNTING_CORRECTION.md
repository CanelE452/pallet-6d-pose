# The 3,000-step gate accounting was mine to get wrong

`af3e8cf`, `e8deaa0` and `4f0c0ec` stand unedited.  The runner's verdict was
right throughout; my reading of its per-mark log was not.

## What misled me

`summarise()` carries two separate booleans:

```
PASS     = angle median <= 1.0  AND  offset median <= 0.5     (medians only)
SAFETY   = angle p90    <= 2.0  AND  offset p90    <= 1.0     (p90 only)
```

The per-mark log line prints `PASS=` -- the median-only field.  The overall
verdict needs all four gates, which is what `gates()` computes at the decision
step, and that function was correct.  I read `PASS=True` at 3,000 and 4,500 as
an overall pass.  It is not.

## All four gates, at every mark

```
                    angle med   offset med   angle p90   offset p90   overall
historical @1,500    1.728384 F   2.752819 F  11.002086 F  12.592102 F  FAIL
historical @3,000    0.597847 P   0.523695 F   2.095702 F   2.213698 F  FAIL
extension  @1,500    1.453671 F   2.221141 F  11.002086 F  11.835798 F  FAIL
extension  @3,000    0.504826 P   0.415072 P   1.945399 P   2.260027 F  FAIL
extension  @4,500    0.485104 P   0.429530 P   1.473585 P   1.191151 F  FAIL
extension  @6,000    0.337997 P   0.230449 P   0.889731 P   0.553381 P  PASS

gate                 <= 1.0       <= 0.5       <= 2.0       <= 1.0
```

Offset p90 is what fails at 3,000 and 4,500, by 126% and 19%.

## Withdrawn

```
withdrawn   "two runs of the same 3,000 steps land on opposite sides of the gate"
withdrawn   "a single draw there cannot separate too-few-steps from an unlucky draw"
withdrawn   RUN_TO_RUN_VERDICT_INSTABILITY_AT_3000
withdrawn   "run-to-run variability that straddles the gate"
```

Both 3,000-step runs are overall FAIL.  Nothing crossed a verdict boundary.

## What replaces it

```
RUN_TO_RUN_COMPONENT_METRIC_DRIFT_PRESENT

historical -> extension at 3,000
  angle median   -15.6%
  angle p90       -7.2%
  offset median  -20.7%
  offset p90      +2.1%
OVERALL_VERDICT   FAIL -> FAIL

RUN_TO_RUN_VERDICT_INSTABILITY_AT_3000   NOT ESTABLISHED
```

The drift is real and its cause is still `UNRESOLVED` -- the `grid_sample`
attribution stays withdrawn, since 60 steps reproduce bit-exactly within and
across processes, the code has no diff since `620bda9`, and no convolution in
this model carries a gradient.  But the drift is in components, not in verdicts,
and it must not be described as gate-crossing.

## And the step-budget reading is restored

```
extension @3,000  FAIL
extension @4,500  FAIL
extension @6,000  PASS

DIRECT_HOUGH_OVERFIT_EXTENDED_PASS      CONFIRMED
OVERFIT_STEP_BUDGET_INSUFFICIENT        SUPPORTED_BY_EXTENSION
```

The first overall pass appears exactly at the extended budget, and `af3e8cf`'s
`DIRECT_HOUGH_NETWORK_FIT_FAIL` at 3,000 stands as its own result.  This does
**not** claim the budget is the sole cause of every historical metric
difference; the component drift above is unexplained and separate.

## Scope of the running FULL

```
POSITION_INFORMATION_PRESENT        TRUE   role-query tokens are F50 concat (x, y)
MAP_ADDITIVE_XY_BRANCH_PRESERVED    FALSE  DirectHoughModel builds AbsoluteXY
                                           and never calls it
family name                         DIRECT_HOUGH_TOKEN_XY_V0
```

`self.position` is neither deleted nor wired up.  Its constructor consumes RNG
before the encoder and head are built, so removing it would change every
initialisation in this family and invalidate the comparison.  The current FULL is
reported as `DIRECT_HOUGH_TOKEN_XY_V0` and never as "direct Hough with the map's
XY factor preserved".

## Stability rule, locked before the FULL result is read

If FULL qualifies -- 40% reduction on both medians, or task plus safety PASS --
one additional replicate runs first:

```
same seed, same FULL protocol, fresh init, no threshold change

both replicates in the same qualification class   FULL_VERDICT_STABLE
different classes                                 DIRECT_HOUGH_FULL_VERDICT_UNSTABLE
                                                  -> promotion blocked
```

Promotion to the whole LINE_DEV, and CIGM after it, require
`FULL_VERDICT_STABLE` and a causal role shuffle.  A FULL that fails to reach 40%
is `DIRECT_HOUGH_TOKEN_XY_FULL_FAIL` and blocks CIGM outright.

The running FULL is not modified, interrupted or restarted.
