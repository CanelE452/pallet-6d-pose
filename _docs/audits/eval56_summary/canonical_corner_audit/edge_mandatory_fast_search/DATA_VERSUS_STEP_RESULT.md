# DATA_SCALE_NOT_THE_BOTTLENECK, and K2_STEP_SCALE_WEAK

Neither more optimizer steps nor a bigger pool brings the map within reach of the
budget, and the two legs each fall short of the 40% diagnostic threshold that was
fixed before the run.

```
trajectory  step    D0_SEEN512        D2_LINE_DEV512    train loss (last 250)   slope
K2          1,250   6.6040 / 2.7023   6.8450 / 2.7717        0.07897          +1.1e-06
K2          2,500   5.8472 / 2.3016   6.1957 / 2.5498        0.07779          +1.9e-06
K2          5,000   5.0943 / 1.9420   6.1496 / 2.4944        0.07665          +2.1e-06
K2          8,515   4.2845 / 1.6389   5.9320 / 2.4032        0.07578          -7.6e-07
FULL        1,250   7.2683 / 2.9629   7.0451 / 2.9216        0.07964          -4.0e-06
FULL        2,500   6.7188 / 2.7639   6.6700 / 2.7448        0.07831          -5.1e-06
FULL        5,000   6.3448 / 2.5352   5.9964 / 2.4276        0.07725          -5.6e-07
FULL        8,515   5.7079 / 2.2668   5.5966 / 2.2597        0.07624          -3.9e-06

budget <= 1.0 degree / <= 0.5 canonical50 cell        A drift 0.000e+00
```

## Condition A reproduces exactly

All eight metrics of the fresh K2 trajectory at step 1,250 equal the recorded
`13ca73d` checkpoint to `0.000e+00`, on both populations.  The training semantics
did not change, so this 2x2 is measured with the same instrument as everything
before it.

## The step leg

```
A -> C on D0_SEEN512    angle -35.1%   offset -39.4%     threshold 40%
A -> C on D2_LINE_DEV512 angle -13.3%   offset -13.3%
```

Both fall under 40%, on both populations, so `OPTIMIZATION_STEPS_RESCUE_MAP_FIT`
does not fire and neither does `OPTIMIZATION_SIGNAL_PRESENT_BUT_INSUFFICIENT`.
`K2_STEP_SCALE_WEAK` is the recorded outcome.

But the two populations disagree about *where* the improvement went, and that is
the more informative fact.  Between 1,250 and 8,515 steps the K2 trajectory
improves by 35% on the frames it trains on and by 13% on the holdout, and the gap
between them widens monotonically:

```
K2 step     D0      D2      gap
1,250     6.6040  6.8450   +3.6%
2,500     5.8472  6.1957   +6.0%
5,000     5.0943  6.1496  +20.7%
8,515     4.2845  5.9320  +38.5%
```

`13ca73d` found no train-to-holdout gap and concluded underfit.  That was true at
1,250 steps and it does not survive a longer schedule: run the same 2,000 frames
34 times and a gap appears.  `SEARCH2K_MODEL_UNDERFIT_CONFIRMED` stands as a
statement about that checkpoint, not about the recipe at every budget.

## The data leg

```
C -> D on D0_SEEN512     angle +33.2%   offset +38.3%    (worse)
C -> D on D2_LINE_DEV512 angle  -5.7%   offset  -6.0%    (better)
```

The sign flips with the population, and the reason is not subtle: `D0_SEEN512` is
drawn from `line_search2k`, so it is C's own training data and D has never seen
it.  On that population C must win and the comparison says nothing about data
scale.  On the holdout, which is fair to both, the full pool is better by 6% --
real, consistent across angle and offset, and nowhere near the 40% threshold.

The verdict function reads `D0`, which is why the recorded `C_to_D` reductions
are negative.  The label is the same under either population, since no leg
reaches 40% and nothing passes, but the honest reading of the data effect is the
holdout one: **about 6%, not a rescue**.

Data does buy the thing K2 loses.  The FULL trajectory's train-to-holdout gap
stays under 2% at every step while K2's reaches 38.5%, so a larger pool prevents
the specialisation rather than raising the ceiling.

## Optimization state

The train map loss falls from 0.0790 to 0.0758 on K2 and 0.0796 to 0.0762 on
FULL -- about 4% over 8,515 steps.  Final 250-step slopes are order 1e-6, and
K2's is `-7.6e-07`, effectively flat.  `OPTIMIZATION_NOT_CONVERGED` is not
claimed: the loss is no longer moving at a rate that could matter.

## Verdict

```
CAUSE                 DATA_SCALE_NOT_THE_BOTTLENECK
                      K2_STEP_SCALE_WEAK
condition A drift     0.000e+00
best anywhere         D_FULL_LONG on the holdout, 5.5966 degree / 2.2597 cell
                      5.6x and 4.5x the budget
```

Neither lever moves the result to within a factor of five of the target, and the
training loss has stopped moving.  Both of the pre-registered conditions for
opening an architecture screen are now met:

```
1) training loss and geometry have plateaued          yes
2) long-budget E0 remains far outside the task        yes, 5.7 degree
```

The pre-registered first candidates are `GLOBAL_CONTEXT` and
`POSITIONAL_COORDINATES`, as a 2x2 rather than one at a time.  `MAP200` stays
behind them: the perfect-map oracle at MAP100 decodes to about 0.006 degree, so
there is still no quantization evidence.  Nothing is built here.

## One defect, mine

The reproduction guard first fired `CONDITION_A_NOT_REPRODUCED` at drift
3.126e-05.  The trajectory was exact; I had hardcoded the reference as
`6.6040 / 2.7023`, four-decimal values transcribed from my own report, and
compared them at 1e-6.  The guard was measuring my rounding.  It now reads the
recorded JSON at full stored precision and the drift is `0.000e+00` -- a stricter
comparison, not a relaxed one, and no threshold changed.

No filtering, no deletion, no M1, no MAP200, no architecture change.  No PnP, no
CIGM, no dimensions.  `untouched`, `eval56`, `wood45` and final-test remain
unopened.
