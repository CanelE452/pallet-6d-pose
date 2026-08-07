# Data scale does not rescue line refinement

Two diagnostics were declared in `V2_SEARCH2K_QUALIFICATION.md` and committed
before the run.  Both are now closed.

```
LOCAL_EDGE_REPRESENTATION_PRECISION_FAIL     all three arms
SLQ                                          NOT BUILT
```

## O1C: even the target edge's own gradient is not enough

O1B gave the refiner a support channel and left the surrounding feature values
intact.  O1C multiplies magnitude and both orientation components by that
support, so everything off the physical edge is literally zero.

```
                    angle med   offset med
identity               3.938       1.974
gate                  <=1.000     <=0.500
O1C overfit32          0.015       0.007
O1C epoch 5            3.821       1.723     -3.0% / -12.7% vs identity
   O1A epoch 5         3.651       1.882
   O1B epoch 5         3.596       1.695
support fraction       0.151
```

**FAIL.**  Handing the refiner exactly the target edge's gradient and nothing
else leaves it 3.8 degrees from a 1.0 degree gate.  The bottleneck is therefore
not clutter or support selection -- raw Scharr does not carry this precision.

Against expectation, O1C is *worse* on angle than O1B (3.821 vs 3.596).  Erasing
the surroundings removed information that helped slightly, so a hard mask is not
strictly better than telling the network where to look.

## The factorial

```
arm           condition       data    steps   angle med  offset med  angle p90  offset p90
C0_F50        A_2K_SHORT     2,000      835      2.703      1.310      6.434      3.234
              B_FULL_SHORT  13,618      835      2.837      1.333      6.652      3.192
              C_2K_LONG      2,000    5,675      2.259      1.188      5.955      3.148
              D_FULL_LONG   13,618    5,675      2.184      1.183      5.850      3.069
C2_MULTI      A_2K_SHORT     2,000      835      2.727      1.365      6.487      3.204
              B_FULL_SHORT  13,618      835      3.108      1.363      6.881      3.303
              C_2K_LONG      2,000    5,675      2.342      1.212      6.060      3.129
              D_FULL_LONG   13,618    5,675      2.307      1.219      6.027      3.070
C3_RGB_STEM   A_2K_SHORT     2,000      835      3.407      1.825      6.978      3.580
              B_FULL_SHORT  13,618      835      3.532      1.712      7.010      3.441
              C_2K_LONG      2,000    5,675      2.847      1.479      6.744      3.331
              D_FULL_LONG   13,618    5,675      2.846      1.506      6.648      3.294

identity                                          3.938      1.974      7.234      3.584
gate                                             <=1.000    <=0.500    <=2.000    <=1.000
```

Against the thresholds fixed before the run:

```
arm           D vs A angle   D vs A offset   SCALING_SIGNAL (>=40%)   APPROACHES_GATE
C0_F50           -19.2%          -9.6%             no                     no
C2_MULTI         -15.4%         -10.7%             no                     no
C3_RGB_STEM      -16.5%         -17.5%             no                     no
```

No arm passes, none reaches the 40% diagnostic signal, and none comes within
1.5 degrees / 0.75 cell.  The best condition anywhere is F50 at D: 2.184 degrees
and 1.183 cell, which is 2.2x and 2.4x the gate.

## Where the improvement comes from

Splitting the 2x2 into its two legs, all three arms agree:

```
arm           C - A (steps x6.8)   D - C (data x6.8)   B - A (data x6.8, same steps)
C0_F50            -0.444 deg           -0.075 deg            +0.134 deg
C2_MULTI          -0.385               -0.035                +0.382
C3_RGB_STEM       -0.559               -0.002                +0.125
```

Optimizer steps move the metric five to nearly three hundred times more than
data does.  Adding 11,618 frames at a fixed step budget makes every arm *worse*
on angle, and adding them at the long budget buys between 0.002 and 0.075
degrees -- on a 3.9 degree starting error.

`scaling_decision` labels every arm `DATA_DIVERSITY_HELPS`, because the
pre-declared rule fires when *either* `B < A` or `D < C`.  The label is what the
declared rule produces and it is left as recorded, but it should not be read as
support for more data: the two legs disagree in sign, and the leg that fires
does so by margins near zero.  `OPTIMIZATION_STEPS_HELP` is the substantive one.

The B-versus-A reversal has a mechanical reading.  At 835 steps the full pool is
still inside its first pass, so every frame is seen once, while the 2k pool has
seen its frames five times each.  B therefore mixes more diversity with less
repetition.  D removes that confound by completing five passes -- and the
advantage it recovers is 0.002 to 0.075 degrees.

## One arm is not reproducible, and that is not a bug

`CONDITION_A_NOT_REPRODUCED` fired for `C3_RGB_STEM` with a drift of 9.7e-02
while F50 and MULTI reproduced their recorded epoch-5 numbers to 0.0 exactly.

Measured, not assumed: twenty identical steps rerun back to back give

```
C3_RGB_STEM   4.008037090  vs  4.009006500   identical = False
C0_F50        4.001834869  vs  4.001834869   identical = True
```

with `cudnn.benchmark=False` and `cudnn.deterministic=True`.  The stem is the
only arm whose parameters sit upstream of `grid_sample`, so it is the only arm
whose backward pass computes the input gradient of `grid_sample` -- an atomicAdd
reduction that `cudnn.deterministic` does not cover.  The frozen-feature arms
detach before the sampler and never take that path.

Reusing a past run as condition A is therefore invalid for the stem arm, and its
own re-measured `k2@S_SHORT` is used instead, which keeps the D-versus-A
comparison inside a single trajectory.  The guard still demands exact
reproduction from the two deterministic arms, so a genuine schedule bug would
still stop the run.  `scaling_decision.json` records
`condition_A_source` and `condition_A_drift_vs_recorded` per arm.

Consequence for reading the RGB numbers: its A-to-D margins carry a run-to-run
component that was not quantified at 835 or 5,675 steps -- only that 20 steps
already diverge by ~1e-3.  The other two arms have no such caveat.

## Verdict

```
SEARCH2K_LINE_REFINEMENT_FAIL             CONFIRMED (406ecf8)
DATA_SCALE_UNRESOLVED                     CLOSED
LOCAL_EDGE_REPRESENTATION_PRECISION_FAIL  all three arms
SLQ                                       NOT BUILT
```

Three causes were on the table -- representation, data, optimizer steps.  Steps
were real but small and already saturating; data at 6.8x was worth almost
nothing and hurt at a fixed budget; and O1C shows the failure survives handing
the refiner the target edge's gradient in isolation.  What remains is the
representation.

The next direction is an explicit line or edge representation, or an encoder
trained with line supervision -- not this refiner with more data.

No PnP, no dimensions, no `validation512` tuning.  `untouched`, `eval56`,
`wood45` and final-test remain unopened.
