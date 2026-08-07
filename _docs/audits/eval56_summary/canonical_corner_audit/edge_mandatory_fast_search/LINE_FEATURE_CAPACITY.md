# FROZEN_FEATURE_LINE_CAPACITY_FAIL

The budget is 1.0 degree and 0.5 belief cell.  Before writing a predictor that
must *find* lines, this asked the easier question: given a coarse line already
within 8 degrees and 4 cells, can any available feature be read precisely enough
to close the gap?

No PnP, no dimensions, no validation512.  Pure 2D line refinement, so the answer
does not depend on the dimension question at all.

## Result, line_dev512

```
arm             angle median   p90      offset median   p90     params
C0_F50            3.834 deg    7.096      1.561 cell   3.415    308,100
C1_F100           3.947        7.146      1.919        3.581    431,300
C2_MULTI          3.916        7.145      1.800        3.565    562,500
C3_RGB_STEM       3.918        7.429      2.020        3.592    177,700
gate              <= 1.0                  <= 0.5
```

Every arm fails, and they fail together: the spread across four very different
feature sources is 0.11 degree and 0.46 cell, against a gap to the gate of
roughly 4x in angle and 3x in offset.  That is not a feature-selection problem.

F50 is nominally best on both axes, which is consistent with the earlier capacity
audit and confirms that assuming F100 would have been wrong -- but the margin is
too small to call a winner, and no arm is admissible anyway.

A trainable RGB stem does not rescue it either, so this is not simply a matter of
the frozen A1 features being stale.

## Reading it honestly

Input jitter was uniform +/-8 degrees and +/-4 cells, so the median input error
was about 4 degrees and 2 cells.  The refiner ends at 3.8 degrees and 1.6 cells.
It is doing something -- offset improves -- but angle is essentially unimproved.
The refiner is not reading line orientation out of these features at all.

This is one epoch on 2,000 frames with a zero-initialised head, which is a real
limit on the claim: it shows the capacity is not *easily* reachable, not that it
is unreachable. What it does establish is that none of the four sources gives it
cheaply, and the previous three architectures all assumed something like this
precision was available.

## Consequence

```
FROZEN_FEATURE_LINE_CAPACITY_FAIL
```

Per the plan, the full SLQ predictor is not built.  Phases 5 through 12 are not
executed and no line-query model was written.

The gap is not closed by choosing a better feature among these four.  It would
need a different attack -- more training than a single epoch, a loss that
supervises orientation directly rather than through a normal vector, a higher
sampling resolution across the strip, or an input representation that makes edge
orientation explicit.  Which of those is worth trying is a research decision.

## Standing

```
LINE_TRAIN   13,618 frames / 260 groups      LINE_DEV   2,393 / 46
group overlap 0 · frame overlap 0 · holdout contamination 0
split sha 70ba7f1e8832bb0c

validation512   not used
untouched / eval56 / wood45 / final-test   unopened
```
