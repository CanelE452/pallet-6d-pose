# Two overstatements in e49e304

The commit stands unedited.  Two sentences in it claim more than the numbers.

## "The two do not interact"

```
D_G1P1 vs C_G0P1 at 8,515 on D2
  angle    -0.905%
  offset   -5.634%
```

A 5.6% offset improvement is not nothing.  What the screen established is that
the combination does not reach the pre-registered bar, not that the factors are
independent.

```
withdrawn    "There is no interaction term to speak of"
recorded     NO_QUALIFYING_INTERACTION
             a small offset-conditional effect may exist and is nowhere near
             qualification
```

## "an order of magnitude short"

Applied to the median, that is wrong.

```
C_G0P1 at 8,515, D2
  angle  median 4.470509  against 1.0    ->  4.47x
  offset median 1.969659  against 0.5    ->  3.94x
  angle  p90    35.083672 against 2.0    -> 17.5x
```

The median gap is about 4x.  The order-of-magnitude language belongs to the
safety tail and only there.  Where the earlier report said the best arm was "an
order of magnitude short" of the budget, read: **4.5x on the median, 17.5x on the
p90 safety line.**

## What is unchanged

Absolute XY remains the largest single architectural effect measured -- 20.1% on
angle and 12.8% on offset against A, monotone across four marks, with p90 falling
28.5%.  `GAP_FILM_GLOBAL_CONTEXT_FAIL` remains the accurate scope for the G
factor, and no claim is made about global context in general.
