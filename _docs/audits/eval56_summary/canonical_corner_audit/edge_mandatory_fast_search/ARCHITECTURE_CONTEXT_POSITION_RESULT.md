# SIMPLE_CONTEXT_POSITION_SCREEN_FAIL

No arm qualifies.  Absolute position is clearly the larger of the two factors and
still buys only a fifth of the error.

```
D2_LINE_DEV512, 8,515 steps        angle med    vs A     offset med    vs A
A_G0P0   baseline                     5.5966     0.0%        2.2597    0.0%
B_G1P0   GAP-FiLM global              5.4089    -3.4%        2.2300   -1.3%
C_G0P1   absolute XY                  4.4705   -20.1%        1.9697  -12.8%
D_G1P1   both                         4.4300   -20.8%        1.8587  -17.7%

40% threshold                        <= 3.35796            <= 1.35582
absolute budget                      <= 1.0                <= 0.5
safety p90                           <= 2.0                <= 1.0
```

Both metrics must clear a threshold for an arm to qualify, and none does on
either.  The best arm is 4.4 times the angle budget and 3.7 times the offset
budget, with p90 at 34.1 degrees against a 2.0 safety line.

## Full trajectories

```
arm      step   D0_SEEN512                D2_LINE_DEV512            train loss
A_G0P0   1,250  7.2683 / 2.9629 p90 54.5  7.0451 / 2.9216 p90 53.7    0.07964
A_G0P0   2,500  6.7188 / 2.7639 p90 54.1  6.6700 / 2.7448 p90 54.2    0.07831
A_G0P0   5,000  6.3448 / 2.5352 p90 52.4  5.9964 / 2.4276 p90 52.1    0.07725
A_G0P0   8,515  5.7079 / 2.2668 p90 47.2  5.5966 / 2.2597 p90 49.1    0.07624
B_G1P0   1,250  7.2731 / 3.0166 p90 54.6  7.0122 / 2.8608 p90 52.8    0.07952
B_G1P0   2,500  6.4515 / 2.5618 p90 51.2  6.2866 / 2.6484 p90 53.4    0.07804
B_G1P0   5,000  5.9957 / 2.3306 p90 52.9  5.9794 / 2.4206 p90 51.6    0.07697
B_G1P0   8,515  5.4769 / 2.1435 p90 48.9  5.4089 / 2.2300 p90 49.4    0.07596
C_G0P1   1,250  6.2349 / 2.8074 p90 47.9  6.1131 / 2.7862 p90 46.7    0.07791
C_G0P1   2,500  5.4155 / 2.5244 p90 45.7  5.3864 / 2.4973 p90 46.5    0.07622
C_G0P1   5,000  4.9706 / 2.1567 p90 40.8  4.8973 / 2.1344 p90 39.2    0.07445
C_G0P1   8,515  4.3724 / 1.8359 p90 35.8  4.4705 / 1.9697 p90 35.1    0.07269
D_G1P1   1,250  6.1049 / 2.7298 p90 47.5  6.1131 / 2.6569 p90 47.4    0.07770
D_G1P1   2,500  5.4836 / 2.3625 p90 45.6  5.4073 / 2.4357 p90 46.7    0.07584
D_G1P1   5,000  4.9747 / 2.0836 p90 40.5  4.9036 / 2.0682 p90 38.1    0.07396
D_G1P1   8,515  4.5164 / 1.8395 p90 36.6  4.4300 / 1.8587 p90 34.1    0.07199
```

A reproduces the recorded baseline at all four marks, and the four arms were
identical functions at step 0 to `0.000e+00`, so every difference above is the
factor and not the draw.

## What the factorial says

**Position is real and insufficient.**  C beats A by 20.1% on angle and 12.8% on
offset, monotonically across all four marks, and it moves the tail hardest:
angle p90 falls 49.1 to 35.1, a 28.5% reduction.  Absolute XY mostly stops the
model from being badly wrong, which is what one would expect if a supporting line
that spans the frame cannot be placed without knowing where the pixel is.

**GAP-FiLM global content contributes almost nothing.**  B is 3.4% and 1.3%,
and at 5,000 steps it was 0.3% -- inside the noise of its own trajectory.

**The two do not interact.**  D matches C on angle at every mark (6.1131 against
6.1131, 5.4073 against 5.3864, 4.9036 against 4.8973, 4.4300 against 4.4705) and
is consistently 2-5% better on offset.  Adding global content on top of position
buys a small offset improvement and no angle improvement, which is the same size
of effect B showed on its own.  There is no interaction term to speak of.

## Naming the failure narrowly

```
SIMPLE_CONTEXT_POSITION_SCREEN_FAIL
```

The two implementations tested were GAP-FiLM conditioning and additive
normalised XY.  What is established is that **those two** do not explain the
precision gap.  None of the following is claimed:

```
not claimed   global context is unnecessary
not claimed   position is unnecessary
not claimed   ALL_GLOBAL_CONTEXT_FAIL
```

`GAP_FILM_GLOBAL_CONTEXT_FAIL` is the accurate scope for the G factor.  For P the
honest statement is the opposite of a rejection: absolute position is the largest
single architectural effect measured so far, and it is still an order of
magnitude short.

## Role causality was not run

The pre-registered rule runs the shuffle only on a qualifying arm.  None
qualifies, so `role_shuffle.json` does not exist and `ROLE_SEMANTICS` is not
evaluated for these arms.

## Next, per the locked branch

MAP200 is not the move.  The perfect-map oracle at MAP100 decodes to about 0.006
degree, so raster resolution cannot account for an error of 4.4 degrees.

The registered direction is a separate pre-registration for richer non-local
structural reasoning or role-conditioned global decoding.  Nothing is designed
here, and the factor definitions, gates, step budget and data pool of this screen
are not revisited to search again inside it.

```
STRUCTURAL_LINE_MAP_CAPACITY   still unmeasured at the task budget
LINE_MAP_GO                    not reached
CIGM / PnP                     NOT BUILT
```

No PnP, no CIGM, no dimensions, no MAP200, no `validation512`.  `untouched`,
`eval56`, `wood45` and final-test remain unopened.
