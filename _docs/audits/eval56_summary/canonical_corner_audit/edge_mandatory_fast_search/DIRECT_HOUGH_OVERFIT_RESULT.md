# DIRECT_HOUGH_NETWORK_FIT_FAIL

The pre-registered decision was at 3,000 steps and the model missed it.  FULL is
blocked.

```
overfit32, 32 frames, 368 supported roles

              angle med   angle p90   offset med   offset p90   train CE
@1,500         1.728384   11.002086     2.752819    12.592102    6.10696
@3,000         0.597847    2.095702     0.523695     2.213698    4.27863

task gate      <= 1.0      <= 2.0       <= 0.5       <= 1.0
```

Three of four are missed, and narrowly: angle p90 by 4.8%, offset median by
4.7%, offset p90 by 121%.  Angle median passes at 0.598.

```
tail            frac >5 deg   frac >10 deg   frac >2 cell
@1,500            0.2310         0.1168         0.5897
@3,000            0.0408         0.0109         0.1168
```

```
NEAR_GATE_BUT_STILL_FAIL
RAPIDLY_IMPROVING_TRAJECTORY
FULL = BLOCKED
```

## What this is not

The trajectory halves between the two marks -- angle 1.73 to 0.60, offset 2.75
to 0.52, cross-entropy 6.11 to 4.28, the >2-cell fraction 0.59 to 0.12 -- so this
is a miss at a chosen step count, not a converged failure.  That distinction is
recorded and it does not move the gate: 0.5237 was seen before any threshold
could be discussed, and 3,000 was fixed in `620bda9`.

The number must also not be divided by the image-map result.  0.598 degree is a
32-frame memorisation figure and 4.1793 degree is a held-out figure from a
different screen; their ratio is not a representation gain and no such claim is
made here.

## Standing

```
oracles          O_DOMAIN / O_GRID / O_TARGET / O_SCORER all PASS
overfit32        FAIL at the pre-registered 3,000-step decision
FULL             not started -- the runner raised before it
```

The formulation is already cleared by O_SCORER, so whatever this is, it is not
the scorer.  No PnP, no CIGM, no dimensions.  `untouched`, `eval56`, `wood45`
and final-test remain unopened.
