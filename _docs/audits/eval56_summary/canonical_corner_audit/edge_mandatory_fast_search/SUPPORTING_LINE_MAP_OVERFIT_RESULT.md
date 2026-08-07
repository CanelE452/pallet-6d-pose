# SUPPORTING_LINE_MAP_OPTIMIZATION_FAIL, and what that label hides

Both arms miss the overfit32 gate, so `search2k` is blocked by the protocol and
did not run.  The number that matters is not the verdict but the distance.

```
                     angle med   angle p90   offset med   offset p90   finite
O_LOSS free logits      0.0065      0.0129       0.0062       0.0116
M0_F50_SLINE            0.1268      0.4531       0.0586       0.1822    1.000
M1_F50_RGB_SLINE        0.1426      0.4715       0.0614       0.2139    1.000

overfit32 gate         <=0.10      <=0.25       <=0.05       <=0.15
primary budget         <=1.00      <=2.00       <=0.50       <=1.00
```

Both arms fail all four overfit gates.  Both arms also sit **inside the primary
budget by a factor of about eight** on the same 32 frames, and clear its safety
and approach thresholds.  The gate they missed was set an order of magnitude
tighter than the budget the screen ultimately cares about, deliberately, as an
optimisation sanity check -- so `SUPPORTING_LINE_MAP_OPTIMIZATION_FAIL` is
correct as declared and would be badly misread as "the arms cannot fit 32
frames".

I am not moving the gate.  It was fixed in `cc82012` before the run and the
protocol says both-fail blocks `search2k`; that is the result.  What the numbers
say is recorded here so the next locked screen can decide with them.

## Where the gap is

The loss oracle reached 0.0065 degree with free logits, so the loss and the
decoder are not the limit.  The network's map is about twenty times worse than
free logits, and the diagnostics say why:

```
                   positive MSE   negative MSE   map NCC   mass    peak
M0_F50_SLINE            0.0419         0.0086     0.7749  756.0   0.982
M1_F50_RGB_SLINE        0.0379         0.0087     0.7927  792.5   0.985
```

Peak probability is 0.98, so the head is confident and the ridge is there; the
correlation with the target map is 0.77-0.79, so roughly a fifth of the map
structure is wrong.  Positive MSE is five times negative MSE -- the error is
concentrated on the line itself, not on the background.  The failure is a
blurred or displaced ridge, not a missing one.

## The RGB stem does not help

M1 is slightly *worse* than M0 on every metric, on data it is being asked to
memorise.  Adding a trainable RGB stem to a frozen F50 did not buy capacity
here, which is the same ordering the V2 arms showed.

## Partial roles no longer collapse

```
                  IN_FRAME_FULL (n=326)     IN_FRAME_PARTIAL (n=47)
M0                0.1255 / p90 0.4295       0.1478 / p90 0.5504
M1                0.1365 / p90 0.4702       0.1768 / p90 0.4873
```

Under the finite-segment target, partial roles ran an order of magnitude worse
than full ones at every stage.  Under the supporting-line target they are within
20% of each other.  That is the extent-mismatch correction holding up in a
*trained* setting rather than only in an oracle -- and it is the strongest
positive result in this run.

Caveat: 47 partial roles.  The direction is clear, the size is not.

## Standing

```
O_LOSS                              PASS
M0 / M1 overfit32                   FAIL (both), inside the primary budget
search2k                            BLOCKED, not run
SUPPORTING_LINE_MAP_CAPACITY        STILL UNMEASURED
role shuffle / full-vs-partial dev  NOT RUN
off-frame                           NOT TESTED
CIGM / PnP                          NOT BUILT
```

Nothing is repaired here.  Two things a next locked screen would have to decide,
recorded and not acted on:

1. Whether an overfit gate ten times tighter than the deployment budget is the
   right sanity check, or whether it should be stated relative to the budget.
   Deciding that now, having seen 0.1268 against 0.10, would be moving a
   threshold to pass it.
2. Whether the remaining gap is head capacity or feature resolution.  The map
   NCC of 0.78 with a confident peak points at a ridge that is present but
   imprecise, which a deeper head or a finer output grid would address
   differently.

No PnP, no CIGM, no dimensions, no `validation512`.  `untouched`, `eval56`,
`wood45` and final-test remain unopened.
