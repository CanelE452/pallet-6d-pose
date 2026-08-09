# DIRECT_HOUGH_OVERFIT_EXTENDED_PASS, with two things the run also exposed

```
extension, fresh 0 -> 6,000, 32 frames, 368 supported roles

           angle med   angle p90   offset med   offset p90
@1,500      1.453671   11.002086     2.221141    11.835798
@3,000      0.504826    1.945399     0.415072     2.260027
@4,500      0.485104    1.473585     0.429530     1.191151
@6,000      0.337997    0.889731     0.230449     0.553381

gate        <= 1.0      <= 2.0       <= 0.5       <= 1.0
margin       2.96x       2.25x        2.17x        1.81x
tail @6,000  frac >5 deg 0.0000   >10 deg 0.0000   >2 cell 0.0000
```

All four gates pass at the pre-registered 6,000-step decision, with 1.8x to 3.0x
of margin and an empty tail.

`af3e8cf` is untouched and `DIRECT_HOUGH_NETWORK_FIT_FAIL` stands as the result
of the 3,000-step decision.

## The pre-registered interpretation does not survive

The plan said a pass here would mean the original failure was
`OVERFIT_STEP_BUDGET_INSUFFICIENT`.  **It cannot be read that way.**

```
@3,000        angle med   angle p90   offset med   offset p90   verdict
recorded       0.597847    2.095702     0.523695     2.213698   FAIL 3/4
extension      0.504826    1.945399     0.415072     2.260027   PASS 4/4
delta           -15.6%       -7.2%       -20.7%        +2.1%
```

Two runs of the same 3,000 steps land on opposite sides of the gate.  So the
original miss was not only a budget question -- at 3,000 the outcome is not
stable, and a screen that reads a single draw there cannot distinguish "too few
steps" from "unlucky draw".

What the 6,000 result does support is narrower and still useful:

```
supported     at 6,000 the fit clears every gate with 1.8-3.0x margin, which
              survives the +-21% run-to-run spread observed at 1,500 and 3,000
not supported OVERFIT_STEP_BUDGET_INSUFFICIENT as the explanation of af3e8cf
```

## The divergence is unexplained, and it is not what I first guessed

I said the cause was probably `grid_sample`'s non-deterministic input gradient,
which this session established for a different arm.  That is now **withdrawn**:

```
60 steps, same process, twice        final loss 8.8777322769 == 8.8777322769
60 steps, separate processes, twice  8.877732276917 == 8.877732276917
runner file since 620bda9            no diff
imported modules since 620bda9       no diff
model                                no convolution carries a gradient here;
                                     only Linear, LayerNorm, MultiheadAttention
```

The trajectory is bit-reproducible at 60 steps across processes, the code did
not change, and the one mechanism I proposed is ruled out.  The 16-21% spread at
1,500 and 3,000 is therefore **not explained**, and I am not going to invent a
second mechanism for it.  A plausible untested candidate is reduction-order
drift in the attention matmuls accumulating over thousands of steps, which 60
steps would not reveal.

What this needs is a run-to-run variability measurement -- the same trajectory
repeated N times, spread reported at each mark -- and that is a separate lock,
not something to bolt onto this screen.

## A real defect found while checking

`DirectHoughModel` constructs `self.position = ARCH.AbsoluteXY()` and never
calls it.  An AST check of `forward` and `descriptors` confirms it:

```
constructed   encoder, head, position
referenced    encoder, head
never used    position
```

The XY signal is not absent -- the encoder builds its tokens from
`F50 concat (x, y)`, which is where the coordinate information enters, and the
`O_SCORER`/`O_DOMAIN` results are unaffected because neither uses this class.
But the additive `AbsoluteXY` branch that carried a 20.1% effect in the map
family is dead weight here, and every number above was produced without it.
That is recorded, not patched: changing the architecture now would invalidate the
comparison this screen was locked to make.

## Standing

```
DIRECT_HOUGH_OVERFIT_EXTENDED_PASS      at 6,000, 4/4, margin 1.8-3.0x
DIRECT_HOUGH_NETWORK_FIT_FAIL           af3e8cf, unchanged, at 3,000
OVERFIT_STEP_BUDGET_INSUFFICIENT        NOT SUPPORTED
run-to-run variability                  UNMEASURED, and it straddles the gate
                                        at 3,000
AbsoluteXY branch in DirectHoughModel   dead, recorded not patched
FULL                                    running under the extension's own
                                        eligibility; af3e8cf is not overwritten
```

No PnP, no CIGM, no dimensions.  `untouched`, `eval56`, `wood45` and final-test
remain unopened.
