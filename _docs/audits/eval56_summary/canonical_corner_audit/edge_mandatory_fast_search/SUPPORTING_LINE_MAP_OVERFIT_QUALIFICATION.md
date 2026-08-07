# What the overfit failure did and did not establish

`7c6602a` stands unedited.  Its verdict is kept as a historical result against
the gate it was declared for.

```
historical
  SUPPORTING_LINE_MAP_OPTIMIZATION_FAIL
  condition = the ultra-tight overfit32 gate (0.10 deg / 0.05 cell median,
              0.25 / 0.15 p90), set an order of magnitude inside the budget

confirmed observation
  M0 and M1 both land inside the actual primary task budget on those same
  32 frames -- 0.1268 / 0.0586 and 0.1426 / 0.0614 against 1.0 / 0.5 --
  and clear its safety and approach thresholds

NOT established
  NETWORK_OPTIMIZATION_INCAPABLE
  SUPPORTING_LINE_MAP_CAPACITY_FAIL
```

Missing a sanity gate that sits ten times inside the budget is not evidence that
the model cannot optimise.  The screen never asked the budget question, because
the protocol blocked `search2k` before it could.

```
open question
  CAN_THE_LOCKED_MAP_MODEL_GENERALIZE_TO_1DEG_0P5CELL
```

## Two diagnostic sentences withdrawn

Both came from my own report and both overclaim.

**"map NCC 0.77-0.79, so roughly a fifth of the map structure is wrong."**  That
is not what a correlation coefficient says.  NCC is invariant to scale and
offset and mixes ridge placement, ridge width and background texture into one
number; it does not partition the map into right and wrong fractions.  Correct
reading: *NCC is a correlation diagnostic only*, useful for comparing arms and
epochs against each other and not for attributing a share of error.

**"peak probability is 0.98, so the head is confident and the ridge is there."**
A high maximum says some pixel is confident.  It says nothing about whether that
pixel lies on the correct line, nor whether the ridge is one line or several.
Peak probability alone is not evidence of a correct ridge and is not used as
such here.

The accurate reading of the overfit diagnostics is exactly this and no more:

```
the probability has a strong peak
positive-region MSE > negative-region MSE
the decoded line is nevertheless inside the task budget on memorised data
therefore the precision of the line-region prediction is the live issue
```

## One inference withdrawn

**"The RGB stem does not help."**  M1 was slightly worse than M0 on 32 frames it
was asked to memorise.  A trainable stem adds parameters that help most where
generalisation matters and can easily cost a little on pure memorisation, so an
overfit comparison cannot rule out an RGB adaptation benefit.  The ordering is
recorded; the conclusion is not drawn.

## Standing

`O_LOSS` passes, the decoder is settled by `f5ac650`, and the population is the
locked 27,684 roles at sha `00c605b9116e214b`.  The architecture, target, loss
and decoder are unchanged.  `STRUCTURAL_LINE_MAP_CAPACITY` remains unmeasured
and the overfit gate is not moved.
