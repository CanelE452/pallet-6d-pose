# The first O_MAP decoded a map the network never produces

`20eb590` and `548585b` stand unedited.  The P0 numbers are real; what they are
numbers *about* was not the locked decoder.

## The mismatch

`batch_terms` does two different things with one logit:

```
probability = sigmoid(logit)      supervised against the raster target
weight      = softplus(logit)     handed to weighted_tls
```

So "the network predicts a perfect map" means `sigmoid(logit) == target`, and
the readout then receives

```
softplus(logit(target)) = -log1p(-target)
```

`run_omap` called `weighted_tls(target)` instead.  That decodes the target as if
it were the weight, which no forward pass ever produces.

```
OMAP_ORACLE_SEMANTICS_MISMATCH
```

The identity is exact to `9.5e-7` over `1e-6 < p < 1-1e-6`, and a test pins it
below `1e-6`.

## Why it is not a cosmetic difference

The transform is strongly expansive near the ridge and nearly linear in the
skirt:

```
target   1.0      0.999   0.9     0.5     0.1     0.01    0
weight  15.94     6.91    2.30    0.69    0.105   0.010   0 (exact)
```

The tube's ridge is exactly 1, so the parity weight concentrates far more mass
on the line itself and comparatively less on the skirt that the image boundary
can cut.  Whether that is enough to change the verdict is the measurement, not
an assumption.

The clamp is `1 - eps` for the tensor's own dtype, never a chosen cap; the
transform diverges as the target approaches 1.  `target == 0` stays exactly 0,
matching a perfect prediction there (`logit = -inf`, `softplus = 0`).

## What is preserved and what changed

```
preserved   weighted_tls, raster target, sigma, grid size, gates,
            sign alignment, population, LINE_DEV split
changed     the oracle's input construction only
```

`run_omap` gained a `parity` flag that steers one conditional expression.  A
test asserts there is no statement-level branch on it, so the two oracles cannot
diverge anywhere else.

## Naming

```
P0   O_MAP_P0_TARGET_AS_WEIGHT      historical, 548585b, kept
     LINE_DEV 2,393 frames, 27,684 supported roles
     angle med 0.0015  p90 0.3780   offset med 0.0006  p90 0.1380
     -> TARGET_AS_WEIGHT_TLS_FAIL

P1   forward-parity oracle          structural_line_map_omap_parity.json
     PASS -> LOCKED_SOFTPLUS_TLS_VALID
     FAIL -> LOCKED_SOFTPLUS_TLS_FAIL / MAP_TO_LINE_DECODER_FAIL_CONFIRMED
```

`MAP_TO_LINE_DECODER_FAIL` from `548585b` stays as a historical protocol result.
Training is now gated on the parity file, because the P0 oracle cannot speak for
a decoder it did not simulate.

## Units

Bare "cell" is retired from these reports.

```
sigma              1.5 MAP100 pixel = 0.75 canonical50 cell
border threshold   1.5 canonical50 cell = 3 MAP100 pixel = 2 sigma
visible threshold  2.0 canonical50 cell = 4 MAP100 pixel
angle              degree
offset             canonical50 cell
```

## One claim withdrawn

`548585b` said the border was the cause of the tail.  With `visible < 2
canonical50 cell` sitting at angle p90 0.369 degrees, that was stated more
strongly than a one-way split supports.  P1 reports a border x visible-length
cross-tab so the two are separated rather than asserted.
