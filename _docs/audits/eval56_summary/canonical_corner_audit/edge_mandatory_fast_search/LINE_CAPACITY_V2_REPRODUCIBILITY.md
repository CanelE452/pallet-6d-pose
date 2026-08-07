# V2 is now runnable from the repository, and the strip was too narrow

## The defect this fixes

`9dfa414` claimed a committed script.  It committed a helper module -- sampler,
refiner, losses, raster -- with no loader, no jitter, no training loop, no
evaluation and no `main`.  O0 ran from a scratchpad heredoc: the same failure the
V1 addendum had named one commit earlier.

`scripts/stage0/line_feature_capacity_v2.py` now carries the execution path:
`audit`, `o0`, `o1`, `features`, `all`, with the split sha checked at entry, a
sealed-set guard on every read, deterministic jitter, the epoch ladder,
checkpointing and JSON output.

```
O0_PREVIOUS   HISTORICAL_RESULT   0.0149 deg / 0.0415 cell   (heredoc)
O0_REPRO_RUN  committed runner    0.0228 deg / 0.0674 cell   PASS
```

Not required to match; the gate is what must pass, and it does with room.

## Jitter is addressed, not drawn

Per `(frame_uid, role, epoch, purpose)` through sha256, so a sample's
perturbation does not depend on batch order, worker scheduling or how many
samples preceded it.  DEV uses one fixed set, so epoch 1, 3 and 5 checkpoints are
compared on identical perturbations.

## The strip was too narrow -- again

The V1 addendum said radius 3 was narrower than the 4-cell offset jitter.  V2
set 6 on that reasoning, and the coverage audit shows 6 was still wrong.

```
required radius to cover 90% of a GT line's points
median 2.47   p95 6.67   p99 8.68   max 12.53 cell

pair coverage    radius 6   92.48%
                 radius 8   98.27%
                 radius 10  99.67%     <- smallest clearing the 99.5% gate
                 radius 12  99.97%
```

"Wider than the offset jitter" was the wrong argument.  Eight degrees of angular
drift over a long chord displaces the far end by much more than the offset, so
the requirement scales with edge length, not with the jitter alone.  At radius 6
the first percentile of point coverage was 0.0 -- some frame-role pairs had **no**
part of their true line inside the sampled strip, and the loss was still asking
for that correction.

Radius is now 10 with 21 transverse samples, chosen from the split's own geometry
before any feature was judged, and O0 was rerun because the sampler changed.

```
coverage after   dev 0.9967   train 0.9977   gate 0.995
```

## Standing

O1A, O1B and the four feature arms are not yet run.  No SLQ predictor exists.  No
PnP, no dimensions, no validation512; untouched, eval56, wood45 and final-test
unopened.
