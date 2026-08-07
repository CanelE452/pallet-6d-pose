# V2 O0: the refiner is identifiable

Before judging any feature, V2 asks whether the refiner can represent and learn
the correction at all.  Evidence is perfect: the twelve ground-truth lines
rasterised as an anti-aliased likelihood map, with the same coarse jitter the
real arms will see -- +/-8 degrees, +/-4 cells.

```
GT_LINE_COVERED   100.00%        gate >= 99.5%
overfit32         1,000 steps, 32 frames, 384 valid roles

               median      p90       gate
angle          0.0149 deg  0.0516    <= 0.20 / 0.50
offset         0.0415 cell 0.0648    <= 0.10 / 0.25

O0_PASS
```

Angle lands at a sixtieth of the 1.0 degree budget and offset at a twelfth of the
0.5 cell budget.  The architecture, the sampler, the strip width and the
budget-normalised loss are all sufficient when the evidence is there.

## What this settles about V1

V1's `FROZEN_FEATURE_LINE_CAPACITY_FAIL` cannot stand on the same refiner.  The
identical architecture reaches 0.015 degrees here, so V1's 3.8 degrees was not
the refiner being unable to express a correction.  With O0 passing, V1's four
confounds -- the offset-dominated loss, the origin-anchored sampler, the strip
narrower than the jitter, and the unpreserved runner -- are the live explanations,
and the feature question is genuinely open again rather than answered.

Note also `GT_LINE_COVERED = 100%` under the V2 strip radius of 6 cells.  Under
V1's radius of 3, offset jitter of up to 4 cells put the true line outside the
sampled strip on a substantial share of draws, so part of V1's training signal
was asking for a correction whose evidence had not been sampled.

## Next

O1, the deterministic image-gradient control, then the four real feature arms at
one, three and five epochs.  Gates unchanged: angle median <= 1.0 degree, offset
median <= 0.5 cell, with p90 <= 2.0 and 1.0 as safety diagnostics.

No PnP, no dimensions, no validation512.  untouched, eval56, wood45 and
final-test unopened.  No SLQ predictor written.
