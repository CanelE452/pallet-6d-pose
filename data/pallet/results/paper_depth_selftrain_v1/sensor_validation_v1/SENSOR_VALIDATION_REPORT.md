# Sensor validation — does the stored depth agree with manual RGB geometry

Gate 0B could not settle how the depth was recorded, because no writer for the
layout exists anywhere. Where the repository holds manual keypoints, known
dimensions, depth and an intrinsic on the same frame, that question can be asked of
the data instead of the records. No model prediction is used here.

```text
FINAL = NOT_READY_FOR_GATE1
```

## Population

```text
frames                    506
measured                  506
recordings                18
day / night               390 / 116
plastic / wood            506 / 0
overlaps PAPER_EVAL       116
```

Frames were found by hashing every raw image and matching content, not by reading
session names — the import manifest records the annotation folder as the source for
most frames, so a path-based search finds only a fifth of them.

This population is development sensor validation. It must never be reused later as
an independent confirmation that the method works; that would be circular.

## Reference quality

```text
subset                      n  ref reproj px  surface cm   ring cm  face<ring
─────────────────────────────────────────────────────────────────────────────
all                       506           8.16         2.8      10.9      0.935
excluding pallet11        263           1.61         2.6      10.7      0.981
```

`capturepallet11` supplies 243 of the 506 frames and its reference reprojects at 22.8
pixels against 1.6 elsewhere. That is `pallet11_gt`, recorded as apriltag-broken and
not-to-be-used long before this measurement. It is separated, not deleted, and both
views are shown.

## Per recording

```text
recording                 n   ref px   valid  surface cm   ring cm  face<ring  edge px
──────────────────────────────────────────────────────────────────────────────────────
capturenight01            6     0.75   0.989         2.5      15.3      1.000     51.5
capturenight02           14     1.68   0.993         1.8       6.5      1.000     62.1
capturenight03           20     0.37   0.990         2.6      16.7      1.000     19.2
capturenight04            5     0.82   0.983         3.7       6.0      1.000     76.5
capturenight05           12     1.32   0.991         2.9      10.5      1.000    200.5
capturenight06           15     3.60   0.992         4.3      26.7      0.933     83.8
capturenight07           16     4.40   0.993         9.5      33.3      0.875    128.5
capturenight08           12     1.55   0.995         2.2      12.8      1.000    187.6
capturenight09           16     1.93   0.996         2.1       8.4      1.000     57.2
capturepallet02           7     1.28   1.000         2.7      13.7      1.000     42.2
capturepallet03           9     1.53   0.999         5.8      19.6      1.000     40.9
capturepallet04           6     3.54   0.990        36.5      39.7      1.000     41.0
capturepallet05           7     1.01   0.999         5.0      23.0      1.000     41.4
capturepallet07          27     2.34   0.995         1.3       4.9      1.000     46.0
capturepallet08          25     1.29   0.921         3.1      38.6      1.000     10.2
capturepallet09          33     1.12   0.978         2.6       8.9      0.970     24.8
capturepallet11         243    22.80   0.964         3.1      11.4      0.885     26.8
capturepalletcad         33     2.37   0.999         1.3       2.6      0.970      6.1
```

## What the numbers say

**Geometry.** Backprojected depth points sit 2.6 cm from a cuboid placed by
manual keypoints and registered dimensions alone. On an object 1.1 by 1.3 by 0.15
metres seen at a few metres, that is agreement. The ray-to-surface depth residual is
much larger at 15 cm, because it is sensitive to reference pose error at grazing
incidence; both are reported and neither was tuned.

**Scale.** Only 0.001 was evaluated. Nothing was fitted or swept. A wrong factor
would misplace the depth by metres against a reference built purely from image
geometry and known dimensions; the residual is centimetres.

**Intrinsics.** The day and night groups give 2.5 and 2.7 cm. Each is
empirically compatible with its own recordings' colour geometry. That is the
strongest permitted statement — no acquisition record exists to confirm the stream.

**Discrimination.** This answers what Gate 0B left open. The face interior is
explained by the known cuboid about four times better than the ring around it, in
98.1% of clean-reference frames. The good plane is the pallet, not the ground.

## Verdict

```text
RGB-depth geometry                SUPPORTED
K compatibility                   DAY_SUPPORTED, NIGHT_SUPPORTED
Scale                             SUPPORTED
Pallet vs background              SUPPORTED

condition 1  depth compatible with manual RGB geometry        MET
condition 2  frozen scale not contradicted                     MET
condition 3  pallet better explained than background           MET
condition 4  no repeated systematic boundary displacement      NOT ESTABLISHED

FINAL                             NOT_READY_FOR_GATE1
```

three of the four preregistered conditions are met, and met convincingly. The fourth is not met, and not because evidence runs against it — the boundary measurement as specified cannot answer it, since the manual hull includes an edge that touches the ground where no depth discontinuity exists. Gate 0B's shift analysis is the other half of that question and it was graded UNCLEAR. Declaring READY would mean treating an unanswerable measurement as a pass.

The boundary distance is 35 px overall and 85 px at night, but that number cannot
carry the conclusion: it measures distance rather than displacement, and the manual
hull includes the bottom edge resting on the ground, where no depth discontinuity can
exist. A large distance is expected there whatever the alignment is.

What would settle it:

- restrict the boundary comparison to the pallet's upper silhouette, where a depth step must exist, instead of the full hull
- or measure displacement direction rather than distance, and test whether it is consistent across frames

Nothing here says depth correction would improve accuracy. Three of four legs now
stand on measurement rather than assumption, which is more than Gate 0B had.

`NEXT_ACTION = USER_REVIEW_SENSOR_VALIDATION`

