# GATE 0 — RGB-D sensor contract audit

This asks one thing: is the depth in the existing unlabelled recordings good enough
to correct the teacher's pseudo-label coordinates. Nothing was trained, no
pseudo-label was made, and no evaluation ground truth was read.

```text
GATE0 = PARTIAL
```

## Inventory

```text
recording               RGB  depth  paired   zero%   sat%  metadata files
────────────────────────────────────────────────────────────────────────────────────
capturepallet01          42     42   1.000    0.33   8.58  cam_K.txt only
capturepallet10         613    613   1.000    1.20  12.93  cam_K.txt only
capturepallet11        1572   1572   1.000    2.85  12.44  cam_K.txt only
capturenight01         1254   1254   1.000    4.72   4.18  cam_K.txt only
capturenight02          782    782   1.000    3.90   2.60  cam_K.txt only
capturenight03         1219   1219   1.000    0.69  11.16  cam_K.txt only
capturenight04         1075   1075   1.000    0.82  10.55  cam_K.txt only
capturenight10         1474   1474   1.000    3.41   6.37  cam_K.txt only
```

Every recording carries a single 3x3 intrinsic file and nothing else. There is no
session metadata, no depth intrinsic, no depth-to-colour transform and no declared
depth scale. The day recordings and the night recordings do not share the same
intrinsics, so they must not be pooled.

## Metric scale

```text
value    0.001 m per unit
source   run_live_io.NpDepthFrame — docstring 'uint16(mm)', get_distance divides by 1000
         classify_daytime_visibility.py — DEPTH_SCALE_MM = 0.001
status   consumer contract in this repository; no acquisition record states it
```

This is stronger than assuming a sensor default and weaker than a calibration file.
Between 2.6% and 12.9% of pixels sit at the 16-bit maximum. No consumer in this
repository special-cases that value, so it would convert to about 65 m if treated as
a measurement. This audit reports the fraction and invents no clipping rule.

## Pairing

```text
rate            1.000 (8031 / 8031), every recording
method          exact filename stem
timestamp delta UNMEASURABLE
```

One nanosecond stem names both files. There is no second, depth-side timestamp to
difference against, so the capture delta cannot be measured from what is stored.
Reporting it as zero would be circular.

## Depth support in the pallet ROI

```text
group                 frames  ROI valid  usable rate
────────────────────────────────────────────────────
ALL                     2727      0.973        0.998
DAY                     2227      0.950        0.999
NIGHT                    500      0.988        0.998
```

The declared criterion passes everywhere. It counts valid depth pixels; it does not
ask whether those pixels describe the pallet. They often do not.

```text
group                 ROI range m  ROI spread cm  background - object cm
────────────────────────────────────────────────────────────────────────
DAY                          5.36          461.0                    78.7
NIGHT                        3.62          124.4                    51.5
capturenight01               4.09          127.3                    87.7
capturenight02               1.74           98.4                   128.8
capturenight03               3.57           96.9                   -26.8
capturenight04               4.28         1019.7                     0.6
capturenight10               5.33          150.9                     1.5
capturepallet01             22.61         1378.9                  -565.2
capturepallet10              6.49          448.2                   100.9
capturepallet11              4.70          462.5                    75.4
```

A pallet is at most about 1.3 m deep. A daytime ROI whose depth spans 4.6 m is
therefore dominated by background, not by the object. Three night recordings
(capturenight03, 04, 10) show no usable depth step between the ROI and the ring
around it, and capturepallet01 is broken outright at a 22.6 m median range.

The visual sheets agree. Where the pallet is close and seen from above, its deck
separates cleanly in depth and its boundary lands on the RGB boundary. Where the
view is edge-on or distant, the ROI contains only a smooth ground-plane gradient.

## Alignment

```text
recording             zero-shift agree  best shift = 0  median |shift| px
─────────────────────────────────────────────────────────────────────────
capturepallet01                  0.582            0.00               14.0
capturepallet10                  0.496            0.00                9.0
capturepallet11                  0.501            0.10                6.0
capturenight01                   0.084            0.00               24.0
capturenight02                   0.169            0.00               19.0
capturenight03                   0.140            0.00               16.0
capturenight04                   0.137            0.10               11.0
capturenight10                   0.118            0.00               17.0
```

The automatic proxy rarely peaks at zero shift, which read alone would suggest a
6-24 pixel offset. It is not trusted here: the score is asymmetric, depth edges bleed
at object boundaries, and at night the RGB edge map is so sparse that the agreement
value collapses for reasons unrelated to alignment. The visual audit shows no
repeated systematic displacement. The gate forbids deciding on the automatic number
alone, so this stays UNCLEAR.

## Back-projection smoke

```text
round-trip pixel residual   ~1e-13   (numerical, as expected)
NaN or inf                  none
axis or reflection problem  none observed
z median                    3.2 - 8.3 m, plausible for these scenes
z max after excluding saturation   about 61 m, beyond any D4xx range
```

The 61 m tail means values just below saturation are already not measurements.
No threshold was invented to remove them.

## Verdict

```text
Calibration contract    PARTIAL
RGB-D pairing           PASS_ON_RATE_WITH_UNVERIFIABLE_SYNC
ROI support             PASS_ON_DECLARED_CRITERION
Alignment               UNCLEAR
FINAL                   PARTIAL
```

PARTIAL means the sensor data is not disqualified but the method cannot start as
specified. Two things are assumed rather than documented — the metric scale and the
colour alignment — and on top of that the pallet is frequently not separable from its
surroundings in depth, which is the property the whole correction idea depends on.

Nothing here says depth correction would improve accuracy.

`NEXT_ACTION = USER_REVIEW_DEPTH_GATE`

