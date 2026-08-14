# What the no-response frames have in common

13 matched pairs.  Matching rule, fixed before any response was inspected:
same domain, same session preferred, then nearest |log bbox area ratio|, greedy
without replacement, ordered by centroid peak.  Thirteen pairs is a description,
not a test -- no p-values are quoted.

```
              field  no-response med  control med  paired delta med  dead > control
───────────────────────────────────────────────────────────────────────────────────
           luma_p10           23.000       23.000            +0.000            5/13
           luma_p50           48.000       51.000            -3.000            5/13
           luma_p90          117.000      135.000           -11.000            4/13
         blur_score         1059.917     1091.603          -116.627            3/13
    bbox_area_ratio            0.209        0.120            +0.116           10/13
         bbox_width          540.000      402.001          +137.999           11/13
        bbox_height           98.000       89.000           +16.198            8/13
         distance_m            1.920        2.392            -0.402            2/13
        azimuth_deg           -4.876        2.250            -5.398            4/13
       n_gt_inframe            6.000        8.000            -2.000            0/13
         n_gt_valid            7.000        8.000            -1.000            0/13
border_proximity_px          -32.809       63.000          -108.809            1/13
```

Reading down the column that matters:

```
truncated              10 of 13 no-response   vs   1 of 13 control
in-frame GT corners     6 (median)            vs   8;  0 of 13 pairs has dead > control (3 tied)
border proximity      -32.8 px                vs  +63.0 px;  12 of 13 pairs closer to the edge
bbox width            540 px                  vs 402;  11 of 13 larger
distance             1.92 m                   vs 2.39;  11 of 13 nearer
luma p10               23                     vs  23;  no separation
blur score           1060                     vs 1092;  no separation
```

A negative border proximity means the bounding box runs past the frame edge.

**These are not dark frames.**  The tenth-percentile luma is identical and the
median differs by 3 grey levels; the blur score is if anything slightly better on
the controls.  The domain split (8 night, 5 outside) is a session artefact, not
an illumination effect.

**They are near, large, truncated pallets.**  The overlay in
`figures/nrf_examples.png` shows it directly: several GT corners sit outside the
image, and the pallet is cut by the frame edge and heavily occluded.

This is the failure mode this programme has already characterised twice -- the
V<8 truncated population and the near-face border cut -- arriving here as the
thing that blocks deployment compatibility.
