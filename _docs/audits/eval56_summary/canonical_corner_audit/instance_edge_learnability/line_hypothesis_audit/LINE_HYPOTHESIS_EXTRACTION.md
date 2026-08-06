# Deterministic hypothesis extraction

The policy was chosen on synthetic validation alone and then frozen; canonical results never touched it.

```
{
 "extractor": "E1_COMPONENT_TLS",
 "parameter": 0.9,
 "top_k": 5,
 "selection_set": "synthetic validation",
 "selection_arm": "L12-MS seed2",
 "selection_frames": 300,
 "primary_metric": "edge top-5 strict infinite-line availability",
 "tie_breaks": [
  "corner triplet availability",
  "oracle-selected corner <=20px",
  "simplest extractor E1 > E2 > E3",
  "threshold 0.5 > 0.7 > 0.3 > 0.9"
 ],
 "untouched_stride": 4,
 "constants": {
  "strict_angle_deg": 5.0,
  "strict_offset_cells": 3.0,
  "loose_angle_deg": 10.0,
  "loose_offset_cells": 5.0,
  "strict_overlap": 0.5,
  "loose_overlap": 0.25,
  "hough_support_cells": 1.5,
  "condition_max": 1000.0
 },
 "policy_sha256": "b36610f80aa1f1eb292b175cd7dbb98a497330e222529802a196133b931b638c"
}
```

## every predeclared arm

```
        extractor  parameter  edge_top5_strict_line  corner_triplet_top5  oracle_corner_le20  edges  corners  seconds
 E1_COMPONENT_TLS        0.3               0.269167             0.041250            0.979798   3600     2400      1.4
 E1_COMPONENT_TLS        0.5               0.407222             0.097083            0.948498   3600     2400      1.3
 E1_COMPONENT_TLS        0.7               0.510833             0.147500            0.963277   3600     2400      1.3
 E1_COMPONENT_TLS        0.9               0.621944             0.209583            0.972167   3600     2400      1.2
E2_WEIGHTED_HOUGH        0.3               0.270278             0.006667            0.687500   3600     2400      5.2
E2_WEIGHTED_HOUGH        0.5               0.302222             0.010833            0.730769   3600     2400      4.5
E2_WEIGHTED_HOUGH        0.7               0.352500             0.015417            0.783784   3600     2400      4.1
E2_WEIGHTED_HOUGH        0.9               0.480833             0.034583            0.915663   3600     2400      3.2
  E3_TOP_MASS_TLS        1.0               0.079167             0.000833            1.000000   3600     2400      1.4
  E3_TOP_MASS_TLS        2.0               0.028889             0.000000            0.000000   3600     2400      1.8
  E3_TOP_MASS_TLS        5.0               0.013889             0.000000            0.000000   3600     2400      2.0
  E3_TOP_MASS_TLS       10.0               0.010000             0.000000            0.000000   3600     2400      2.1
```
