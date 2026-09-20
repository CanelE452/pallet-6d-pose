# Table 2. Downstream 6D pose

| Model | Rotation median (deg) | Yaw median (deg) | Translation median (cm) | Oriented IoU3D median | ADD-sym AUC | Pose coverage (%) |
|---|---|---|---|---|---|---|
| R0 | 2.5389 | 1.3162 | 7.8969 | 0.594286 | 0.376603 | 100.00 |
| OLD_P | 2.1542 | 1.1511 | 7.1530 | 0.636719 | 0.408006 | 100.00 |
| N2_DIM_ONLY | 2.0897 | 1.1424 | 7.0106 | 0.633525 | 0.412592 | 100.00 |
| N3_DIM_SYM | 2.0704 | 1.1340 | 7.0676 | 0.630878 | 0.412193 | 100.00 |

Downstream pose on the same reused DEV319, with the current common DCP prediction-only selector and SQPnP/RefineLM geometry. Refiners: three-seed mean of per-seed statistics. The real reference is a geometry-reconstructed 6D reference, not independent physical GT. Conditional pose medians use available predictions; ADD-sym AUC uses all 319 frames (failures count as infinity), whole-object C2 corresponding-corner ADD normalized by object diameter, 1,001 thresholds from 0 to 0.1. PnP is not gated by GT box matching.
