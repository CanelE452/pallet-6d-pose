# Table 1. Main 2D keypoint refinement

| Model | Matched corner median (px) | Matched P90 (px) | Full PCK10 (%) | Full E_sym | Detected | Matched IoU>=0.5 |
|---|---|---|---|---|---|---|
| R0 | 6.7207 | 43.8900 | 63.425 | 0.04952390 | 319/319 | 311/319 |
| OLD_P | 5.9381 | 42.6314 | 67.494 | 0.04868357 | 319/319 | 311/319 |
| N2_DIM_ONLY | 5.7777 | 42.4595 | 68.587 | 0.04842188 | 319/319 | 311/319 |
| N3_DIM_SYM | 5.7782 | 42.1338 | 68.587 | 0.04841909 | 319/319 | 311/319 |

Main 2D refinement on reused C2 DEV319. Refiners: mean of three per-seed statistics; R0: frozen baseline. Median/P90 pool 2,445 observed matched corners. PCK10 includes all 2,499 valid corners; unmatched/missing predictions incur image-diagonal error. E-sym is the frame-mean diagonal-normalized error over all 319 frames. One whole-object symmetry branch per frame, eight corners only; center excluded. Detection and matching coverage are distinct.

Full-penalty median/P90 are also retained per arm in TABLES.json; they are not substituted into the matched columns.
