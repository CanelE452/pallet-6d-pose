# Clean-start 60k architecture extension

Exploratory frozen-cache DEV only. This is not an independent TEST and must not be copied into a paper FINAL table.

| Arm | DEV balanced accuracy | Shuffled dimensions | Mean +/- sample SD (3 seeds) | Params | Delta vs S1 (95% cluster CI) |
|---|---:|---:|---:|---:|---:|
| S0_SPATIAL_NO_DIMS | 0.855254 | -- | 0.854394 +/- 0.002840 | 32034 | -0.048886 [-0.059274, -0.038527] |
| S1_SPATIAL_CONCAT | 0.904141 | 0.745329 | 0.904170 +/- 0.001722 | 32034 | reference |
| S2_SPATIAL_FILM | 0.903476 | 0.749569 | 0.903881 +/- 0.000734 | 42802 | -0.000665 [-0.007461, +0.006133] |
| S3_SPATIAL_CROSS_ATTENTION | 0.907430 | 0.740492 | 0.906935 +/- 0.001509 | 42722 | +0.003289 [-0.003923, +0.010615] |
| D0_DIMS_ONLY | 0.521393 | 0.508000 | 0.522135 +/- 0.001585 | 4834 | -0.382748 [-0.397853, -0.367395] |
| K0_KP_ONLY | 0.832364 | -- | 0.832241 +/- 0.001176 | 7650 | -0.071777 [-0.083515, -0.059893] |
| C1_CENTER_CONCAT | 0.891059 | 0.745418 | 0.890921 +/- 0.000695 | 32034 | -0.013082 [-0.021252, -0.005112] |

P0 and TEX rows with the same `pair_group_id` are resampled as one paired scenario cluster; each G38 `pair_group_id` is a singleton cluster.
