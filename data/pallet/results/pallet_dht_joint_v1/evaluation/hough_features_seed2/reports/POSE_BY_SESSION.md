# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       0.743       0.777
eval_night08         12       0.478       0.266
eval_night09         16       0.508       0.470
eval_noapril         12       0.675       0.750
eval_outside         10       0.621       0.666
eval_pallet07        27       0.661       0.647
eval_pallet09        33       0.429       0.457
plastic_day_01       44       0.506       0.565
plastic_night_01     22       0.597       0.566
wood_183705          25       0.790       0.779
wood_184309          20       0.691       0.773
wood_day_01          24       0.559       0.565
wood_night_01        56       0.532       0.519
───────────────────────────────────────────────
ALL                 319       0.603       0.608
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       0.803       0.812
eval_night08         12       0.158       0.116
eval_night09         16       0.228       0.185
eval_noapril         12       0.624       0.716
eval_outside         10       0.292       0.310
eval_pallet07        27       0.469       0.481
eval_pallet09        33       0.143       0.137
plastic_day_01       44       0.232       0.294
plastic_night_01     22       0.403       0.403
wood_183705          25       0.695       0.683
wood_184309          20       0.519       0.649
wood_day_01          24       0.367       0.374
wood_night_01        56       0.291       0.287
───────────────────────────────────────────────
ALL                 319       0.428       0.442
```

## translation median [cm]  (lower is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18        2.17        2.21
eval_night08         12       19.54       34.59
eval_night09         16       16.13       23.23
eval_noapril         12        6.08        4.23
eval_outside         10       11.54       10.63
eval_pallet07        27        7.48        7.37
eval_pallet09        33       19.16       21.87
plastic_day_01       44       12.68       10.62
plastic_night_01     22        9.47        8.02
wood_183705          25        1.59        1.93
wood_184309          20        3.63        1.99
wood_day_01          24        5.23        5.50
wood_night_01        56        7.84        8.60
───────────────────────────────────────────────
ALL                 319        7.90        7.66
```

## axis accuracy  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.417
eval_night09         16       0.625       0.562
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.500
eval_pallet07        27       0.630       0.704
eval_pallet09        33       0.303       0.303
plastic_day_01       44       0.795       0.841
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.880
wood_184309          20       0.900       1.000
wood_day_01          24       0.875       0.917
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.746
```

