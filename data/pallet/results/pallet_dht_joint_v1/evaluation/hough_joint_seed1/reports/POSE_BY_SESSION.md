# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       0.743       0.765
eval_night08         12       0.478       0.400
eval_night09         16       0.508       0.458
eval_noapril         12       0.675       0.633
eval_outside         10       0.621       0.606
eval_pallet07        27       0.661       0.541
eval_pallet09        33       0.429       0.420
plastic_day_01       44       0.506       0.494
plastic_night_01     22       0.597       0.614
wood_183705          25       0.790       0.754
wood_184309          20       0.691       0.674
wood_day_01          24       0.559       0.558
wood_night_01        56       0.532       0.545
───────────────────────────────────────────────
ALL                 319       0.603       0.585
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       0.803       0.793
eval_night08         12       0.158       0.200
eval_night09         16       0.228       0.194
eval_noapril         12       0.624       0.587
eval_outside         10       0.292       0.316
eval_pallet07        27       0.469       0.385
eval_pallet09        33       0.143       0.134
plastic_day_01       44       0.232       0.192
plastic_night_01     22       0.403       0.387
wood_183705          25       0.695       0.635
wood_184309          20       0.519       0.493
wood_day_01          24       0.367       0.342
wood_night_01        56       0.291       0.332
───────────────────────────────────────────────
ALL                 319       0.428       0.411
```

## translation median [cm]  (lower is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18        2.17        2.15
eval_night08         12       19.54       17.98
eval_night09         16       16.13       18.14
eval_noapril         12        6.08        6.65
eval_outside         10       11.54       10.65
eval_pallet07        27        7.48        8.60
eval_pallet09        33       19.16       22.42
plastic_day_01       44       12.68       13.44
plastic_night_01     22        9.47        8.03
wood_183705          25        1.59        2.25
wood_184309          20        3.63        3.42
wood_day_01          24        5.23        5.54
wood_night_01        56        7.84        6.58
───────────────────────────────────────────────
ALL                 319        7.90        8.07
```

## axis accuracy  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.500
eval_night09         16       0.625       0.500
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.556
eval_pallet09        33       0.303       0.424
plastic_day_01       44       0.795       0.795
plastic_night_01     22       0.864       0.773
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.850
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.768
───────────────────────────────────────────────
ALL                 319       0.749       0.730
```

