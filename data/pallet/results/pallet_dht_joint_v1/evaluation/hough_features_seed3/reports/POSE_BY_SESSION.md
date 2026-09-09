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
eval_cad             18       0.743       0.742
eval_night08         12       0.478       0.370
eval_night09         16       0.508       0.355
eval_noapril         12       0.675       0.663
eval_outside         10       0.621       0.707
eval_pallet07        27       0.661       0.646
eval_pallet09        33       0.429       0.430
plastic_day_01       44       0.506       0.572
plastic_night_01     22       0.597       0.598
wood_183705          25       0.790       0.792
wood_184309          20       0.691       0.747
wood_day_01          24       0.559       0.568
wood_night_01        56       0.532       0.471
───────────────────────────────────────────────
ALL                 319       0.603       0.609
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       0.803       0.801
eval_night08         12       0.158       0.123
eval_night09         16       0.228       0.272
eval_noapril         12       0.624       0.653
eval_outside         10       0.292       0.318
eval_pallet07        27       0.469       0.471
eval_pallet09        33       0.143       0.146
plastic_day_01       44       0.232       0.290
plastic_night_01     22       0.403       0.424
wood_183705          25       0.695       0.652
wood_184309          20       0.519       0.619
wood_day_01          24       0.367       0.363
wood_night_01        56       0.291       0.269
───────────────────────────────────────────────
ALL                 319       0.428       0.434
```

## translation median [cm]  (lower is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18        2.17        1.66
eval_night08         12       19.54       28.79
eval_night09         16       16.13       13.83
eval_noapril         12        6.08        5.54
eval_outside         10       11.54       10.42
eval_pallet07        27        7.48        7.57
eval_pallet09        33       19.16       20.27
plastic_day_01       44       12.68       10.88
plastic_night_01     22        9.47        8.54
wood_183705          25        1.59        1.72
wood_184309          20        3.63        2.07
wood_day_01          24        5.23        4.70
wood_night_01        56        7.84        9.18
───────────────────────────────────────────────
ALL                 319        7.90        7.86
```

## axis accuracy  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.500
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.424
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.950
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.714
───────────────────────────────────────────────
ALL                 319       0.749       0.752
```

