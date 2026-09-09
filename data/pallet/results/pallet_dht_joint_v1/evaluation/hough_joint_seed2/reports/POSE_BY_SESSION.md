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
eval_cad             18       0.743       0.697
eval_night08         12       0.478       0.329
eval_night09         16       0.508       0.483
eval_noapril         12       0.675       0.707
eval_outside         10       0.621       0.635
eval_pallet07        27       0.661       0.678
eval_pallet09        33       0.429       0.407
plastic_day_01       44       0.506       0.558
plastic_night_01     22       0.597       0.542
wood_183705          25       0.790       0.809
wood_184309          20       0.691       0.795
wood_day_01          24       0.559       0.623
wood_night_01        56       0.532       0.509
───────────────────────────────────────────────
ALL                 319       0.603       0.595
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       0.803       0.756
eval_night08         12       0.158       0.113
eval_night09         16       0.228       0.161
eval_noapril         12       0.624       0.625
eval_outside         10       0.292       0.312
eval_pallet07        27       0.469       0.471
eval_pallet09        33       0.143       0.140
plastic_day_01       44       0.232       0.285
plastic_night_01     22       0.403       0.425
wood_183705          25       0.695       0.685
wood_184309          20       0.519       0.632
wood_day_01          24       0.367       0.406
wood_night_01        56       0.291       0.308
───────────────────────────────────────────────
ALL                 319       0.428       0.433
```

## translation median [cm]  (lower is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18        2.17        3.05
eval_night08         12       19.54       26.20
eval_night09         16       16.13       27.89
eval_noapril         12        6.08        4.80
eval_outside         10       11.54       10.58
eval_pallet07        27        7.48        6.83
eval_pallet09        33       19.16       22.91
plastic_day_01       44       12.68       11.47
plastic_night_01     22        9.47        7.06
wood_183705          25        1.59        1.80
wood_184309          20        3.63        2.45
wood_day_01          24        5.23        4.04
wood_night_01        56        7.84        8.11
───────────────────────────────────────────────
ALL                 319        7.90        7.76
```

## axis accuracy  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.562
eval_noapril         12       1.000       0.917
eval_outside         10       0.700       0.500
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.394
plastic_day_01       44       0.795       0.841
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.917
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.749
```

