# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       0.743       0.705
eval_night08         12       0.478       0.341
eval_night09         16       0.508       0.506
eval_noapril         12       0.675       0.590
eval_outside         10       0.621       0.645
eval_pallet07        27       0.661       0.637
eval_pallet09        33       0.429       0.456
plastic_day_01       44       0.506       0.566
plastic_night_01     22       0.597       0.622
wood_183705          25       0.790       0.758
wood_184309          20       0.691       0.589
wood_day_01          24       0.559       0.501
wood_night_01        56       0.532       0.458
───────────────────────────────────────────────
ALL                 319       0.603       0.586
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       0.803       0.774
eval_night08         12       0.158       0.152
eval_night09         16       0.228       0.221
eval_noapril         12       0.624       0.505
eval_outside         10       0.292       0.287
eval_pallet07        27       0.469       0.455
eval_pallet09        33       0.143       0.139
plastic_day_01       44       0.232       0.297
plastic_night_01     22       0.403       0.442
wood_183705          25       0.695       0.612
wood_184309          20       0.519       0.442
wood_day_01          24       0.367       0.330
wood_night_01        56       0.291       0.276
───────────────────────────────────────────────
ALL                 319       0.428       0.413
```

## translation median [cm]  (lower is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18        2.17        2.43
eval_night08         12       19.54       34.48
eval_night09         16       16.13       15.38
eval_noapril         12        6.08        6.65
eval_outside         10       11.54       11.39
eval_pallet07        27        7.48        7.07
eval_pallet09        33       19.16       18.06
plastic_day_01       44       12.68       10.31
plastic_night_01     22        9.47        7.40
wood_183705          25        1.59        2.05
wood_184309          20        3.63        4.40
wood_day_01          24        5.23        5.38
wood_night_01        56        7.84        8.26
───────────────────────────────────────────────
ALL                 319        7.90        7.86
```

## axis accuracy  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.812
eval_noapril         12       1.000       0.833
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.333
plastic_day_01       44       0.795       0.750
plastic_night_01     22       0.864       0.909
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.750
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.768
───────────────────────────────────────────────
ALL                 319       0.749       0.734
```

