# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       0.743       0.756
eval_night08         12       0.478       0.416
eval_night09         16       0.508       0.466
eval_noapril         12       0.675       0.647
eval_outside         10       0.621       0.643
eval_pallet07        27       0.661       0.499
eval_pallet09        33       0.429       0.409
plastic_day_01       44       0.506       0.530
plastic_night_01     22       0.597       0.648
wood_183705          25       0.790       0.731
wood_184309          20       0.691       0.715
wood_day_01          24       0.559       0.571
wood_night_01        56       0.532       0.529
───────────────────────────────────────────────
ALL                 319       0.603       0.594
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       0.803       0.754
eval_night08         12       0.158       0.149
eval_night09         16       0.228       0.241
eval_noapril         12       0.624       0.611
eval_outside         10       0.292       0.346
eval_pallet07        27       0.469       0.346
eval_pallet09        33       0.143       0.195
plastic_day_01       44       0.232       0.225
plastic_night_01     22       0.403       0.428
wood_183705          25       0.695       0.627
wood_184309          20       0.519       0.556
wood_day_01          24       0.367       0.330
wood_night_01        56       0.291       0.321
───────────────────────────────────────────────
ALL                 319       0.428       0.422
```

## translation median [cm]  (lower is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18        2.17        2.39
eval_night08         12       19.54       24.53
eval_night09         16       16.13       29.22
eval_noapril         12        6.08        5.86
eval_outside         10       11.54        9.80
eval_pallet07        27        7.48        8.55
eval_pallet09        33       19.16       21.34
plastic_day_01       44       12.68       12.65
plastic_night_01     22        9.47        7.23
wood_183705          25        1.59        2.11
wood_184309          20        3.63        2.48
wood_day_01          24        5.23        6.22
wood_night_01        56        7.84        6.75
───────────────────────────────────────────────
ALL                 319        7.90        7.70
```

## axis accuracy  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       1.000       0.944
eval_night08         12       0.750       0.500
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.556
eval_pallet09        33       0.303       0.485
plastic_day_01       44       0.795       0.773
plastic_night_01     22       0.864       0.909
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.950
wood_day_01          24       0.875       0.917
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.755
```

