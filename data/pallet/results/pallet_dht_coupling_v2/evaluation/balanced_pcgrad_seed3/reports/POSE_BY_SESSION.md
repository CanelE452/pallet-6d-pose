# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       0.743       0.754
eval_night08         12       0.478       0.258
eval_night09         16       0.508       0.427
eval_noapril         12       0.675       0.607
eval_outside         10       0.621       0.609
eval_pallet07        27       0.661       0.698
eval_pallet09        33       0.429       0.458
plastic_day_01       44       0.506       0.562
plastic_night_01     22       0.597       0.608
wood_183705          25       0.790       0.748
wood_184309          20       0.691       0.723
wood_day_01          24       0.559       0.515
wood_night_01        56       0.532       0.499
───────────────────────────────────────────────
ALL                 319       0.603       0.585
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       0.803       0.775
eval_night08         12       0.158       0.140
eval_night09         16       0.228       0.201
eval_noapril         12       0.624       0.601
eval_outside         10       0.292       0.288
eval_pallet07        27       0.469       0.480
eval_pallet09        33       0.143       0.145
plastic_day_01       44       0.232       0.278
plastic_night_01     22       0.403       0.443
wood_183705          25       0.695       0.638
wood_184309          20       0.519       0.555
wood_day_01          24       0.367       0.355
wood_night_01        56       0.291       0.299
───────────────────────────────────────────────
ALL                 319       0.428       0.425
```

## translation median [cm]  (lower is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18        2.17        1.83
eval_night08         12       19.54       40.07
eval_night09         16       16.13       15.21
eval_noapril         12        6.08        6.64
eval_outside         10       11.54       10.84
eval_pallet07        27        7.48        6.62
eval_pallet09        33       19.16       19.50
plastic_day_01       44       12.68       11.00
plastic_night_01     22        9.47        6.91
wood_183705          25        1.59        1.79
wood_184309          20        3.63        2.93
wood_day_01          24        5.23        5.47
wood_night_01        56        7.84        7.57
───────────────────────────────────────────────
ALL                 319        7.90        7.32
```

## axis accuracy  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       1.000       0.944
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.688
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.455
plastic_day_01       44       0.795       0.773
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.755
```

