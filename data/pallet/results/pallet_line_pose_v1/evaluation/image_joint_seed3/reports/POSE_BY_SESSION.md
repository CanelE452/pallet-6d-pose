# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       0.743       0.756
eval_night08         12       0.478       0.429
eval_night09         16       0.508       0.502
eval_noapril         12       0.675       0.713
eval_outside         10       0.621       0.633
eval_pallet07        27       0.661       0.654
eval_pallet09        33       0.429       0.447
plastic_day_01       44       0.506       0.543
plastic_night_01     22       0.597       0.642
wood_183705          25       0.790       0.756
wood_184309          20       0.691       0.672
wood_day_01          24       0.559       0.613
wood_night_01        56       0.532       0.567
───────────────────────────────────────────────
ALL                 319       0.603       0.634
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       0.803       0.817
eval_night08         12       0.158       0.148
eval_night09         16       0.228       0.213
eval_noapril         12       0.624       0.671
eval_outside         10       0.292       0.352
eval_pallet07        27       0.469       0.488
eval_pallet09        33       0.143       0.170
plastic_day_01       44       0.232       0.262
plastic_night_01     22       0.403       0.438
wood_183705          25       0.695       0.681
wood_184309          20       0.519       0.539
wood_day_01          24       0.367       0.405
wood_night_01        56       0.291       0.308
───────────────────────────────────────────────
ALL                 319       0.428       0.447
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18        2.17        1.87
eval_night08         12       19.54       20.09
eval_night09         16       16.13       19.34
eval_noapril         12        6.08        4.95
eval_outside         10       11.54        9.38
eval_pallet07        27        7.48        5.93
eval_pallet09        33       19.16       20.24
plastic_day_01       44       12.68       11.99
plastic_night_01     22        9.47        8.38
wood_183705          25        1.59        1.91
wood_184309          20        3.63        3.68
wood_day_01          24        5.23        4.86
wood_night_01        56        7.84        7.42
───────────────────────────────────────────────
ALL                 319        7.90        7.57
```

## axis accuracy  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.424
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.768
───────────────────────────────────────────────
ALL                 319       0.749       0.768
```

