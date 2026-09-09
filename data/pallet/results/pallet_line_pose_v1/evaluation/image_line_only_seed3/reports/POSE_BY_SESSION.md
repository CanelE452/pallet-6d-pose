# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n   Frozen R0 image_line_
───────────────────────────────────────────────
eval_cad             18       0.743       0.754
eval_night08         12       0.478       0.437
eval_night09         16       0.508       0.472
eval_noapril         12       0.675       0.716
eval_outside         10       0.621       0.625
eval_pallet07        27       0.661       0.651
eval_pallet09        33       0.429       0.455
plastic_day_01       44       0.506       0.537
plastic_night_01     22       0.597       0.643
wood_183705          25       0.790       0.772
wood_184309          20       0.691       0.687
wood_day_01          24       0.559       0.614
wood_night_01        56       0.532       0.578
───────────────────────────────────────────────
ALL                 319       0.603       0.625
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 image_line_
───────────────────────────────────────────────
eval_cad             18       0.803       0.818
eval_night08         12       0.158       0.154
eval_night09         16       0.228       0.204
eval_noapril         12       0.624       0.660
eval_outside         10       0.292       0.346
eval_pallet07        27       0.469       0.490
eval_pallet09        33       0.143       0.161
plastic_day_01       44       0.232       0.261
plastic_night_01     22       0.403       0.436
wood_183705          25       0.695       0.685
wood_184309          20       0.519       0.543
wood_day_01          24       0.367       0.400
wood_night_01        56       0.291       0.310
───────────────────────────────────────────────
ALL                 319       0.428       0.445
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 image_line_
───────────────────────────────────────────────
eval_cad             18        2.17        1.90
eval_night08         12       19.54       20.54
eval_night09         16       16.13       22.08
eval_noapril         12        6.08        5.04
eval_outside         10       11.54        9.47
eval_pallet07        27        7.48        6.11
eval_pallet09        33       19.16       20.14
plastic_day_01       44       12.68       11.78
plastic_night_01     22        9.47        8.19
wood_183705          25        1.59        1.87
wood_184309          20        3.63        3.39
wood_day_01          24        5.23        4.91
wood_night_01        56        7.84        7.50
───────────────────────────────────────────────
ALL                 319        7.90        7.67
```

## axis accuracy  (higher is better)

```text
session               n   Frozen R0 image_line_
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.562
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.394
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.755
```

