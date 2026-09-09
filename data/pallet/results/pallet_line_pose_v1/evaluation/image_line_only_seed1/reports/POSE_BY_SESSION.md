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
eval_cad             18       0.743       0.757
eval_night08         12       0.478       0.432
eval_night09         16       0.508       0.473
eval_noapril         12       0.675       0.701
eval_outside         10       0.621       0.654
eval_pallet07        27       0.661       0.654
eval_pallet09        33       0.429       0.508
plastic_day_01       44       0.506       0.551
plastic_night_01     22       0.597       0.655
wood_183705          25       0.790       0.766
wood_184309          20       0.691       0.674
wood_day_01          24       0.559       0.620
wood_night_01        56       0.532       0.584
───────────────────────────────────────────────
ALL                 319       0.603       0.637
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 image_line_
───────────────────────────────────────────────
eval_cad             18       0.803       0.814
eval_night08         12       0.158       0.153
eval_night09         16       0.228       0.202
eval_noapril         12       0.624       0.659
eval_outside         10       0.292       0.346
eval_pallet07        27       0.469       0.486
eval_pallet09        33       0.143       0.193
plastic_day_01       44       0.232       0.267
plastic_night_01     22       0.403       0.439
wood_183705          25       0.695       0.697
wood_184309          20       0.519       0.544
wood_day_01          24       0.367       0.399
wood_night_01        56       0.291       0.318
───────────────────────────────────────────────
ALL                 319       0.428       0.450
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 image_line_
───────────────────────────────────────────────
eval_cad             18        2.17        1.79
eval_night08         12       19.54       20.86
eval_night09         16       16.13       21.24
eval_noapril         12        6.08        5.45
eval_outside         10       11.54        9.64
eval_pallet07        27        7.48        5.87
eval_pallet09        33       19.16       18.21
plastic_day_01       44       12.68       11.87
plastic_night_01     22        9.47        7.84
wood_183705          25        1.59        1.75
wood_184309          20        3.63        3.45
wood_day_01          24        5.23        4.54
wood_night_01        56        7.84        7.59
───────────────────────────────────────────────
ALL                 319        7.90        7.61
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
eval_pallet09        33       0.303       0.515
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.771
```

