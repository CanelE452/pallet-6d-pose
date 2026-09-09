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
eval_cad             18       0.743       0.747
eval_night08         12       0.478       0.386
eval_night09         16       0.508       0.475
eval_noapril         12       0.675       0.734
eval_outside         10       0.621       0.627
eval_pallet07        27       0.661       0.654
eval_pallet09        33       0.429       0.430
plastic_day_01       44       0.506       0.533
plastic_night_01     22       0.597       0.584
wood_183705          25       0.790       0.813
wood_184309          20       0.691       0.772
wood_day_01          24       0.559       0.654
wood_night_01        56       0.532       0.547
───────────────────────────────────────────────
ALL                 319       0.603       0.618
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       0.803       0.753
eval_night08         12       0.158       0.157
eval_night09         16       0.228       0.203
eval_noapril         12       0.624       0.709
eval_outside         10       0.292       0.300
eval_pallet07        27       0.469       0.484
eval_pallet09        33       0.143       0.164
plastic_day_01       44       0.232       0.294
plastic_night_01     22       0.403       0.391
wood_183705          25       0.695       0.680
wood_184309          20       0.519       0.615
wood_day_01          24       0.367       0.418
wood_night_01        56       0.291       0.306
───────────────────────────────────────────────
ALL                 319       0.428       0.446
```

## translation median [cm]  (lower is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18        2.17        3.39
eval_night08         12       19.54       24.27
eval_night09         16       16.13       26.30
eval_noapril         12        6.08        4.36
eval_outside         10       11.54       12.19
eval_pallet07        27        7.48        6.73
eval_pallet09        33       19.16       22.97
plastic_day_01       44       12.68       12.03
plastic_night_01     22        9.47        9.93
wood_183705          25        1.59        1.49
wood_184309          20        3.63        2.38
wood_day_01          24        5.23        4.77
wood_night_01        56        7.84        7.24
───────────────────────────────────────────────
ALL                 319        7.90        7.21
```

## axis accuracy  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.500
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.364
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.768
───────────────────────────────────────────────
ALL                 319       0.749       0.749
```

