# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       0.743       0.724
eval_night08         12       0.478       0.237
eval_night09         16       0.508       0.247
eval_noapril         12       0.675       0.612
eval_outside         10       0.621       0.667
eval_pallet07        27       0.661       0.699
eval_pallet09        33       0.429       0.450
plastic_day_01       44       0.506       0.552
plastic_night_01     22       0.597       0.564
wood_183705          25       0.790       0.718
wood_184309          20       0.691       0.680
wood_day_01          24       0.559       0.534
wood_night_01        56       0.532       0.520
───────────────────────────────────────────────
ALL                 319       0.603       0.577
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       0.803       0.786
eval_night08         12       0.158       0.137
eval_night09         16       0.228       0.158
eval_noapril         12       0.624       0.595
eval_outside         10       0.292       0.304
eval_pallet07        27       0.469       0.422
eval_pallet09        33       0.143       0.157
plastic_day_01       44       0.232       0.255
plastic_night_01     22       0.403       0.425
wood_183705          25       0.695       0.619
wood_184309          20       0.519       0.565
wood_day_01          24       0.367       0.329
wood_night_01        56       0.291       0.283
───────────────────────────────────────────────
ALL                 319       0.428       0.414
```

## translation median [cm]  (lower is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18        2.17        2.48
eval_night08         12       19.54       39.21
eval_night09         16       16.13      329.86
eval_noapril         12        6.08        6.75
eval_outside         10       11.54       10.55
eval_pallet07        27        7.48        6.82
eval_pallet09        33       19.16       22.97
plastic_day_01       44       12.68       12.14
plastic_night_01     22        9.47        6.16
wood_183705          25        1.59        2.19
wood_184309          20        3.63        2.81
wood_day_01          24        5.23        5.79
wood_night_01        56        7.84        7.84
───────────────────────────────────────────────
ALL                 319        7.90        7.28
```

## axis accuracy  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.704
eval_pallet09        33       0.303       0.485
plastic_day_01       44       0.795       0.773
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.800
wood_184309          20       0.900       0.950
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.759
```

