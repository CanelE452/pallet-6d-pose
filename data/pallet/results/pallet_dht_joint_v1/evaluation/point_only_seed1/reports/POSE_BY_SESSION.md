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
eval_cad             18       0.743       0.773
eval_night08         12       0.478       0.395
eval_night09         16       0.508       0.350
eval_noapril         12       0.675       0.631
eval_outside         10       0.621       0.644
eval_pallet07        27       0.661       0.659
eval_pallet09        33       0.429       0.420
plastic_day_01       44       0.506       0.462
plastic_night_01     22       0.597       0.673
wood_183705          25       0.790       0.735
wood_184309          20       0.691       0.667
wood_day_01          24       0.559       0.535
wood_night_01        56       0.532       0.552
───────────────────────────────────────────────
ALL                 319       0.603       0.582
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       0.803       0.785
eval_night08         12       0.158       0.207
eval_night09         16       0.228       0.194
eval_noapril         12       0.624       0.609
eval_outside         10       0.292       0.305
eval_pallet07        27       0.469       0.421
eval_pallet09        33       0.143       0.136
plastic_day_01       44       0.232       0.191
plastic_night_01     22       0.403       0.416
wood_183705          25       0.695       0.632
wood_184309          20       0.519       0.540
wood_day_01          24       0.367       0.303
wood_night_01        56       0.291       0.309
───────────────────────────────────────────────
ALL                 319       0.428       0.412
```

## translation median [cm]  (lower is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18        2.17        2.36
eval_night08         12       19.54       24.44
eval_night09         16       16.13       30.14
eval_noapril         12        6.08        6.39
eval_outside         10       11.54       11.05
eval_pallet07        27        7.48        8.07
eval_pallet09        33       19.16       21.52
plastic_day_01       44       12.68       13.58
plastic_night_01     22        9.47        8.39
wood_183705          25        1.59        2.23
wood_184309          20        3.63        2.95
wood_day_01          24        5.23        5.84
wood_night_01        56        7.84        7.59
───────────────────────────────────────────────
ALL                 319        7.90        8.12
```

## axis accuracy  (higher is better)

```text
session               n Historical  point_only_
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.333
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.500
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.364
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.909
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.737
```

