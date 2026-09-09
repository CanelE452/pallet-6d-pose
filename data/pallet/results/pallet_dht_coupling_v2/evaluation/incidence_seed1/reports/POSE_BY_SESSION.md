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
eval_cad             18       0.743       0.744
eval_night08         12       0.478       0.426
eval_night09         16       0.508       0.402
eval_noapril         12       0.675       0.700
eval_outside         10       0.621       0.682
eval_pallet07        27       0.661       0.631
eval_pallet09        33       0.429       0.432
plastic_day_01       44       0.506       0.500
plastic_night_01     22       0.597       0.603
wood_183705          25       0.790       0.766
wood_184309          20       0.691       0.698
wood_day_01          24       0.559       0.585
wood_night_01        56       0.532       0.529
───────────────────────────────────────────────
ALL                 319       0.603       0.585
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       0.803       0.787
eval_night08         12       0.158       0.158
eval_night09         16       0.228       0.157
eval_noapril         12       0.624       0.670
eval_outside         10       0.292       0.334
eval_pallet07        27       0.469       0.440
eval_pallet09        33       0.143       0.183
plastic_day_01       44       0.232       0.229
plastic_night_01     22       0.403       0.417
wood_183705          25       0.695       0.631
wood_184309          20       0.519       0.565
wood_day_01          24       0.367       0.322
wood_night_01        56       0.291       0.321
───────────────────────────────────────────────
ALL                 319       0.428       0.423
```

## translation median [cm]  (lower is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18        2.17        2.21
eval_night08         12       19.54       25.73
eval_night09         16       16.13       32.82
eval_noapril         12        6.08        5.22
eval_outside         10       11.54        9.67
eval_pallet07        27        7.48        6.96
eval_pallet09        33       19.16       19.17
plastic_day_01       44       12.68       13.17
plastic_night_01     22        9.47        7.97
wood_183705          25        1.59        1.99
wood_184309          20        3.63        2.90
wood_day_01          24        5.23        5.50
wood_night_01        56        7.84        6.34
───────────────────────────────────────────────
ALL                 319        7.90        6.99
```

## axis accuracy  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.417
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.500
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.424
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.749
```

