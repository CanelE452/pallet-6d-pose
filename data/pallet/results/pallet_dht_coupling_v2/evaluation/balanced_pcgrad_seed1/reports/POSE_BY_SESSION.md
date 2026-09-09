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
eval_cad             18       0.743       0.765
eval_night08         12       0.478       0.514
eval_night09         16       0.508       0.551
eval_noapril         12       0.675       0.662
eval_outside         10       0.621       0.698
eval_pallet07        27       0.661       0.628
eval_pallet09        33       0.429       0.418
plastic_day_01       44       0.506       0.556
plastic_night_01     22       0.597       0.663
wood_183705          25       0.790       0.776
wood_184309          20       0.691       0.636
wood_day_01          24       0.559       0.618
wood_night_01        56       0.532       0.540
───────────────────────────────────────────────
ALL                 319       0.603       0.602
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       0.803       0.780
eval_night08         12       0.158       0.246
eval_night09         16       0.228       0.230
eval_noapril         12       0.624       0.616
eval_outside         10       0.292       0.367
eval_pallet07        27       0.469       0.427
eval_pallet09        33       0.143       0.152
plastic_day_01       44       0.232       0.250
plastic_night_01     22       0.403       0.412
wood_183705          25       0.695       0.619
wood_184309          20       0.519       0.446
wood_day_01          24       0.367       0.389
wood_night_01        56       0.291       0.308
───────────────────────────────────────────────
ALL                 319       0.428       0.421
```

## translation median [cm]  (lower is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18        2.17        2.20
eval_night08         12       19.54       12.68
eval_night09         16       16.13       18.48
eval_noapril         12        6.08        6.11
eval_outside         10       11.54       10.10
eval_pallet07        27        7.48        7.18
eval_pallet09        33       19.16       19.33
plastic_day_01       44       12.68       12.47
plastic_night_01     22        9.47        9.10
wood_183705          25        1.59        2.05
wood_184309          20        3.63        3.92
wood_day_01          24        5.23        4.95
wood_night_01        56        7.84        7.44
───────────────────────────────────────────────
ALL                 319        7.90        7.78
```

## axis accuracy  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.333
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.800
wood_184309          20       0.900       0.800
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.737
```

