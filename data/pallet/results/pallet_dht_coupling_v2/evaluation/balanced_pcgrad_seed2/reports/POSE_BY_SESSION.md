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
eval_cad             18       0.743       0.771
eval_night08         12       0.478       0.459
eval_night09         16       0.508       0.212
eval_noapril         12       0.675       0.767
eval_outside         10       0.621       0.665
eval_pallet07        27       0.661       0.653
eval_pallet09        33       0.429       0.416
plastic_day_01       44       0.506       0.527
plastic_night_01     22       0.597       0.545
wood_183705          25       0.790       0.784
wood_184309          20       0.691       0.665
wood_day_01          24       0.559       0.548
wood_night_01        56       0.532       0.537
───────────────────────────────────────────────
ALL                 319       0.603       0.593
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       0.803       0.783
eval_night08         12       0.158       0.139
eval_night09         16       0.228       0.248
eval_noapril         12       0.624       0.747
eval_outside         10       0.292       0.412
eval_pallet07        27       0.469       0.445
eval_pallet09        33       0.143       0.128
plastic_day_01       44       0.232       0.259
plastic_night_01     22       0.403       0.379
wood_183705          25       0.695       0.676
wood_184309          20       0.519       0.538
wood_day_01          24       0.367       0.382
wood_night_01        56       0.291       0.289
───────────────────────────────────────────────
ALL                 319       0.428       0.432
```

## translation median [cm]  (lower is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18        2.17        2.38
eval_night08         12       19.54       21.05
eval_night09         16       16.13      140.43
eval_noapril         12        6.08        3.68
eval_outside         10       11.54        8.95
eval_pallet07        27        7.48        6.46
eval_pallet09        33       19.16       23.40
plastic_day_01       44       12.68       12.85
plastic_night_01     22        9.47        8.83
wood_183705          25        1.59        1.73
wood_184309          20        3.63        3.36
wood_day_01          24        5.23        5.57
wood_night_01        56        7.84        8.87
───────────────────────────────────────────────
ALL                 319        7.90        7.49
```

## axis accuracy  (higher is better)

```text
session               n Historical  balanced_pc
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.562
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.242
plastic_day_01       44       0.795       0.841
plastic_night_01     22       0.864       0.773
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.734
```

