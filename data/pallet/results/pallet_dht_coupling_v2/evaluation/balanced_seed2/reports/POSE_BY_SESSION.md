# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       0.743       0.754
eval_night08         12       0.478       0.425
eval_night09         16       0.508       0.312
eval_noapril         12       0.675       0.696
eval_outside         10       0.621       0.623
eval_pallet07        27       0.661       0.613
eval_pallet09        33       0.429       0.487
plastic_day_01       44       0.506       0.548
plastic_night_01     22       0.597       0.570
wood_183705          25       0.790       0.776
wood_184309          20       0.691       0.692
wood_day_01          24       0.559       0.598
wood_night_01        56       0.532       0.483
───────────────────────────────────────────────
ALL                 319       0.603       0.584
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       0.803       0.784
eval_night08         12       0.158       0.183
eval_night09         16       0.228       0.142
eval_noapril         12       0.624       0.677
eval_outside         10       0.292       0.333
eval_pallet07        27       0.469       0.393
eval_pallet09        33       0.143       0.158
plastic_day_01       44       0.232       0.237
plastic_night_01     22       0.403       0.417
wood_183705          25       0.695       0.655
wood_184309          20       0.519       0.589
wood_day_01          24       0.367       0.322
wood_night_01        56       0.291       0.304
───────────────────────────────────────────────
ALL                 319       0.428       0.420
```

## translation median [cm]  (lower is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18        2.17        2.36
eval_night08         12       19.54       23.37
eval_night09         16       16.13       34.97
eval_noapril         12        6.08        4.75
eval_outside         10       11.54       10.13
eval_pallet07        27        7.48        8.38
eval_pallet09        33       19.16       20.75
plastic_day_01       44       12.68       11.60
plastic_night_01     22        9.47        8.21
wood_183705          25        1.59        1.89
wood_184309          20        3.63        2.79
wood_day_01          24        5.23        5.34
wood_night_01        56        7.84        8.55
───────────────────────────────────────────────
ALL                 319        7.90        7.70
```

## axis accuracy  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.500
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.593
eval_pallet09        33       0.303       0.485
plastic_day_01       44       0.795       0.750
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.740
```

