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
eval_night08         12       0.478       0.527
eval_night09         16       0.508       0.383
eval_noapril         12       0.675       0.600
eval_outside         10       0.621       0.644
eval_pallet07        27       0.661       0.635
eval_pallet09        33       0.429       0.433
plastic_day_01       44       0.506       0.491
plastic_night_01     22       0.597       0.532
wood_183705          25       0.790       0.726
wood_184309          20       0.691       0.570
wood_day_01          24       0.559       0.637
wood_night_01        56       0.532       0.486
───────────────────────────────────────────────
ALL                 319       0.603       0.571
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       0.803       0.770
eval_night08         12       0.158       0.223
eval_night09         16       0.228       0.212
eval_noapril         12       0.624       0.539
eval_outside         10       0.292       0.305
eval_pallet07        27       0.469       0.451
eval_pallet09        33       0.143       0.169
plastic_day_01       44       0.232       0.262
plastic_night_01     22       0.403       0.342
wood_183705          25       0.695       0.653
wood_184309          20       0.519       0.475
wood_day_01          24       0.367       0.402
wood_night_01        56       0.291       0.282
───────────────────────────────────────────────
ALL                 319       0.428       0.415
```

## translation median [cm]  (lower is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18        2.17        2.29
eval_night08         12       19.54       18.58
eval_night09         16       16.13       24.26
eval_noapril         12        6.08        7.06
eval_outside         10       11.54       11.55
eval_pallet07        27        7.48        8.14
eval_pallet09        33       19.16       21.37
plastic_day_01       44       12.68       11.64
plastic_night_01     22        9.47       10.61
wood_183705          25        1.59        1.81
wood_184309          20        3.63        4.80
wood_day_01          24        5.23        3.30
wood_night_01        56        7.84        8.48
───────────────────────────────────────────────
ALL                 319        7.90        7.95
```

## axis accuracy  (higher is better)

```text
session               n Historical  balanced_se
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       0.917
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.424
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.773
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.752
```

