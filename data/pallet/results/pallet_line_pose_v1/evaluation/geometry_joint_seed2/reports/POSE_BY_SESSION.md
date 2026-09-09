# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n   Frozen R0 geometry_jo
───────────────────────────────────────────────
eval_cad             18       0.743       0.747
eval_night08         12       0.478       0.473
eval_night09         16       0.508       0.505
eval_noapril         12       0.675       0.699
eval_outside         10       0.621       0.628
eval_pallet07        27       0.661       0.661
eval_pallet09        33       0.429       0.491
plastic_day_01       44       0.506       0.555
plastic_night_01     22       0.597       0.614
wood_183705          25       0.790       0.804
wood_184309          20       0.691       0.713
wood_day_01          24       0.559       0.582
wood_night_01        56       0.532       0.507
───────────────────────────────────────────────
ALL                 319       0.603       0.604
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 geometry_jo
───────────────────────────────────────────────
eval_cad             18       0.803       0.807
eval_night08         12       0.158       0.150
eval_night09         16       0.228       0.229
eval_noapril         12       0.624       0.648
eval_outside         10       0.292       0.285
eval_pallet07        27       0.469       0.473
eval_pallet09        33       0.143       0.142
plastic_day_01       44       0.232       0.281
plastic_night_01     22       0.403       0.420
wood_183705          25       0.695       0.700
wood_184309          20       0.519       0.531
wood_day_01          24       0.367       0.373
wood_night_01        56       0.291       0.284
───────────────────────────────────────────────
ALL                 319       0.428       0.438
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 geometry_jo
───────────────────────────────────────────────
eval_cad             18        2.17        2.16
eval_night08         12       19.54       19.01
eval_night09         16       16.13       16.13
eval_noapril         12        6.08        5.58
eval_outside         10       11.54       10.85
eval_pallet07        27        7.48        7.38
eval_pallet09        33       19.16       18.91
plastic_day_01       44       12.68       11.55
plastic_night_01     22        9.47        8.49
wood_183705          25        1.59        1.53
wood_184309          20        3.63        3.30
wood_day_01          24        5.23        5.68
wood_night_01        56        7.84        7.90
───────────────────────────────────────────────
ALL                 319        7.90        7.68
```

## axis accuracy  (higher is better)

```text
session               n   Frozen R0 geometry_jo
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.750
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.364
plastic_day_01       44       0.795       0.795
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.752
```

