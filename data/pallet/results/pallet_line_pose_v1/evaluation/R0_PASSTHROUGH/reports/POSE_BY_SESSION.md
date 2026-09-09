# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n   Frozen R0 R0_PASSTHRO
───────────────────────────────────────────────
eval_cad             18       0.743       0.743
eval_night08         12       0.478       0.478
eval_night09         16       0.508       0.508
eval_noapril         12       0.675       0.675
eval_outside         10       0.621       0.621
eval_pallet07        27       0.661       0.661
eval_pallet09        33       0.429       0.429
plastic_day_01       44       0.506       0.506
plastic_night_01     22       0.597       0.597
wood_183705          25       0.790       0.790
wood_184309          20       0.691       0.691
wood_day_01          24       0.559       0.559
wood_night_01        56       0.532       0.532
───────────────────────────────────────────────
ALL                 319       0.603       0.603
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 R0_PASSTHRO
───────────────────────────────────────────────
eval_cad             18       0.803       0.803
eval_night08         12       0.158       0.158
eval_night09         16       0.228       0.228
eval_noapril         12       0.624       0.624
eval_outside         10       0.292       0.292
eval_pallet07        27       0.469       0.469
eval_pallet09        33       0.143       0.143
plastic_day_01       44       0.232       0.232
plastic_night_01     22       0.403       0.403
wood_183705          25       0.695       0.695
wood_184309          20       0.519       0.519
wood_day_01          24       0.367       0.367
wood_night_01        56       0.291       0.291
───────────────────────────────────────────────
ALL                 319       0.428       0.428
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 R0_PASSTHRO
───────────────────────────────────────────────
eval_cad             18        2.17        2.17
eval_night08         12       19.54       19.54
eval_night09         16       16.13       16.13
eval_noapril         12        6.08        6.08
eval_outside         10       11.54       11.54
eval_pallet07        27        7.48        7.48
eval_pallet09        33       19.16       19.16
plastic_day_01       44       12.68       12.68
plastic_night_01     22        9.47        9.47
wood_183705          25        1.59        1.59
wood_184309          20        3.63        3.63
wood_day_01          24        5.23        5.23
wood_night_01        56        7.84        7.84
───────────────────────────────────────────────
ALL                 319        7.90        7.90
```

## axis accuracy  (higher is better)

```text
session               n   Frozen R0 R0_PASSTHRO
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.750
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.303
plastic_day_01       44       0.795       0.795
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.749
```

