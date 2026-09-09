# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       0.743       0.743
eval_night08         12       0.478       0.440
eval_night09         16       0.508       0.509
eval_noapril         12       0.675       0.706
eval_outside         10       0.621       0.634
eval_pallet07        27       0.661       0.657
eval_pallet09        33       0.429       0.497
plastic_day_01       44       0.506       0.546
plastic_night_01     22       0.597       0.655
wood_183705          25       0.790       0.764
wood_184309          20       0.691       0.688
wood_day_01          24       0.559       0.623
wood_night_01        56       0.532       0.578
───────────────────────────────────────────────
ALL                 319       0.603       0.624
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       0.803       0.816
eval_night08         12       0.158       0.158
eval_night09         16       0.228       0.212
eval_noapril         12       0.624       0.659
eval_outside         10       0.292       0.350
eval_pallet07        27       0.469       0.487
eval_pallet09        33       0.143       0.183
plastic_day_01       44       0.232       0.262
plastic_night_01     22       0.403       0.434
wood_183705          25       0.695       0.688
wood_184309          20       0.519       0.546
wood_day_01          24       0.367       0.398
wood_night_01        56       0.291       0.315
───────────────────────────────────────────────
ALL                 319       0.428       0.448
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18        2.17        1.67
eval_night08         12       19.54       20.82
eval_night09         16       16.13       18.76
eval_noapril         12        6.08        5.17
eval_outside         10       11.54        9.23
eval_pallet07        27        7.48        5.89
eval_pallet09        33       19.16       19.09
plastic_day_01       44       12.68       11.68
plastic_night_01     22        9.47        8.32
wood_183705          25        1.59        1.87
wood_184309          20        3.63        3.28
wood_day_01          24        5.23        4.91
wood_night_01        56        7.84        7.21
───────────────────────────────────────────────
ALL                 319        7.90        7.67
```

## axis accuracy  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.485
plastic_day_01       44       0.795       0.795
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.768
```

