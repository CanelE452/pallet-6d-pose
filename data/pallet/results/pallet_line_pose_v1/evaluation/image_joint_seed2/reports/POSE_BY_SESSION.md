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
eval_cad             18       0.743       0.741
eval_night08         12       0.478       0.434
eval_night09         16       0.508       0.478
eval_noapril         12       0.675       0.716
eval_outside         10       0.621       0.624
eval_pallet07        27       0.661       0.640
eval_pallet09        33       0.429       0.498
plastic_day_01       44       0.506       0.558
plastic_night_01     22       0.597       0.648
wood_183705          25       0.790       0.774
wood_184309          20       0.691       0.681
wood_day_01          24       0.559       0.622
wood_night_01        56       0.532       0.545
───────────────────────────────────────────────
ALL                 319       0.603       0.633
```

## ADDsym AUC  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       0.803       0.822
eval_night08         12       0.158       0.150
eval_night09         16       0.228       0.198
eval_noapril         12       0.624       0.653
eval_outside         10       0.292       0.360
eval_pallet07        27       0.469       0.491
eval_pallet09        33       0.143       0.184
plastic_day_01       44       0.232       0.283
plastic_night_01     22       0.403       0.442
wood_183705          25       0.695       0.691
wood_184309          20       0.519       0.547
wood_day_01          24       0.367       0.406
wood_night_01        56       0.291       0.308
───────────────────────────────────────────────
ALL                 319       0.428       0.452
```

## translation median [cm]  (lower is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18        2.17        1.75
eval_night08         12       19.54       20.38
eval_night09         16       16.13       20.89
eval_noapril         12        6.08        5.50
eval_outside         10       11.54        8.82
eval_pallet07        27        7.48        6.23
eval_pallet09        33       19.16       16.51
plastic_day_01       44       12.68       12.11
plastic_night_01     22        9.47        8.24
wood_183705          25        1.59        1.86
wood_184309          20        3.63        3.46
wood_day_01          24        5.23        5.22
wood_night_01        56        7.84        7.52
───────────────────────────────────────────────
ALL                 319        7.90        7.48
```

## axis accuracy  (higher is better)

```text
session               n   Frozen R0 image_joint
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.562
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.630
eval_pallet09        33       0.303       0.485
plastic_day_01       44       0.795       0.841
plastic_night_01     22       0.864       0.864
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.771
```

