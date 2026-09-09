# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       0.743       0.698
eval_night08         12       0.478       0.365
eval_night09         16       0.508       0.356
eval_noapril         12       0.675       0.619
eval_outside         10       0.621       0.564
eval_pallet07        27       0.661       0.630
eval_pallet09        33       0.429       0.429
plastic_day_01       44       0.506       0.540
plastic_night_01     22       0.597       0.620
wood_183705          25       0.790       0.761
wood_184309          20       0.691       0.658
wood_day_01          24       0.559       0.589
wood_night_01        56       0.532       0.489
───────────────────────────────────────────────
ALL                 319       0.603       0.605
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       0.803       0.778
eval_night08         12       0.158       0.146
eval_night09         16       0.228       0.184
eval_noapril         12       0.624       0.566
eval_outside         10       0.292       0.231
eval_pallet07        27       0.469       0.483
eval_pallet09        33       0.143       0.184
plastic_day_01       44       0.232       0.267
plastic_night_01     22       0.403       0.440
wood_183705          25       0.695       0.632
wood_184309          20       0.519       0.604
wood_day_01          24       0.367       0.385
wood_night_01        56       0.291       0.300
───────────────────────────────────────────────
ALL                 319       0.428       0.429
```

## translation median [cm]  (lower is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18        2.17        2.22
eval_night08         12       19.54       26.57
eval_night09         16       16.13       22.24
eval_noapril         12        6.08        6.73
eval_outside         10       11.54       13.44
eval_pallet07        27        7.48        7.49
eval_pallet09        33       19.16       15.33
plastic_day_01       44       12.68       11.79
plastic_night_01     22        9.47        6.97
wood_183705          25        1.59        1.85
wood_184309          20        3.63        3.13
wood_day_01          24        5.23        4.35
wood_night_01        56        7.84        8.25
───────────────────────────────────────────────
ALL                 319        7.90        7.56
```

## axis accuracy  (higher is better)

```text
session               n Historical  hough_joint
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.667
eval_night09         16       0.625       0.688
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.741
eval_pallet09        33       0.303       0.455
plastic_day_01       44       0.795       0.750
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.950
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.765
```

