# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       0.743       0.741
eval_night08         12       0.478       0.456
eval_night09         16       0.508       0.486
eval_noapril         12       0.675       0.623
eval_outside         10       0.621       0.618
eval_pallet07        27       0.661       0.630
eval_pallet09        33       0.429       0.420
plastic_day_01       44       0.506       0.501
plastic_night_01     22       0.597       0.612
wood_183705          25       0.790       0.729
wood_184309          20       0.691       0.635
wood_day_01          24       0.559       0.572
wood_night_01        56       0.532       0.549
───────────────────────────────────────────────
ALL                 319       0.603       0.574
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       0.803       0.750
eval_night08         12       0.158       0.186
eval_night09         16       0.228       0.242
eval_noapril         12       0.624       0.557
eval_outside         10       0.292       0.297
eval_pallet07        27       0.469       0.422
eval_pallet09        33       0.143       0.140
plastic_day_01       44       0.232       0.226
plastic_night_01     22       0.403       0.352
wood_183705          25       0.695       0.626
wood_184309          20       0.519       0.492
wood_day_01          24       0.367       0.351
wood_night_01        56       0.291       0.285
───────────────────────────────────────────────
ALL                 319       0.428       0.407
```

## translation median [cm]  (lower is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18        2.17        2.15
eval_night08         12       19.54       21.99
eval_night09         16       16.13       14.33
eval_noapril         12        6.08        6.76
eval_outside         10       11.54       11.04
eval_pallet07        27        7.48        8.44
eval_pallet09        33       19.16       19.93
plastic_day_01       44       12.68       13.88
plastic_night_01     22        9.47        9.53
wood_183705          25        1.59        2.06
wood_184309          20        3.63        4.19
wood_day_01          24        5.23        5.20
wood_night_01        56        7.84        8.10
───────────────────────────────────────────────
ALL                 319        7.90        8.41
```

## axis accuracy  (higher is better)

```text
session               n Historical  hough_featu
───────────────────────────────────────────────
eval_cad             18       1.000       0.944
eval_night08         12       0.750       0.500
eval_night09         16       0.625       0.500
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.364
plastic_day_01       44       0.795       0.818
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.850
wood_day_01          24       0.875       0.875
wood_night_01        56       0.732       0.768
───────────────────────────────────────────────
ALL                 319       0.749       0.740
```

