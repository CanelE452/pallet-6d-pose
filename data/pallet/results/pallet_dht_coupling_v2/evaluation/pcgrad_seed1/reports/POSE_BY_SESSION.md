# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       0.743       0.755
eval_night08         12       0.478       0.406
eval_night09         16       0.508       0.416
eval_noapril         12       0.675       0.696
eval_outside         10       0.621       0.599
eval_pallet07        27       0.661       0.524
eval_pallet09        33       0.429       0.409
plastic_day_01       44       0.506       0.572
plastic_night_01     22       0.597       0.584
wood_183705          25       0.790       0.779
wood_184309          20       0.691       0.751
wood_day_01          24       0.559       0.583
wood_night_01        56       0.532       0.569
───────────────────────────────────────────────
ALL                 319       0.603       0.590
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       0.803       0.778
eval_night08         12       0.158       0.126
eval_night09         16       0.228       0.188
eval_noapril         12       0.624       0.572
eval_outside         10       0.292       0.285
eval_pallet07        27       0.469       0.391
eval_pallet09        33       0.143       0.108
plastic_day_01       44       0.232       0.277
plastic_night_01     22       0.403       0.408
wood_183705          25       0.695       0.649
wood_184309          20       0.519       0.622
wood_day_01          24       0.367       0.337
wood_night_01        56       0.291       0.334
───────────────────────────────────────────────
ALL                 319       0.428       0.422
```

## translation median [cm]  (lower is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18        2.17        2.57
eval_night08         12       19.54       27.11
eval_night09         16       16.13       30.65
eval_noapril         12        6.08        5.32
eval_outside         10       11.54       10.25
eval_pallet07        27        7.48        9.10
eval_pallet09        33       19.16       22.71
plastic_day_01       44       12.68       11.62
plastic_night_01     22        9.47        8.06
wood_183705          25        1.59        1.82
wood_184309          20        3.63        2.26
wood_day_01          24        5.23        6.05
wood_night_01        56        7.84        6.73
───────────────────────────────────────────────
ALL                 319        7.90        8.06
```

## axis accuracy  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.688
eval_noapril         12       1.000       0.833
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.519
eval_pallet09        33       0.303       0.455
plastic_day_01       44       0.795       0.841
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.917
wood_night_01        56       0.732       0.786
───────────────────────────────────────────────
ALL                 319       0.749       0.755
```

