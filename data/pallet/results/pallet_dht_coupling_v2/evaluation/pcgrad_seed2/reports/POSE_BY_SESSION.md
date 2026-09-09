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
eval_cad             18       0.743       0.768
eval_night08         12       0.478       0.333
eval_night09         16       0.508       0.176
eval_noapril         12       0.675       0.761
eval_outside         10       0.621       0.601
eval_pallet07        27       0.661       0.599
eval_pallet09        33       0.429       0.424
plastic_day_01       44       0.506       0.538
plastic_night_01     22       0.597       0.584
wood_183705          25       0.790       0.813
wood_184309          20       0.691       0.687
wood_day_01          24       0.559       0.656
wood_night_01        56       0.532       0.565
───────────────────────────────────────────────
ALL                 319       0.603       0.594
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       0.803       0.799
eval_night08         12       0.158       0.188
eval_night09         16       0.228       0.155
eval_noapril         12       0.624       0.730
eval_outside         10       0.292       0.321
eval_pallet07        27       0.469       0.421
eval_pallet09        33       0.143       0.150
plastic_day_01       44       0.232       0.276
plastic_night_01     22       0.403       0.421
wood_183705          25       0.695       0.667
wood_184309          20       0.519       0.534
wood_day_01          24       0.367       0.402
wood_night_01        56       0.291       0.328
───────────────────────────────────────────────
ALL                 319       0.428       0.433
```

## translation median [cm]  (lower is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18        2.17        2.46
eval_night08         12       19.54       26.87
eval_night09         16       16.13       46.08
eval_noapril         12        6.08        3.75
eval_outside         10       11.54       10.70
eval_pallet07        27        7.48        7.55
eval_pallet09        33       19.16       23.03
plastic_day_01       44       12.68       12.19
plastic_night_01     22        9.47        7.31
wood_183705          25        1.59        1.67
wood_184309          20        3.63        2.63
wood_day_01          24        5.23        5.48
wood_night_01        56        7.84        6.87
───────────────────────────────────────────────
ALL                 319        7.90        7.62
```

## axis accuracy  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.500
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.700
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.364
plastic_day_01       44       0.795       0.841
plastic_night_01     22       0.864       0.773
wood_183705          25       0.880       0.840
wood_184309          20       0.900       0.850
wood_day_01          24       0.875       0.917
wood_night_01        56       0.732       0.768
───────────────────────────────────────────────
ALL                 319       0.749       0.749
```

