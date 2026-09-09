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
eval_cad             18       0.743       0.753
eval_night08         12       0.478       0.546
eval_night09         16       0.508       0.000
eval_noapril         12       0.675       0.638
eval_outside         10       0.621       0.661
eval_pallet07        27       0.661       0.664
eval_pallet09        33       0.429       0.423
plastic_day_01       44       0.506       0.522
plastic_night_01     22       0.597       0.583
wood_183705          25       0.790       0.778
wood_184309          20       0.691       0.693
wood_day_01          24       0.559       0.586
wood_night_01        56       0.532       0.488
───────────────────────────────────────────────
ALL                 319       0.603       0.609
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       0.803       0.792
eval_night08         12       0.158       0.236
eval_night09         16       0.228       0.104
eval_noapril         12       0.624       0.594
eval_outside         10       0.292       0.335
eval_pallet07        27       0.469       0.429
eval_pallet09        33       0.143       0.130
plastic_day_01       44       0.232       0.275
plastic_night_01     22       0.403       0.416
wood_183705          25       0.695       0.663
wood_184309          20       0.519       0.558
wood_day_01          24       0.367       0.356
wood_night_01        56       0.291       0.279
───────────────────────────────────────────────
ALL                 319       0.428       0.422
```

## translation median [cm]  (lower is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18        2.17        1.68
eval_night08         12       19.54       13.77
eval_night09         16       16.13      446.56
eval_noapril         12        6.08        6.03
eval_outside         10       11.54       10.30
eval_pallet07        27        7.48        7.55
eval_pallet09        33       19.16       22.83
plastic_day_01       44       12.68       11.32
plastic_night_01     22        9.47        7.36
wood_183705          25        1.59        1.58
wood_184309          20        3.63        3.07
wood_day_01          24        5.23        4.59
wood_night_01        56        7.84        8.42
───────────────────────────────────────────────
ALL                 319        7.90        7.55
```

## axis accuracy  (higher is better)

```text
session               n Historical  pcgrad_seed
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.562
eval_noapril         12       1.000       0.917
eval_outside         10       0.700       0.500
eval_pallet07        27       0.630       0.667
eval_pallet09        33       0.303       0.455
plastic_day_01       44       0.795       0.750
plastic_night_01     22       0.864       0.818
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.917
wood_night_01        56       0.732       0.750
───────────────────────────────────────────────
ALL                 319       0.749       0.746
```

