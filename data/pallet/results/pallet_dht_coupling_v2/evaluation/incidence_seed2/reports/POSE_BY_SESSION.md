# 6D pose per evaluation session

Same frozen selector, same ground truth, same metrics as the main pose
table. Only the aggregation axis changed. No model ran again — the cached
2D predictions were re-read, and the pooled numbers reproduce the existing
per-arm files exactly, which is what makes this split trustworthy.

**Session sample sizes are small (10-56).** A rank change between two arms
inside one session is not evidence on its own.

## IoU3D median  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       0.743       0.714
eval_night08         12       0.478       0.296
eval_night09         16       0.508       0.234
eval_noapril         12       0.675       0.736
eval_outside         10       0.621       0.633
eval_pallet07        27       0.661       0.635
eval_pallet09        33       0.429       0.407
plastic_day_01       44       0.506       0.550
plastic_night_01     22       0.597       0.549
wood_183705          25       0.790       0.796
wood_184309          20       0.691       0.763
wood_day_01          24       0.559       0.643
wood_night_01        56       0.532       0.503
───────────────────────────────────────────────
ALL                 319       0.603       0.602
```

## ADDsym AUC  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       0.803       0.776
eval_night08         12       0.158       0.073
eval_night09         16       0.228       0.134
eval_noapril         12       0.624       0.713
eval_outside         10       0.292       0.329
eval_pallet07        27       0.469       0.428
eval_pallet09        33       0.143       0.123
plastic_day_01       44       0.232       0.245
plastic_night_01     22       0.403       0.375
wood_183705          25       0.695       0.691
wood_184309          20       0.519       0.597
wood_day_01          24       0.367       0.355
wood_night_01        56       0.291       0.296
───────────────────────────────────────────────
ALL                 319       0.428       0.412
```

## translation median [cm]  (lower is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18        2.17        2.98
eval_night08         12       19.54       32.93
eval_night09         16       16.13       41.15
eval_noapril         12        6.08        4.51
eval_outside         10       11.54       10.13
eval_pallet07        27        7.48        6.89
eval_pallet09        33       19.16       22.08
plastic_day_01       44       12.68       11.24
plastic_night_01     22        9.47        9.07
wood_183705          25        1.59        1.63
wood_184309          20        3.63        2.38
wood_day_01          24        5.23        5.30
wood_night_01        56        7.84        7.95
───────────────────────────────────────────────
ALL                 319        7.90        7.83
```

## axis accuracy  (higher is better)

```text
session               n Historical  incidence_s
───────────────────────────────────────────────
eval_cad             18       1.000       1.000
eval_night08         12       0.750       0.583
eval_night09         16       0.625       0.625
eval_noapril         12       1.000       1.000
eval_outside         10       0.700       0.600
eval_pallet07        27       0.630       0.556
eval_pallet09        33       0.303       0.485
plastic_day_01       44       0.795       0.773
plastic_night_01     22       0.864       0.727
wood_183705          25       0.880       0.880
wood_184309          20       0.900       0.900
wood_day_01          24       0.875       0.833
wood_night_01        56       0.732       0.732
───────────────────────────────────────────────
ALL                 319       0.749       0.737
```

