# Square real-supervised secondary

Dimensions are constant within this cohort: dimension utility is not identifiable. Real-supervised engineering evidence, not synthetic-only main evidence. 696 original train images, matched usable subset; 96,000 exposures per fit. DEV155 reused/interleaved; no early stopping. Train probes at1000/3000/6000 are fixed and never select checkpoints.

## SQUARE

| arm | E_sym | median px | P90 px | PCK10 |
|---|---:|---:|---:|---:|
| S0_FIXED | 0.00316741 | 1.6536 | 5.1696 | 0.9714 |
| S1_SYM | 0.00261229 | 1.5319 | 4.1916 | 0.9892 |
| S2_META_SYM | 0.00255598 | 1.5168 | 4.0292 | 0.9892 |

- S1_SYM__minus__S0_FIXED: Δ=-0.00055511634, 95% CI [-0.0008681659830718384, -4.974218323230012e-05]; SUPPORTED_DIAGNOSTIC_ONLY.
- S2_META_SYM__minus__S1_SYM: Δ=-5.6308892e-05, 95% CI [-9.095492471949546e-05, -1.3299171535761001e-05]; UNRESOLVED.
- S2_META_SYM__minus__S0_FIXED: Δ=-0.00061142523, 95% CI [-0.0009410125734507835, -8.190760437279328e-05]; SUPPORTED_DIAGNOSTIC_ONLY.
