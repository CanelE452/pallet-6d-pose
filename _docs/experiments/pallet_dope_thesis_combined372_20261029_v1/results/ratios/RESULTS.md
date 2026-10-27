# R2 unique dataset-count sensitivity

Unique synthetic dataset-size sensitivity with fixed real pseudo-label subsets. A/B/C/D/E use synthetic/real counts 0.2/0.5/1/2/4. C is a new count-sampler model. Each branch has 8 draws/update, 828 R2 updates, Lsyn(mean)+Lreal(mean), lambda_real=1. Primary372 comprises the original DEV292 plus explicitly authorized WOOD80; WOOD125 is primary. One training seed; missing predictions fail full-denominator PCK. Conditional precision and pixel errors use observed references. GREEN manual600 is separate. Paired frame bootstrap, 10000 draws, seed20261026, percentile95, no multiplicity correction. Shared synthetic control has a different R1 history; its contrast is contextual. The added WOOD80 shares recording metadata with adaptation data, so the merged set is not an independent recording test. PCK pools native pixel resolutions; resolution metadata is retained. Only the exact separately authorized WOOD80 RGB/cache exception is used; other protected FINAL and every raw reference annotation are forbidden. Original45 is retained solely for reproduction QC.

Phase: all

| Kind | Condition | Real unique | Synthetic unique | PCK5 % | PCK10 % | PCK20 % | Coverage % | Conditional precision10 % | Error mean ± SD px | Median px | n observed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| PLASTIC | A | 20 | 4 | 15.635 | 34.822 | 55.635 | 91.269 | 38.154 | 32.696 ± 45.067 | 14.429 | 899 |
| PLASTIC | B | 20 | 10 | 16.853 | 36.244 | 55.431 | 91.269 | 39.711 | 32.637 ± 44.186 | 13.981 | 899 |
| PLASTIC | C | 20 | 20 | 18.274 | 38.680 | 56.751 | 89.746 | 43.100 | 30.734 ± 41.735 | 12.785 | 884 |
| PLASTIC | D | 20 | 40 | 18.680 | 39.188 | 57.360 | 90.964 | 43.080 | 32.123 ± 44.327 | 12.670 | 896 |
| PLASTIC | E | 20 | 80 | 20.203 | 40.305 | 58.173 | 91.777 | 43.916 | 31.919 ± 44.989 | 12.633 | 904 |
| WOOD | A | 550 | 110 | 20.976 | 61.475 | 83.489 | 93.354 | 65.851 | 18.634 ± 52.441 | 7.880 | 899 |
| WOOD | B | 550 | 275 | 20.768 | 60.852 | 83.074 | 92.627 | 65.695 | 18.965 ± 54.489 | 7.919 | 892 |
| WOOD | C | 550 | 550 | 20.249 | 61.371 | 83.489 | 93.458 | 65.667 | 19.421 ± 55.803 | 7.820 | 900 |
| WOOD | D | 550 | 1100 | 20.976 | 62.409 | 82.555 | 92.731 | 67.301 | 18.071 ± 51.325 | 7.663 | 893 |
| WOOD | E | 550 | 2200 | 20.768 | 61.163 | 82.658 | 92.420 | 66.180 | 17.430 ± 49.466 | 7.585 | 890 |
| GREEN | A | 550 | 110 | 36.903 | 67.338 | 80.064 | 100.000 | 67.338 | 19.772 ± 36.984 | 6.525 | 943 |
| GREEN | B | 550 | 275 | 39.449 | 69.247 | 81.866 | 100.000 | 69.247 | 19.252 ± 37.116 | 6.343 | 943 |
| GREEN | C | 550 | 550 | 40.085 | 69.671 | 82.821 | 100.000 | 69.671 | 19.204 ± 37.376 | 6.402 | 943 |
| GREEN | D | 550 | 1100 | 40.827 | 69.141 | 82.397 | 99.470 | 69.510 | 18.930 ± 37.325 | 6.205 | 938 |
| GREEN | E | 550 | 2200 | 40.721 | 69.565 | 82.397 | 99.258 | 70.085 | 18.826 ± 37.378 | 5.989 | 936 |

## Paired PCK10 differences

| Kind | Candidate | Comparator | Difference pp | 95% CI pp | Decision |
|---|---|---|---:|---|---|
| PLASTIC | A | C | -3.858 | [-5.306, -2.469] | candidate_lower |
| PLASTIC | B | C | -2.437 | [-3.963, -1.009] | candidate_lower |
| PLASTIC | D | C | 0.508 | [-0.599, 1.598] | includes_zero |
| PLASTIC | E | C | 1.624 | [0.000, 3.216] | includes_zero |
| PLASTIC | A | control_R2 | -8.223 | [-11.492, -4.888] | candidate_lower |
| PLASTIC | B | control_R2 | -6.802 | [-10.135, -3.410] | candidate_lower |
| PLASTIC | C | control_R2 | -4.365 | [-7.411, -1.313] | candidate_lower |
| PLASTIC | D | control_R2 | -3.858 | [-6.768, -0.913] | candidate_lower |
| PLASTIC | E | control_R2 | -2.741 | [-5.567, 0.202] | includes_zero |
| WOOD | A | C | 0.104 | [-2.169, 2.513] | includes_zero |
| WOOD | B | C | -0.519 | [-1.951, 1.045] | includes_zero |
| WOOD | D | C | 1.038 | [-0.310, 2.457] | includes_zero |
| WOOD | E | C | -0.208 | [-1.452, 1.050] | includes_zero |
| WOOD | A | control_R2 | 7.477 | [3.368, 11.674] | candidate_higher |
| WOOD | B | control_R2 | 6.854 | [2.766, 11.030] | candidate_higher |
| WOOD | C | control_R2 | 7.373 | [3.276, 11.531] | candidate_higher |
| WOOD | D | control_R2 | 8.411 | [4.474, 12.422] | candidate_higher |
| WOOD | E | control_R2 | 7.165 | [3.202, 11.204] | candidate_higher |
| GREEN | A | C | -2.333 | [-4.000, -0.845] | candidate_lower |
| GREEN | B | C | -0.424 | [-1.697, 0.852] | includes_zero |
| GREEN | D | C | -0.530 | [-1.510, 0.425] | includes_zero |
| GREEN | E | C | -0.106 | [-1.596, 1.273] | includes_zero |
| GREEN | A | control_R2 | -2.863 | [-6.561, 0.850] | includes_zero |
| GREEN | B | control_R2 | -0.954 | [-4.666, 2.860] | includes_zero |
| GREEN | C | control_R2 | -0.530 | [-4.034, 2.972] | includes_zero |
| GREEN | D | control_R2 | -1.060 | [-4.614, 2.447] | includes_zero |
| GREEN | E | control_R2 | -0.636 | [-4.030, 2.754] | includes_zero |

GREEN manual600 is reported separately in GREEN_MANUAL_SUMMARY.csv; no hypothesis of decline was enforced.
