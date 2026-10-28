# Count-proportional R2 results

Count-proportional image sampling; original-view primary372 evaluation; PLASTIC128/985, WOOD125/963, GREEN119/943. Missing predictions fail PCK. Observed native-pixel error SD is population SD (ddof=0). Training-seed metric SD is sample SD (ddof=1), seeds42/43/44; three R2 sampling/augmentation seeds conditional on the same original R1 weights, Adam state, and fixed pseudo-label/subset sets, not three complete self-training pipeline repetitions. Frame bootstrap uses10000 draws, seed20261026, percentile95, no adjustment. Seed-aggregate frame CIs condition on the three trained seeds; they do not quantify uncertainty from new training seeds. Shared synthetic control has a different R1 history and is contextual. GREEN manual600 is separate. WOOD125 mixes native resolutions and added80 shares recordings with adaptation data; it is not an independent-recording test.

| Kind | Condition | PCK10 mean ± training-seed SD % | Coverage mean ± SD % | Conditional precision10 mean ± SD % |
|---|---|---:|---:|---:|
| GREEN | A | 62.142 ± 0.281 | 100.000 ± 0.000 | 62.142 ± 0.281 |
| GREEN | B | 67.515 ± 0.429 | 100.000 ± 0.000 | 67.515 ± 0.429 |
| GREEN | C | 69.388 ± 0.267 | 99.647 ± 0.341 | 69.635 ± 0.197 |
| GREEN | D | 69.459 ± 0.281 | 99.505 ± 0.221 | 69.804 ± 0.156 |
| GREEN | E | 69.989 ± 0.106 | 99.081 ± 0.061 | 70.639 ± 0.116 |
| PLASTIC | A | 34.416 ± 0.565 | 91.506 ± 0.577 | 37.610 ± 0.411 |
| PLASTIC | B | 35.702 ± 0.117 | 91.201 ± 0.762 | 39.148 ± 0.355 |
| PLASTIC | C | 38.342 ± 0.155 | 90.660 ± 0.634 | 42.293 ± 0.335 |
| PLASTIC | D | 40.203 ± 0.352 | 90.694 ± 0.960 | 44.334 ± 0.847 |
| PLASTIC | E | 42.064 ± 0.480 | 90.829 ± 1.042 | 46.312 ± 0.282 |
| WOOD | A | 56.144 ± 0.334 | 92.835 ± 0.453 | 60.479 ± 0.647 |
| WOOD | B | 58.463 ± 1.198 | 92.973 ± 0.120 | 62.883 ± 1.362 |
| WOOD | C | 61.475 ± 1.179 | 93.700 ± 0.060 | 65.608 ± 1.300 |
| WOOD | D | 63.586 ± 0.779 | 93.319 ± 0.317 | 68.137 ± 0.697 |
| WOOD | E | 64.901 ± 0.519 | 92.662 ± 1.239 | 70.046 ± 0.664 |

| Kind | Candidate | Comparator | Mean-seed PCK10 difference pp | Conditional frame95% CI pp | Decision |
|---|---|---|---:|---|---|
| PLASTIC | A | C | -3.926 | [-5.572, -2.306] | candidate_lower |
| PLASTIC | B | C | -2.640 | [-4.022, -1.354] | candidate_lower |
| PLASTIC | D | C | 1.861 | [0.790, 2.994] | candidate_higher |
| PLASTIC | E | C | 3.723 | [2.105, 5.375] | candidate_higher |
| PLASTIC | A | control_R2 | -8.629 | [-12.306, -4.933] | candidate_lower |
| PLASTIC | B | control_R2 | -7.343 | [-10.695, -3.933] | candidate_lower |
| PLASTIC | C | control_R2 | -4.704 | [-7.724, -1.643] | candidate_lower |
| PLASTIC | D | control_R2 | -2.843 | [-5.527, -0.101] | candidate_lower |
| PLASTIC | E | control_R2 | -0.981 | [-3.610, 1.687] | includes_zero |
| WOOD | A | C | -5.331 | [-7.465, -3.081] | candidate_lower |
| WOOD | B | C | -3.011 | [-4.564, -1.495] | candidate_lower |
| WOOD | D | C | 2.111 | [0.731, 3.485] | candidate_higher |
| WOOD | E | C | 3.427 | [1.698, 5.051] | candidate_higher |
| WOOD | A | control_R2 | 2.146 | [-2.157, 6.481] | includes_zero |
| WOOD | B | control_R2 | 4.465 | [0.313, 8.681] | candidate_higher |
| WOOD | C | control_R2 | 7.477 | [3.523, 11.581] | candidate_higher |
| WOOD | D | control_R2 | 9.588 | [5.848, 13.417] | candidate_higher |
| WOOD | E | control_R2 | 10.903 | [7.341, 14.546] | candidate_higher |
| GREEN | A | C | -7.246 | [-9.589, -4.953] | candidate_lower |
| GREEN | B | C | -1.873 | [-3.355, -0.390] | candidate_lower |
| GREEN | D | C | 0.071 | [-1.098, 1.212] | includes_zero |
| GREEN | E | C | 0.601 | [-1.166, 2.295] | includes_zero |
| GREEN | A | control_R2 | -8.059 | [-12.134, -4.011] | candidate_lower |
| GREEN | B | control_R2 | -2.686 | [-6.477, 1.136] | includes_zero |
| GREEN | C | control_R2 | -0.813 | [-4.334, 2.730] | includes_zero |
| GREEN | D | control_R2 | -0.742 | [-3.976, 2.502] | includes_zero |
| GREEN | E | control_R2 | -0.212 | [-3.238, 2.801] | includes_zero |
