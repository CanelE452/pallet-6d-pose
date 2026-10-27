# Primary372: six main self-training comparisons

Primary372: PLASTIC128/985 reference points, WOOD125/963, GREEN119/943. Missing predictions fail full-denominator PCK. Conditional precision uses observed references. All intervals are paired-frame percentile95 with 10000 draws, seed20261026, no multiplicity correction. GREEN manual600 and WOOD manual330 are separate panels. Four-recording WOOD intervals are supplementary; any zero-denominator replicate makes the whole interval NA, with no dropped or redrawn replicates. Native pixel thresholds pool WOOD1280x720 and640x480. The added80 retain their explicitly authorized FINAL role and share training recordings; this is not a recording-independent test. One training seed; bootstrap intervals do not quantify training-seed variation. These are 2D metrics, not independent 6D accuracy.

## Primary all annotated references

| Kind | Panel | Candidate minus comparator | PCK10 difference pp | 95% CI pp | Decision |
|---|---|---|---:|---|---|
| PLASTIC | all_annotated | full_R1_minus_control_R1 | 1.726 | [-0.815, 4.295] | includes_zero |
| PLASTIC | all_annotated | full_R2_minus_control_R2 | -3.553 | [-6.357, -0.809] | candidate_lower |
| PLASTIC | all_annotated | full_R1_minus_no_flip_R1 | 4.264 | [1.217, 7.322] | candidate_higher |
| PLASTIC | all_annotated | full_R2_minus_no_flip_R2 | 4.365 | [1.327, 7.441] | candidate_higher |
| PLASTIC | all_annotated | full_R2_minus_full_R1 | 1.421 | [-1.010, 3.793] | includes_zero |
| PLASTIC | all_annotated | control_R2_minus_control_R1 | 6.701 | [4.082, 9.422] | candidate_higher |
| WOOD | all_annotated | full_R1_minus_control_R1 | -2.700 | [-6.875, 1.655] | includes_zero |
| WOOD | all_annotated | full_R2_minus_control_R2 | 8.307 | [4.456, 12.358] | candidate_higher |
| WOOD | all_annotated | full_R1_minus_no_flip_R1 | -2.700 | [-6.361, 0.731] | includes_zero |
| WOOD | all_annotated | full_R2_minus_no_flip_R2 | 0.000 | [-3.739, 3.575] | includes_zero |
| WOOD | all_annotated | full_R2_minus_full_R1 | 9.242 | [6.590, 11.967] | candidate_higher |
| WOOD | all_annotated | control_R2_minus_control_R1 | -1.765 | [-4.289, 0.719] | includes_zero |
| GREEN | all_annotated | full_R1_minus_control_R1 | 0.636 | [-2.335, 3.708] | includes_zero |
| GREEN | all_annotated | full_R2_minus_control_R2 | -0.530 | [-3.899, 2.963] | includes_zero |
| GREEN | all_annotated | full_R1_minus_no_flip_R1 | 1.166 | [-1.596, 3.895] | includes_zero |
| GREEN | all_annotated | full_R2_minus_no_flip_R2 | 1.909 | [0.106, 3.797] | candidate_higher |
| GREEN | all_annotated | full_R2_minus_full_R1 | 0.530 | [-2.436, 3.609] | includes_zero |
| GREEN | all_annotated | control_R2_minus_control_R1 | 1.697 | [-0.213, 3.629] | includes_zero |

## GREEN manual600

| Kind | Panel | Candidate minus comparator | PCK10 difference pp | 95% CI pp | Decision |
|---|---|---|---:|---|---|
| GREEN | GREEN_manual600 | full_R1_minus_control_R1 | 2.167 | [-0.511, 5.090] | includes_zero |
| GREEN | GREEN_manual600 | full_R2_minus_control_R2 | 0.667 | [-1.993, 3.344] | includes_zero |
| GREEN | GREEN_manual600 | full_R1_minus_no_flip_R1 | -0.333 | [-2.680, 1.964] | includes_zero |
| GREEN | GREEN_manual600 | full_R2_minus_no_flip_R2 | 0.667 | [-1.174, 2.471] | includes_zero |
| GREEN | GREEN_manual600 | full_R2_minus_full_R1 | 1.000 | [-1.667, 3.704] | includes_zero |
| GREEN | GREEN_manual600 | control_R2_minus_control_R1 | 2.500 | [0.828, 4.355] | candidate_higher |

## WOOD manual330

| Kind | Panel | Candidate minus comparator | PCK10 difference pp | 95% CI pp | Decision |
|---|---|---|---:|---|---|
| WOOD | WOOD_manual330 | full_R1_minus_control_R1 | 1.818 | [-3.916, 7.902] | includes_zero |
| WOOD | WOOD_manual330 | full_R2_minus_control_R2 | 11.515 | [5.682, 17.470] | candidate_higher |
| WOOD | WOOD_manual330 | full_R1_minus_no_flip_R1 | -4.545 | [-10.345, 0.645] | includes_zero |
| WOOD | WOOD_manual330 | full_R2_minus_no_flip_R2 | -2.727 | [-8.657, 3.114] | includes_zero |
| WOOD | WOOD_manual330 | full_R2_minus_full_R1 | 6.364 | [3.175, 9.873] | candidate_higher |
| WOOD | WOOD_manual330 | control_R2_minus_control_R1 | -3.333 | [-7.643, 0.909] | includes_zero |

## WOOD four-recording supplementary

| Kind | Panel | Candidate minus comparator | PCK10 difference pp | 95% CI pp | Decision |
|---|---|---|---:|---|---|
| WOOD | all_annotated | full_R1_minus_control_R1 | -2.700 | [-5.420, -1.190] | candidate_lower |
| WOOD | WOOD_manual330 | full_R1_minus_control_R1 | 1.818 | [NA, NA] | NA_undefined_denominator_no_draws_dropped |
| WOOD | all_annotated | full_R2_minus_control_R2 | 8.307 | [6.175, 9.820] | candidate_higher |
| WOOD | WOOD_manual330 | full_R2_minus_control_R2 | 11.515 | [NA, NA] | NA_undefined_denominator_no_draws_dropped |
| WOOD | all_annotated | full_R1_minus_no_flip_R1 | -2.700 | [-4.466, 0.595] | includes_zero |
| WOOD | WOOD_manual330 | full_R1_minus_no_flip_R1 | -4.545 | [NA, NA] | NA_undefined_denominator_no_draws_dropped |
| WOOD | all_annotated | full_R2_minus_no_flip_R2 | 0.000 | [-3.448, 5.202] | includes_zero |
| WOOD | WOOD_manual330 | full_R2_minus_no_flip_R2 | -2.727 | [NA, NA] | NA_undefined_denominator_no_draws_dropped |
| WOOD | all_annotated | full_R2_minus_full_R1 | 9.242 | [6.845, 13.279] | candidate_higher |
| WOOD | WOOD_manual330 | full_R2_minus_full_R1 | 6.364 | [NA, NA] | NA_undefined_denominator_no_draws_dropped |
| WOOD | all_annotated | control_R2_minus_control_R1 | -1.765 | [-3.448, 0.578] | includes_zero |
| WOOD | WOOD_manual330 | control_R2_minus_control_R1 | -3.333 | [NA, NA] | NA_undefined_denominator_no_draws_dropped |

All five metrics are retained in CSV. Positive differences favor the named candidate. An interval including zero is inconclusive, not equivalence or proof of no effect. Observed-point precision must be read with full-denominator PCK and coverage.
