# PRIMARY372 same-teacher filter diagnostics

Same-teacher filter diagnostics on PRIMARY372: P128/985, W125/963, G119/943; WOOD manual330 and GREEN manual600 separate. Component removal recomputes each specified filter; cumulative T3 stages condition on the original final selected hypothesis and fixed original fit/medians, with no stage-specific refit or reselection. Missing teacher predictions are excluded from observed incorrect-teacher denominators and fail full-denominator rejected PCK. Zero denominators are NA. Selection and equivalence are frozen before reference cache reads. Native pixel resolutions differ; added80 keep their authorized FINAL role and share training recordings, so this is no independent TEST. Provided K/dimensions are unchanged, K quality UNKNOWN is not repaired. This is not a trained-model performance ablation or independent physical6D accuracy measurement.

| kind | condition | panel | accepted_frames | scorable_accepted_points | precision_le5px | precision_le10px | precision_le20px | correct_retention_le10px | incorrect_acceptance_gt10px |
|---|---|---|---|---|---|---|---|---|---|
| PLASTIC | full | all_annotated_reference | 7 | 43 | 0.186047 | 0.767442 | 1.000000 | 0.127907 | 0.019881 |
| PLASTIC | no_confidence | all_annotated_reference | 12 | 73 | 0.136986 | 0.630137 | 0.945205 | 0.155932 | 0.039359 |
| PLASTIC | no_ransac | all_annotated_reference | 18 | 123 | 0.162602 | 0.560976 | 0.837398 | 0.267442 | 0.107356 |
| PLASTIC | no_loo | all_annotated_reference | 7 | 43 | 0.186047 | 0.767442 | 1.000000 | 0.127907 | 0.019881 |
| PLASTIC | no_flip | all_annotated_reference | 28 | 180 | 0.072222 | 0.405556 | 0.638889 | 0.282946 | 0.212724 |
| PLASTIC | no_final_support | all_annotated_reference | 20 | 97 | 0.154639 | 0.628866 | 0.865979 | 0.236434 | 0.071571 |
| PLASTIC | confidence_only | all_annotated_reference | 92 | 674 | 0.093472 | 0.329377 | 0.580119 | 0.860465 | 0.898608 |
| WOOD | full | all_annotated_reference | 27 | 180 | 0.055556 | 0.516667 | 0.944444 | 0.308970 | 0.179381 |
| WOOD | full | verified_manual_only | 27 | 74 | 0.067568 | 0.594595 | 0.932432 | 0.325926 | 0.229008 |
| WOOD | no_confidence | all_annotated_reference | 28 | 186 | 0.059140 | 0.516129 | 0.940860 | 0.302839 | 0.139752 |
| WOOD | no_confidence | verified_manual_only | 28 | 79 | 0.075949 | 0.594937 | 0.936709 | 0.340580 | 0.166667 |
| WOOD | no_ransac | all_annotated_reference | 53 | 386 | 0.059585 | 0.424870 | 0.870466 | 0.544850 | 0.457732 |
| WOOD | no_ransac | verified_manual_only | 53 | 88 | 0.056818 | 0.590909 | 0.954545 | 0.385185 | 0.274809 |
| WOOD | no_loo | all_annotated_reference | 27 | 180 | 0.055556 | 0.516667 | 0.944444 | 0.308970 | 0.179381 |
| WOOD | no_loo | verified_manual_only | 27 | 74 | 0.067568 | 0.594595 | 0.932432 | 0.325926 | 0.229008 |
| WOOD | no_flip | all_annotated_reference | 58 | 398 | 0.050251 | 0.474874 | 0.879397 | 0.627907 | 0.430928 |
| WOOD | no_flip | verified_manual_only | 58 | 194 | 0.041237 | 0.541237 | 0.886598 | 0.777778 | 0.679389 |
| WOOD | no_final_support | all_annotated_reference | 41 | 241 | 0.045643 | 0.506224 | 0.941909 | 0.405316 | 0.245361 |
| WOOD | no_final_support | verified_manual_only | 41 | 112 | 0.044643 | 0.562500 | 0.937500 | 0.466667 | 0.374046 |
| WOOD | confidence_only | all_annotated_reference | 97 | 740 | 0.059459 | 0.395946 | 0.802703 | 0.973422 | 0.921649 |
| WOOD | confidence_only | verified_manual_only | 97 | 251 | 0.043825 | 0.513944 | 0.872510 | 0.955556 | 0.931298 |
| GREEN | full | all_annotated_reference | 48 | 299 | 0.157191 | 0.638796 | 0.903010 | 0.424444 | 0.225470 |
| GREEN | full | verified_manual_only | 48 | 208 | 0.182692 | 0.750000 | 0.937500 | 0.434540 | 0.222222 |
| GREEN | no_confidence | all_annotated_reference | 47 | 293 | 0.160410 | 0.638225 | 0.901024 | 0.413717 | 0.215886 |
| GREEN | no_confidence | verified_manual_only | 47 | 204 | 0.186275 | 0.750000 | 0.936275 | 0.425000 | 0.212500 |
| GREEN | no_ransac | all_annotated_reference | 58 | 391 | 0.140665 | 0.590793 | 0.859335 | 0.513333 | 0.334029 |
| GREEN | no_ransac | verified_manual_only | 58 | 261 | 0.168582 | 0.720307 | 0.911877 | 0.523677 | 0.311966 |
| GREEN | no_loo | all_annotated_reference | 48 | 299 | 0.157191 | 0.638796 | 0.903010 | 0.424444 | 0.225470 |
| GREEN | no_loo | verified_manual_only | 48 | 208 | 0.182692 | 0.750000 | 0.937500 | 0.434540 | 0.222222 |
| GREEN | no_flip | all_annotated_reference | 103 | 688 | 0.126453 | 0.507267 | 0.815407 | 0.775556 | 0.707724 |
| GREEN | no_flip | verified_manual_only | 103 | 442 | 0.158371 | 0.622172 | 0.880090 | 0.766017 | 0.713675 |
| GREEN | no_final_support | all_annotated_reference | 83 | 449 | 0.149220 | 0.590200 | 0.850780 | 0.588889 | 0.384134 |
| GREEN | no_final_support | verified_manual_only | 83 | 307 | 0.172638 | 0.713355 | 0.921824 | 0.610028 | 0.376068 |
| GREEN | confidence_only | all_annotated_reference | 116 | 919 | 0.115343 | 0.482046 | 0.786725 | 0.984444 | 0.993737 |
| GREEN | confidence_only | verified_manual_only | 116 | 586 | 0.148464 | 0.600683 | 0.849829 | 0.980501 | 1.000000 |

All cumulative T3 stages, selected/rejected precision and full-denominator rejected PCK at5/10/20px are in T3_FILTER_STAGES.csv. Observed-error SD uses population ddof0; private frame counter CSVs retain n/sum/sum-squared for independent pooled-moment QA.
