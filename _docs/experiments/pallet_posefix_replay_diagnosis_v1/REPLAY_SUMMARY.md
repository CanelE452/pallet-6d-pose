# PoseFix iterative replay diagnosis — all passes

Historical Replay means synthetic training-data rehearsal, NOT iterative inference. Here: original synthetic-only PRIOR1/2/3, 0 training updates.

Values are arithmetic means of three seed-specific summaries, not medians after pooling seeds. RAW and CAPPED are independent feedback paths. Cap is applied relative to each pass input. No pass, cap or checkpoint is selected on these results.

Median/P90: observed matched corner8. PCK: full fixed GT denominator including missing/mismatch penalties. Pose reference on real/square is geometry reconstructed, not independently measured physical GT. All real sets are reused development diagnostics.

## REAL_DEV

### RAW feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 6.721 | 43.890 | 63.425 | 2.539 | 7.897 | 0.594 | — | — |
| PASS1 | 5.561 | 43.907 | 68.707 | 2.028 | 6.951 | 0.623 | 1.827 | — |
| PASS2 | 5.301 | 44.179 | 68.614 | 1.946 | 6.649 | 0.624 | 0.532 | 2.359 |
| PASS3 | 5.357 | 44.632 | 68.547 | 1.951 | 6.508 | 0.629 | 0.491 | 2.236 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 907 | -0.112 | 0.178 | 0.037 | 49.320 |
| (5,10] | 678 | -1.212 | -1.198 | 0.885 | 27.778 |
| (10,20] | 416 | -2.260 | -2.854 | 4.888 | 26.282 |
| >20 | 444 | -1.445 | -2.290 | 4.054 | 38.889 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

### CAPPED feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 6.721 | 43.890 | 63.425 | 2.539 | 7.897 | 0.594 | — | — |
| PASS1 | 5.559 | 44.802 | 68.694 | 2.025 | 6.895 | 0.624 | 1.827 | — |
| PASS2 | 5.291 | 44.543 | 68.641 | 1.945 | 6.653 | 0.626 | 0.723 | 2.263 |
| PASS3 | 5.358 | 44.931 | 68.467 | 1.950 | 6.472 | 0.629 | 0.641 | 2.168 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 907 | -0.112 | 0.171 | 0.037 | 49.283 |
| (5,10] | 678 | -1.223 | -1.213 | 0.885 | 27.778 |
| (10,20] | 416 | -2.289 | -2.941 | 4.888 | 26.042 |
| >20 | 444 | -1.218 | -2.135 | 4.054 | 38.889 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

## SQUARE_DEV

### RAW feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 3.914 | 8.822 | 93.528 | 0.943 | 10.451 | 0.675 | — | — |
| PASS1 | 3.499 | 8.573 | 92.988 | 0.838 | 9.490 | 0.711 | 0.566 | — |
| PASS2 | 3.556 | 8.965 | 92.341 | 0.858 | 8.921 | 0.724 | 0.135 | 2.751 |
| PASS3 | 3.625 | 9.098 | 91.748 | 0.874 | 8.574 | 0.732 | 0.054 | 3.101 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 807 | -0.144 | 0.064 | 0.041 | 46.138 |
| (5,10] | 349 | -0.491 | -0.566 | 0.096 | 29.704 |
| (10,20] | 76 | -1.878 | -2.466 | 6.579 | 25.877 |
| >20 | 4 | -4.894 | -5.640 | 33.333 | 0.000 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

### CAPPED feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 3.914 | 8.822 | 93.528 | 0.943 | 10.451 | 0.675 | — | — |
| PASS1 | 3.499 | 8.613 | 92.907 | 0.838 | 9.490 | 0.711 | 0.566 | — |
| PASS2 | 3.548 | 8.877 | 92.395 | 0.858 | 8.921 | 0.724 | 0.351 | 2.724 |
| PASS3 | 3.628 | 9.105 | 91.828 | 0.874 | 8.574 | 0.732 | 0.054 | 2.994 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 807 | -0.145 | 0.055 | 0.041 | 46.097 |
| (5,10] | 349 | -0.493 | -0.568 | 0.096 | 29.799 |
| (10,20] | 76 | -1.808 | -2.552 | 6.579 | 25.877 |
| >20 | 4 | -3.968 | -5.609 | 33.333 | 0.000 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

## calibration

### RAW feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 2.047 | 7.057 | 94.148 | 1.234 | 2.656 | 0.835 | — | — |
| PASS1 | 1.427 | 6.186 | 94.880 | 0.985 | 2.063 | 0.866 | 0.467 | — |
| PASS2 | 1.608 | 6.959 | 94.068 | 1.079 | 2.251 | 0.857 | 0.315 | 4.532 |
| PASS3 | 1.800 | 8.130 | 92.479 | 1.270 | 2.568 | 0.836 | 0.282 | 3.495 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 6598 | -0.389 | -0.022 | 0.106 | 41.265 |
| (5,10] | 867 | -1.493 | -1.456 | 1.000 | 26.490 |
| (10,20] | 283 | -1.748 | -2.350 | 3.416 | 31.449 |
| >20 | 181 | -1.407 | -2.422 | 6.446 | 37.201 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

### CAPPED feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 2.047 | 7.057 | 94.148 | 1.234 | 2.656 | 0.835 | — | — |
| PASS1 | 1.426 | 6.186 | 94.863 | 0.985 | 2.055 | 0.866 | 0.467 | — |
| PASS2 | 1.607 | 6.937 | 94.161 | 1.077 | 2.251 | 0.857 | 0.252 | 4.439 |
| PASS3 | 1.795 | 8.083 | 92.618 | 1.265 | 2.567 | 0.836 | 0.227 | 3.448 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 6598 | -0.391 | -0.030 | 0.106 | 41.164 |
| (5,10] | 867 | -1.503 | -1.496 | 1.000 | 26.144 |
| (10,20] | 283 | -1.757 | -2.574 | 3.416 | 31.095 |
| >20 | 181 | -1.244 | -2.378 | 6.446 | 37.017 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

## selection

### RAW feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 1.915 | 6.944 | 93.945 | 1.216 | 2.357 | 0.850 | — | — |
| PASS1 | 1.352 | 6.195 | 94.516 | 0.903 | 1.842 | 0.883 | 0.477 | — |
| PASS2 | 1.530 | 7.032 | 93.768 | 1.007 | 2.077 | 0.876 | 0.210 | 4.522 |
| PASS3 | 1.726 | 8.258 | 92.235 | 1.196 | 2.399 | 0.857 | 0.169 | 3.568 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 6837 | -0.353 | 0.025 | 0.132 | 42.207 |
| (5,10] | 781 | -1.445 | -1.426 | 1.451 | 28.425 |
| (10,20] | 265 | -1.505 | -2.349 | 3.899 | 31.698 |
| >20 | 226 | -0.813 | -1.590 | 3.540 | 46.313 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

### CAPPED feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 1.915 | 6.944 | 93.945 | 1.216 | 2.357 | 0.850 | — | — |
| PASS1 | 1.351 | 6.176 | 94.553 | 0.903 | 1.841 | 0.883 | 0.477 | — |
| PASS2 | 1.531 | 6.999 | 93.838 | 1.007 | 2.077 | 0.876 | 0.271 | 4.316 |
| PASS3 | 1.725 | 8.190 | 92.305 | 1.192 | 2.394 | 0.857 | 0.185 | 3.482 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 6837 | -0.361 | 0.013 | 0.132 | 42.182 |
| (5,10] | 781 | -1.487 | -1.511 | 1.451 | 28.212 |
| (10,20] | 265 | -1.671 | -2.547 | 3.899 | 31.447 |
| >20 | 226 | -0.702 | -1.351 | 3.540 | 46.608 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

## heldout

### RAW feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 1.978 | 7.057 | 93.639 | 1.263 | 2.504 | 0.835 | — | — |
| PASS1 | 1.393 | 6.464 | 94.344 | 0.995 | 1.928 | 0.869 | 0.436 | — |
| PASS2 | 1.563 | 7.264 | 93.426 | 1.111 | 2.232 | 0.858 | 0.243 | 4.354 |
| PASS3 | 1.759 | 8.547 | 91.838 | 1.347 | 2.601 | 0.840 | 0.215 | 3.411 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 13109 | -0.345 | 0.019 | 0.107 | 42.042 |
| (5,10] | 1553 | -1.435 | -1.378 | 0.966 | 29.062 |
| (10,20] | 569 | -1.579 | -2.135 | 3.632 | 35.677 |
| >20 | 427 | -0.121 | -0.580 | 4.372 | 41.998 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.

### CAPPED feedback

| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS0 | 1.978 | 7.057 | 93.639 | 1.263 | 2.504 | 0.835 | — | — |
| PASS1 | 1.392 | 6.464 | 94.350 | 0.995 | 1.930 | 0.869 | 0.436 | — |
| PASS2 | 1.560 | 7.206 | 93.479 | 1.111 | 2.229 | 0.858 | 0.230 | 4.241 |
| PASS3 | 1.757 | 8.490 | 91.951 | 1.346 | 2.600 | 0.840 | 0.215 | 3.373 |

Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.

| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |
|---|---:|---:|---:|---:|---:|
| <=5 | 13109 | -0.347 | 0.013 | 0.107 | 41.986 |
| (5,10] | 1553 | -1.442 | -1.417 | 0.966 | 28.847 |
| (10,20] | 569 | -1.550 | -2.190 | 3.632 | 35.384 |
| >20 | 427 | -0.766 | -1.357 | 4.372 | 41.764 |

Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.
