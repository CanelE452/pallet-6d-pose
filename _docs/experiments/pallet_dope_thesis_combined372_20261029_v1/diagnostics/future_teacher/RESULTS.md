# Hypothetical R3 teacher diagnostics

All R2 training pseudo-label targets were fixed across A-E. The table below evaluates a different, post-training quantity: using each completed R2 model as a hypothetical R3 teacher. No R3 model was trained.

| Kind | Condition | Panel | Accepted frames | Scorable selected points | Precision5 | Precision10 | Precision20 | Selected reference coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PLASTIC | A | all_annotated | 27 | 156 | 0.3269 | 0.6474 | 0.7949 | 0.1584 |
| PLASTIC | B | all_annotated | 30 | 176 | 0.3523 | 0.6761 | 0.8125 | 0.1787 |
| PLASTIC | C | all_annotated | 34 | 200 | 0.3550 | 0.7200 | 0.8300 | 0.2030 |
| PLASTIC | D | all_annotated | 34 | 201 | 0.3383 | 0.7463 | 0.8209 | 0.2041 |
| PLASTIC | E | all_annotated | 31 | 180 | 0.4167 | 0.7778 | 0.8500 | 0.1827 |
| WOOD | A | all_annotated | 62 | 430 | 0.2488 | 0.7884 | 0.9535 | 0.4465 |
| WOOD | B | all_annotated | 60 | 419 | 0.2482 | 0.7828 | 0.9451 | 0.4351 |
| WOOD | C | all_annotated | 60 | 417 | 0.2374 | 0.8010 | 0.9424 | 0.4330 |
| WOOD | D | all_annotated | 58 | 403 | 0.2605 | 0.8189 | 0.9454 | 0.4185 |
| WOOD | E | all_annotated | 57 | 399 | 0.2682 | 0.7945 | 0.9424 | 0.4143 |
| GREEN | A | all_annotated | 85 | 591 | 0.4196 | 0.7547 | 0.8646 | 0.6267 |
| GREEN | A | GREEN_manual600 | 85 | 394 | 0.4898 | 0.8376 | 0.9010 | 0.6567 |
| GREEN | B | all_annotated | 86 | 600 | 0.4617 | 0.7667 | 0.8600 | 0.6363 |
| GREEN | B | GREEN_manual600 | 86 | 402 | 0.5348 | 0.8458 | 0.9030 | 0.6700 |
| GREEN | C | all_annotated | 86 | 605 | 0.4678 | 0.7752 | 0.8711 | 0.6416 |
| GREEN | C | GREEN_manual600 | 86 | 404 | 0.5421 | 0.8465 | 0.9084 | 0.6733 |
| GREEN | D | all_annotated | 84 | 598 | 0.4766 | 0.7759 | 0.8746 | 0.6341 |
| GREEN | D | GREEN_manual600 | 84 | 397 | 0.5441 | 0.8388 | 0.9068 | 0.6617 |
| GREEN | E | all_annotated | 91 | 636 | 0.4701 | 0.7783 | 0.8695 | 0.6744 |
| GREEN | E | GREEN_manual600 | 91 | 428 | 0.5374 | 0.8505 | 0.9065 | 0.7133 |

R1 correctness strata use the same fixed reference frames and include missing student predictions as failures. Observed error persistence is not an identification of the cause of error. UNKNOWN K quality is not repaired.
