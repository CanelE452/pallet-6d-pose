# C complete three-seed screen — authorized repair

Verdict: **C_GEOMETRY_PRESERVING_DA_FAIL**. Evidence: NEGATIVE_DEVELOPMENT; independent confirmation NOT_RUN.

All9 arm/seed fits have900 actual optimizer updates and final checkpoints. Each was evaluated on the same319 positive+2689 negative frames with unchanged canonical2D and MAIN6D evaluators.
Accounting:8100 valid-comparison updates +900 consumed by the historical lost-checkpoint run =9000 cumulative. This authorized resume added6300 updates, including one900-update repair. No new hyperparameter, threshold, seed or architecture search.

## Seed means

| Arm | AP50-95 | Det | paired kp median px | paired P90 px | paired gross20 | translation cm | yaw deg | IoU3D | ADDsym AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.748155 | 0.956113 | 6.7396 | 38.7305 | 0.173293 | 7.8383 | 1.2700 | 0.596232 | 0.420204 |
| C1 | 0.765264 | 0.989551 | 6.9874 | 42.6664 | 0.193867 | 8.3878 | 1.3632 | 0.586872 | 0.402019 |
| C2 | 0.744155 | 0.981191 | 6.5205 | 37.8393 | 0.167881 | 8.0003 | 1.2232 | 0.602404 | 0.424395 |

Paired geometry uses the3-arm common matched-frame set within each seed, not the union of separately detected frames. Seed means are means of per-seed statistics, not pooled independent seeds.

## All seeds

| Seed | Arm | AP50 | AP50-95 | AUROC | FPR95 | Night N | Night Det | matched kp median/P90 px | rotation/yaw deg | translation cm | pose coverage |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | C0 | 0.909405 | 0.738032 | 0.984914 | 0.076608 | 106 | 0.886792 | 6.8596/42.1534 | 2.5006/1.2792 | 7.9888 | 1.0000 |
| 1 | C1 | 0.944470 | 0.760135 | 0.991331 | 0.048717 | 106 | 0.990566 | 7.4944/44.3198 | 2.5647/1.3751 | 9.1888 | 1.0000 |
| 1 | C2 | 0.913203 | 0.743227 | 0.987580 | 0.063221 | 106 | 0.943396 | 6.6618/38.7218 | 2.2198/1.1986 | 8.0536 | 1.0000 |
| 2 | C0 | 0.925632 | 0.758032 | 0.988803 | 0.068799 | 106 | 0.905660 | 6.9014/39.0458 | 2.6505/1.3679 | 7.9322 | 1.0000 |
| 2 | C1 | 0.949335 | 0.772579 | 0.992326 | 0.044254 | 106 | 0.981132 | 7.1402/38.4312 | 2.4065/1.3469 | 7.8863 | 1.0000 |
| 2 | C2 | 0.916608 | 0.747894 | 0.986743 | 0.066939 | 106 | 0.943396 | 6.6700/39.7224 | 2.2625/1.2584 | 7.8936 | 1.0000 |
| 3 | C0 | 0.903133 | 0.748400 | 0.986406 | 0.090740 | 106 | 0.858491 | 6.4827/35.4609 | 2.5151/1.1630 | 7.5940 | 1.0000 |
| 3 | C1 | 0.941737 | 0.763076 | 0.992168 | 0.039420 | 106 | 0.990566 | 7.2111/49.3778 | 2.4837/1.3677 | 8.0883 | 1.0000 |
| 3 | C2 | 0.914810 | 0.741345 | 0.987371 | 0.073261 | 106 | 0.952830 | 6.7483/38.9988 | 2.2521/1.2125 | 8.0536 | 1.0000 |

## Locked gates

- detection_improvement: FAIL
- recovery_70pct: FAIL
- median_px_nonworse: PASS
- p90_px_nonworse: PASS
- gross20_nonworse: PASS
- pose_safe: PASS

C1−C0 AP50-95: 0.01710878; C2−C0: -0.00399952; recovery fraction: -0.233769901249712.
Per-seed contrasts and safety flags are in VERDICT.json. The gate was frozen before the first fit, not chosen from these results.

## Execution and deployment audit

All9 initial tensor hashes match. All900 actual synthetic batch hashes per seed match across3 arms; C1/C2 real batch hashes also match. Existing273 pseudo labels are reused; manual real GT never enters training.
All3 saved C2 checkpoints independently preserve639 non-detection state keys bit-exactly versus R0. Raw dense pose tensors match on two CPU probe sizes. All9 canonicalCSV median/P90 values were independently recomputed.
Deployment remains the stock3,043,704-parameter RGB Pose26 model. No additional branch or sensor is added; runtime in milliseconds was not newly benchmarked. Dense pose invariance does not imply unchanged final detection selection.
The previous BN audit failure, old checkpoint/task-metadata correction and partial reports remain intact. C0/C1 seed1 were not refit. Existing optional-Albumentations and strict-cuBLAS-determinism limitations remain; actual input parity is verified, bit-exact retraining is not claimed.

## Paper interpretation

Recommended path: `NO_NEW_METHOD_FREEZE_EXISTING_STORY`.
C does not qualify under the frozen gate. Do not promote one improved metric into geometry-preserving DA success. Limit the paper to the strong synthetic baseline, controlled target-adaptation study, local-line development result and failure-mechanism analysis; do not add a new module to force a method claim.
A still lacks untouched confirmation data; D/B/E remain their original mechanism FAIL outcomes with zero student fits. No D3 adapter was implemented. No claim that DHT, self-training or depth teaching is impossible.

Optional future work only: because C2 preserves geometry but does not recover detection gain, the master permits recommending a detection-specific residual adapter (D3). It is not implemented, trained or selected here and requires a separately authorized experiment; it is not required for the recommended paper closeout.
