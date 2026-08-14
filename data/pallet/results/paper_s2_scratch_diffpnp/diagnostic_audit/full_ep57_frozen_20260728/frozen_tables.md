# Frozen PAPER_S2 post-analysis

- Source run status: `complete`
- Full run: `True`
- Frozen checkpoint SHA: `c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896`
- Frozen script SHA: `58ba873be26eb9b66af817ec9bce277d864ffe11472ca3230f20bd0e965b704b`
- CI policy: exact passthrough of the frozen 10,000-replicate summary; no resampling here.
- Missing metadata remains null and is never imputed.

## Validity

- **Primary:** strict filter-val outside/night only (full N=87).
- **Exploratory:** manual36 from the capturepallet11 PL pool.
- **Synthetic:** order-free frame aggregates and channel-agnostic heatmap distributions only.
- **Training ablations:** BLOCKED without separately trained matched checkpoints.

## Dataset membership and failures

| role | validity | frames | success | failure |
| --- | --- | --- | --- | --- |
| exploratory_pl_pool_manual | EXPLORATORY_PL_POOL_MANUAL_N36 | 36 | 36 | 0 |
| strict_filterval | PRIMARY_STRICT_N87 | 87 | 87 | 0 |
| synthetic_fixed_val | SYNTHETIC_ORDER_FREE_CHANNEL_AGNOSTIC_ONLY | 500 | 500 | 0 |

## Local covariance coverage

| nominal metric | empirical | n | conditioning |
| --- | --- | --- | --- |
| local_covariance_coverage_50 | 0.35065 | 770 | finite GT correspondence and finite local-7x7 Mahalanobis distance; includes below-threshold heatmaps |
| local_covariance_coverage_80 | 0.47143 | 770 | finite GT correspondence and finite local-7x7 Mahalanobis distance; includes below-threshold heatmaps |
| local_covariance_coverage_90 | 0.52468 | 770 | finite GT correspondence and finite local-7x7 Mahalanobis distance; includes below-threshold heatmaps |
| local_covariance_coverage_95 | 0.55974 | 770 | finite GT correspondence and finite local-7x7 Mahalanobis distance; includes below-threshold heatmaps |

## Confidence / covariance association with keypoint error

| comparison | Spearman rho | n | notes |
| --- | --- | --- | --- |
| peak_vs_softargmax_error_gt_px | -0.64558 | 770 | Spearman is descriptive; no threshold was selected or tuned |
| peak_second_ratio_vs_softargmax_error_gt_px | -0.58654 | 770 | Spearman is descriptive; no threshold was selected or tuned |
| entropy_normalized_vs_softargmax_error_gt_px | 0.66806 | 770 | Spearman is descriptive; no threshold was selected or tuned |
| local_covariance_area_px2_vs_softargmax_error_gt_px | 0.66714 | 770 | Spearman is descriptive; no threshold was selected or tuned |
| flip_consistency_px_vs_original_softargmax_error_gt_px | 0.22351 | 553 | flip reliability is descriptive; no acceptance threshold selected |

## Signed centroid residual

| metric | statistic | value px | n |
| --- | --- | --- | --- |
| _centroid_dx | mean | 3.1043 | 87 |
| _centroid_dx | median | -7.4002 | 87 |
| _centroid_dy | mean | -6.0498 | 87 |
| _centroid_dy | median | 3.3746 | 87 |
| softargmax_error_gt_px | mean | 41.01 | 87 |
| softargmax_error_gt_px | median | 18.521 | 87 |

## Direct solver comparison (predicted softargmax, locked-Y0 reference)

| solver | metric | statistic | value | success | failure | conditioning |
| --- | --- | --- | --- | --- | --- | --- |
| EPnP | pose_success | success_rate | 0.8046 | 70 | 17 | all pose attempts; failures retained |
| EPnP | solver_runtime_ms | median | 0.085682 | 70 | 17 | all pose attempts with finite runtime, including failures |
| EPnP | yaw_error_vs_oracle_sym180_deg | median | 7.4838 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| EPnP | gt_fixed_reproj_error_px | median | 28.459 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| EPnP | adds180_vs_oracle_m | median | 0.32784 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| EPnP+RANSAC | pose_success | success_rate | 0.8046 | 70 | 17 | all pose attempts; failures retained |
| EPnP+RANSAC | solver_runtime_ms | median | 0.30347 | 70 | 17 | all pose attempts with finite runtime, including failures |
| EPnP+RANSAC | yaw_error_vs_oracle_sym180_deg | median | 5.8906 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| EPnP+RANSAC | gt_fixed_reproj_error_px | median | 20.886 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| EPnP+RANSAC | adds180_vs_oracle_m | median | 0.30086 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| ITERATIVE | pose_success | success_rate | 0.68966 | 60 | 27 | all pose attempts; failures retained |
| ITERATIVE | solver_runtime_ms | median | 0.12518 | 60 | 27 | all pose attempts with finite runtime, including failures |
| ITERATIVE | yaw_error_vs_oracle_sym180_deg | median | 6.4376 | 60 | 27 | successful rows with finite common-reference metric; success/failure rate reported separately |
| ITERATIVE | gt_fixed_reproj_error_px | median | 20.945 | 60 | 27 | successful rows with finite common-reference metric; success/failure rate reported separately |
| ITERATIVE | adds180_vs_oracle_m | median | 0.26036 | 60 | 27 | successful rows with finite common-reference metric; success/failure rate reported separately |
| SQPNP | pose_success | success_rate | 0.8046 | 70 | 17 | all pose attempts; failures retained |
| SQPNP | solver_runtime_ms | median | 0.075869 | 70 | 17 | all pose attempts with finite runtime, including failures |
| SQPNP | yaw_error_vs_oracle_sym180_deg | median | 7.0068 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| SQPNP | gt_fixed_reproj_error_px | median | 27.093 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| SQPNP | adds180_vs_oracle_m | median | 0.31514 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| SQPNP+RefineLM | pose_success | success_rate | 0.8046 | 70 | 17 | all pose attempts; failures retained |
| SQPNP+RefineLM | solver_runtime_ms | median | 0.13519 | 70 | 17 | all pose attempts with finite runtime, including failures |
| SQPNP+RefineLM | yaw_error_vs_oracle_sym180_deg | median | 6.7193 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| SQPNP+RefineLM | gt_fixed_reproj_error_px | median | 24.566 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |
| SQPNP+RefineLM | adds180_vs_oracle_m | median | 0.36503 | 70 | 17 | successful rows with finite common-reference metric; success/failure rate reported separately |

## Frozen paired 95% confidence intervals

| comparison | metric | delta | 95% low | 95% high | n | method |
| --- | --- | --- | --- | --- | --- | --- |
| ladder:Y0_minus_Y2 | pose_success_rate_delta | 0.1954 | 0.075472 | 0.38356 | 87 | session_cluster_bootstrap |
| ladder:Y0_minus_Y2 | yaw_error_vs_oracle_sym180_deg | -11.473 | -16.342 | -7.2577 | 70 | session_cluster_bootstrap |
| ladder:Y1_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y1_minus_Y2 | yaw_error_vs_oracle_sym180_deg | 0.16368 | -0.17458 | 0.52005 | 70 | session_cluster_bootstrap |
| ladder:Y3_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y3_minus_Y2 | yaw_error_vs_oracle_sym180_deg | 0.4078 | 0.016127 | 1.0299 | 70 | session_cluster_bootstrap |
| ladder:Y4_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y4_minus_Y2 | yaw_error_vs_oracle_sym180_deg | 0.72265 | -0.10417 | 1.6781 | 70 | session_cluster_bootstrap |
| ladder:Y5_minus_Y2 | pose_success_rate_delta | 0.011494 | 0 | 0.044118 | 87 | session_cluster_bootstrap |
| ladder:Y5_minus_Y2 | yaw_error_vs_oracle_sym180_deg | -2.2417 | -5.1068 | -0.57523 | 70 | session_cluster_bootstrap |
| ladder:Y6_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y6_minus_Y2 | yaw_error_vs_oracle_sym180_deg | -0.56796 | -1.3039 | 0.00040833 | 70 | session_cluster_bootstrap |
| ladder:Y7_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y7_minus_Y2 | yaw_error_vs_oracle_sym180_deg | 0.65843 | -0.11554 | 1.5528 | 70 | session_cluster_bootstrap |
| ladder:Y8_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y8_minus_Y2 | yaw_error_vs_oracle_sym180_deg | -0.24364 | -0.56169 | -0.0046308 | 70 | session_cluster_bootstrap |
| ladder:Y9_minus_Y2 | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| ladder:Y9_minus_Y2 | yaw_error_vs_oracle_sym180_deg | -0.34968 | -1.7596 | 0.69923 | 70 | session_cluster_bootstrap |
| solver:gt:EPnP+RANSAC_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:gt:EPnP+RANSAC_minus_EPnP | yaw_error_vs_oracle_sym180_deg | 0.031573 | -0.015805 | 0.092127 | 87 | session_cluster_bootstrap |
| solver:gt:ITERATIVE_minus_EPnP | pose_success_rate_delta | -0.011494 | -0.030081 | 0 | 87 | session_cluster_bootstrap |
| solver:gt:ITERATIVE_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.1557 | -0.24167 | -0.097227 | 86 | session_cluster_bootstrap |
| solver:gt:SQPNP+RefineLM_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:gt:SQPNP+RefineLM_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.15921 | -0.2425 | -0.10329 | 87 | session_cluster_bootstrap |
| solver:gt:SQPNP_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:gt:SQPNP_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.07796 | -0.099601 | -0.058128 | 87 | session_cluster_bootstrap |
| solver:predicted_argmax:EPnP+RANSAC_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:predicted_argmax:EPnP+RANSAC_minus_EPnP | yaw_error_vs_oracle_sym180_deg | 0.83889 | -0.95894 | 3.092 | 70 | session_cluster_bootstrap |
| solver:predicted_argmax:ITERATIVE_minus_EPnP | pose_success_rate_delta | -0.12644 | -0.20548 | -0.046512 | 87 | session_cluster_bootstrap |
| solver:predicted_argmax:ITERATIVE_minus_EPnP | yaw_error_vs_oracle_sym180_deg | 1.0379 | -0.25446 | 3.3158 | 59 | session_cluster_bootstrap |
| solver:predicted_argmax:SQPNP+RefineLM_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:predicted_argmax:SQPNP+RefineLM_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.43629 | -1.6888 | 0.37313 | 70 | session_cluster_bootstrap |
| solver:predicted_argmax:SQPNP_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:predicted_argmax:SQPNP_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.58288 | -1.939 | 0.25856 | 70 | session_cluster_bootstrap |
| solver:predicted_softargmax:EPnP+RANSAC_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:predicted_softargmax:EPnP+RANSAC_minus_EPnP | yaw_error_vs_oracle_sym180_deg | 0.67116 | -1.6196 | 2.6165 | 70 | session_cluster_bootstrap |
| solver:predicted_softargmax:ITERATIVE_minus_EPnP | pose_success_rate_delta | -0.11494 | -0.18391 | -0.046875 | 87 | session_cluster_bootstrap |
| solver:predicted_softargmax:ITERATIVE_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.041712 | -0.3441 | 0.231 | 60 | session_cluster_bootstrap |
| solver:predicted_softargmax:SQPNP+RefineLM_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:predicted_softargmax:SQPNP+RefineLM_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.39855 | -1.6403 | 0.42783 | 70 | session_cluster_bootstrap |
| solver:predicted_softargmax:SQPNP_minus_EPnP | pose_success_rate_delta | 0 | 0 | 0 | 87 | session_cluster_bootstrap |
| solver:predicted_softargmax:SQPNP_minus_EPnP | yaw_error_vs_oracle_sym180_deg | -0.55206 | -1.8937 | 0.3152 | 70 | session_cluster_bootstrap |

## Training ablations

| ablation | status | reason |
| --- | --- | --- |
| C0-C4 covariance-weighted pose training/inference | BLOCKED | BLOCKED: requires separately trained matched checkpoints/runs; no causal training conclusion may be drawn from this post-analysis |
| decoder retraining or loss ablation | BLOCKED | BLOCKED: requires separately trained matched checkpoints/runs; no causal training conclusion may be drawn from this post-analysis |
| kp5/centroid supervision ablation | BLOCKED | BLOCKED: requires separately trained matched checkpoints/runs; no causal training conclusion may be drawn from this post-analysis |
| self-training causal ablation | BLOCKED | BLOCKED: requires separately trained matched checkpoints/runs; no causal training conclusion may be drawn from this post-analysis |

The machine-readable long-form source for every table above, including domain/session/slice rows, is `frozen_tables.csv`.
