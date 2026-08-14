# PAPER_S2 micro architecture screen — 해결 능력 판별

논문 최종 학습이 아니다.  후보가 **자기 target failure class 를 고치는지**만, 같은 manifest 로 학습한 matched control 대비로 본다.

- checkpoint `net_epoch_0057.pth` SHA `c0055fe7c4210f63…` (불변)
- git HEAD `9045c69224dca86e0225b3b777ad720afa1aa1b3`
- mechanism-val membership SHA `8d086cd7f8a20cf6…`
- baseline gate: GT-2D 87/87, predicted 70/87, yaw 6.025°, reproj 23.162px
- final-test open count **0**
- torch 2.1.1+cu118 / cuda 11.8 / NVIDIA GeForce RTX 3080


## ⚠ 0. 먼저 읽을 것 — B 계열은 설계대로 시험되지 않았다

[확인] Manifest B 의 'F2-like hard frame = far/depth 2D error > 20px' 기준을 만족하는 학습 프레임이 **0개**뿐이다.  ep57 은 자기 synthetic 학습 분포에서 이미 정확하다 — pool far error 중앙값 2.12px, 최대 16.02px.
[확인] 같은 지표로 real mechanism-val 의 F2 는 far error 중앙값 43.3px, 최대 160.7px 다.  즉 **F2_CONFIDENT_WRONG 은 synthetic 학습 데이터에 존재하지 않는 실패 모드**이며, 이는 순수한 sim2real 전이 문제다.
[확인] 수를 억지로 채우지 않고, 가장 어려운 프레임(>= 3.66px)으로 대체해 실행했고 그 사실을 manifest 에 기록했다.
[판정] 따라서 **B1 의 FAIL 은 후보 기각이 아니라 '이 데이터로는 시험 불가'** 다.  Phase 16 의 'B1 FAIL → backbone 으로 이동' 규칙을 이 결과에 적용하면 안 된다.


## Phase 12 — ablation table (계획 외 기능 없음)

```
arm   training_manifest  trainable      target_semantics   residual  edge_loss
──────────────────────────────────────────────────────────────────────────────
ep57  original           none           legacy             no        no       
M0_B  B                  m6_2 tail      legacy             no        no       
B1    B                  residual head  legacy             yes       no       
M0_A  A                  m6_2 tail      legacy             no        no       
A1    A                  m6_2 tail      corrected partial  no        no       
B2    B                  residual head  legacy             yes       yes      
```

[확인] 모든 arm 에서 꺼진 항목: affinity_loss, covariance_loss, diffpnp_loss, mask_aux_loss, old_structural_loss, reliability_loss, symmetric_loss, teacher_loss, visibility_coord_loss (loss 미계산).
[주의] loader 의 `aspect_resize` 는 ep57 전처리(anisotropic squash)와 평가 경로 정합을 위해 모든 arm 에서 동일하게 켠다 — DiffPnP **loss** 는 꺼져 있다.


## 1. 관찰 — arm 별 절대 지표

```
arm   subset              n   far_2d_median_px  matched_2d_median_px  gross_rate  yaw_median_deg  reproj_median_px  pose_success_rate  median_detected_corners  gt_inframe_detection_rate
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ep57  F2_CONFIDENT_WRONG  35  43.344            22.381                0.550       9.945           40.148            1.000              8.000                    0.970                    
ep57  F1_NO_RESPONSE      24  27.885            24.441                0.610       7.697           57.608            0.292              0.000                    0.219                    
ep57  F5_MIXED            23  12.179            10.792                0.129       2.070           9.980             1.000              8.000                    0.967                    
ep57  ALL                 87  23.846            13.656                0.408       6.025           23.162            0.805              8.000                    0.784                    
M0_B  F2_CONFIDENT_WRONG  35  43.344            22.381                0.550       9.945           40.148            1.000              8.000                    0.970                    
M0_B  F1_NO_RESPONSE      24  27.885            24.441                0.610       7.697           57.608            0.292              0.000                    0.219                    
M0_B  F5_MIXED            23  12.179            10.792                0.129       2.070           9.980             1.000              8.000                    0.967                    
M0_B  ALL                 87  23.846            13.656                0.408       6.025           23.162            0.805              8.000                    0.784                    
B1    F2_CONFIDENT_WRONG  35  43.342            22.367                0.550       10.200          40.024            1.000              8.000                    0.970                    
B1    F1_NO_RESPONSE      24  27.814            24.461                0.610       7.698           57.615            0.292              0.000                    0.219                    
B1    F5_MIXED            23  12.168            10.829                0.129       2.048           9.848             1.000              8.000                    0.967                    
B1    ALL                 87  23.846            13.652                0.408       5.973           23.280            0.805              8.000                    0.784                    
M0_A  F2_CONFIDENT_WRONG  35  44.020            21.734                0.561       10.946          40.124            1.000              8.000                    0.970                    
M0_A  F1_NO_RESPONSE      24  25.759            24.464                0.610       7.752           57.699            0.292              0.000                    0.219                    
M0_A  F5_MIXED            23  12.392            10.742                0.123       1.995           9.864             1.000              8.000                    0.967                    
M0_A  ALL                 87  24.051            13.719                0.411       5.821           22.602            0.805              8.000                    0.784                    
A1    F2_CONFIDENT_WRONG  35  43.763            21.792                0.557       9.973           40.610            1.000              8.000                    0.970                    
A1    F1_NO_RESPONSE      24  26.154            24.489                0.610       7.701           57.723            0.292              0.000                    0.219                    
A1    F5_MIXED            23  12.392            10.734                0.123       1.953           9.911             1.000              8.000                    0.967                    
A1    ALL                 87  24.059            13.626                0.409       6.089           23.022            0.805              8.000                    0.784                    
```

## 2. M0 fine-tuning effect (ep57 대비 control)

- **M0_B** (F2_CONFIDENT_WRONG, N=35): far 2D 43.344 → 43.344 px, yaw 9.945 → 9.945°, pose success 1.000 → 1.000, GT-in-frame 검출률 0.970 → 0.970
- **M0_A** (F1_NO_RESPONSE, N=24): far 2D 27.885 → 25.759 px, yaw 7.697 → 7.752°, pose success 0.292 → 0.292, GT-in-frame 검출률 0.219 → 0.219

[확인] 이 차이가 **400장 추가 fine-tuning 자체의 효과**다.  후보의 개선은 이 control 대비로만 계산한다.


## 3. B1 결과 — FAIL

primary:
```
metric                       value 
───────────────────────────────────
far_2d_median_reduction_pct  0.005 
gross_rate_reduction_pp      0.000 
yaw_median_reduction_pct     -2.560
reproj_median_reduction_pct  0.307 
matched_2d_reduction_pct     0.065 
```

guards (regression 방지):
```
guard                         value 
────────────────────────────────────
F1_pose_success_drop_pp       0.000 
F5_pose_success_drop_pp       0.000 
clean_far_error_increase_pct  -0.086
detection_drop_pp             0.000 
```

[확인] primary_pass=False, guard_pass=True → **FAIL**


## 3. A1 결과 — FAIL

primary:
```
metric                       value
──────────────────────────────────
gt_inframe_recovery_pp       0.000
median_detected_corner_gain  0.000
pose_success_gain_pp         0.000
far_detection_gain_pp        0.000
```

guards (regression 방지):
```
guard                        value 
───────────────────────────────────
non_F1_detection_drop_pp     0.000 
non_F1_pose_success_drop_pp  0.000 
F2_error_increase_pct        -0.583
```

[확인] primary_pass=False, guard_pass=True → **FAIL**


## 4. Paired delta + session-cluster bootstrap CI

```
comparison  subset              metric                     reference_median  candidate_median  percent_delta  paired_mean_delta  ci_low  ci_high  n 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ep57->M0_B  ALL                 far_2d_median_px           23.846            23.846            0.000          0.000              0.000   0.000    87
ep57->M0_B  ALL                 yaw_err_deg                6.025             6.025             0.000          0.000              0.000   0.000    87
ep57->M0_B  ALL                 reproj_fixed_gt_px         23.162            23.162            0.000          0.000              0.000   0.000    87
ep57->M0_B  ALL                 pose_success               1.000             1.000             0.000          0.000              0.000   0.000    87
ep57->M0_B  ALL                 gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    87
ep57->M0_B  F1_NO_RESPONSE      far_2d_median_px           27.885            27.885            0.000          0.000              0.000   0.000    24
ep57->M0_B  F1_NO_RESPONSE      yaw_err_deg                7.697             7.697             0.000          0.000              0.000   0.000    24
ep57->M0_B  F1_NO_RESPONSE      reproj_fixed_gt_px         57.608            57.608            0.000          0.000              0.000   0.000    24
ep57->M0_B  F1_NO_RESPONSE      pose_success               0.000             0.000             -              0.000              0.000   0.000    24
ep57->M0_B  F1_NO_RESPONSE      gt_inframe_detection_rate  0.000             0.000             -              0.000              0.000   0.000    24
ep57->M0_B  F2_CONFIDENT_WRONG  far_2d_median_px           43.344            43.344            0.000          0.000              0.000   0.000    35
ep57->M0_B  F2_CONFIDENT_WRONG  yaw_err_deg                9.945             9.945             0.000          0.000              0.000   0.000    35
ep57->M0_B  F2_CONFIDENT_WRONG  reproj_fixed_gt_px         40.148            40.148            0.000          0.000              0.000   0.000    35
ep57->M0_B  F2_CONFIDENT_WRONG  pose_success               1.000             1.000             0.000          0.000              0.000   0.000    35
ep57->M0_B  F2_CONFIDENT_WRONG  gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    35
M0_B->B1    ALL                 far_2d_median_px           23.846            23.846            0.003          -0.011             -0.019  -0.006   87
M0_B->B1    ALL                 yaw_err_deg                6.025             5.973             -0.870         -0.124             -0.498  0.102    87
M0_B->B1    ALL                 reproj_fixed_gt_px         23.162            23.280            0.510          -0.201             -0.764  0.100    87
M0_B->B1    ALL                 pose_success               1.000             1.000             0.000          0.000              0.000   0.000    87
M0_B->B1    ALL                 gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    87
M0_B->B1    F1_NO_RESPONSE      far_2d_median_px           27.885            27.814            -0.255         -0.018             -0.036  -0.004   24
M0_B->B1    F1_NO_RESPONSE      yaw_err_deg                7.697             7.698             0.006          -0.009             -0.028  0.001    24
M0_B->B1    F1_NO_RESPONSE      reproj_fixed_gt_px         57.608            57.615            0.012          -0.001             -0.026  0.015    24
M0_B->B1    F1_NO_RESPONSE      pose_success               0.000             0.000             -              0.000              0.000   0.000    24
M0_B->B1    F1_NO_RESPONSE      gt_inframe_detection_rate  0.000             0.000             -              0.000              0.000   0.000    24
M0_B->B1    F2_CONFIDENT_WRONG  far_2d_median_px           43.344            43.342            -0.005         -0.004             -0.013  0.011    35
M0_B->B1    F2_CONFIDENT_WRONG  yaw_err_deg                9.945             10.200            2.560          -0.264             -0.734  0.214    35
M0_B->B1    F2_CONFIDENT_WRONG  reproj_fixed_gt_px         40.148            40.024            -0.307         -0.399             -1.133  0.293    35
M0_B->B1    F2_CONFIDENT_WRONG  pose_success               1.000             1.000             0.000          0.000              0.000   0.000    35
M0_B->B1    F2_CONFIDENT_WRONG  gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    35
ep57->M0_A  ALL                 far_2d_median_px           23.846            24.051            0.859          0.953              -0.281  3.277    87
ep57->M0_A  ALL                 yaw_err_deg                6.025             5.821             -3.381         0.364              -0.085  1.192    87
ep57->M0_A  ALL                 reproj_fixed_gt_px         23.162            22.602            -2.415         0.297              -0.118  1.041    87
ep57->M0_A  ALL                 pose_success               1.000             1.000             0.000          0.000              0.000   0.000    87
ep57->M0_A  ALL                 gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    87
ep57->M0_A  F1_NO_RESPONSE      far_2d_median_px           27.885            25.759            -7.625         -0.297             -0.641  -0.044   24
ep57->M0_A  F1_NO_RESPONSE      yaw_err_deg                7.697             7.752             0.708          -0.082             -0.308  0.015    24
ep57->M0_A  F1_NO_RESPONSE      reproj_fixed_gt_px         57.608            57.699            0.159          -0.025             -0.309  0.311    24
ep57->M0_A  F1_NO_RESPONSE      pose_success               0.000             0.000             -              0.000              0.000   0.000    24
ep57->M0_A  F1_NO_RESPONSE      gt_inframe_detection_rate  0.000             0.000             -              0.000              0.000   0.000    24
ep57->M0_A  F2_CONFIDENT_WRONG  far_2d_median_px           43.344            44.020            1.559          2.143              -0.326  4.928    35
ep57->M0_A  F2_CONFIDENT_WRONG  yaw_err_deg                9.945             10.946            10.063         0.773              -0.120  1.726    35
ep57->M0_A  F2_CONFIDENT_WRONG  reproj_fixed_gt_px         40.148            40.124            -0.058         0.646              -0.205  1.534    35
ep57->M0_A  F2_CONFIDENT_WRONG  pose_success               1.000             1.000             0.000          0.000              0.000   0.000    35
ep57->M0_A  F2_CONFIDENT_WRONG  gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    35
M0_A->A1    ALL                 far_2d_median_px           24.051            24.059            0.034          0.096              -0.032  0.386    87
M0_A->A1    ALL                 yaw_err_deg                5.821             6.089             4.590          -0.121             -0.348  0.009    87
M0_A->A1    ALL                 reproj_fixed_gt_px         22.602            23.022            1.858          0.053              -0.020  0.172    87
M0_A->A1    ALL                 pose_success               1.000             1.000             0.000          0.000              0.000   0.000    87
M0_A->A1    ALL                 gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    87
M0_A->A1    F1_NO_RESPONSE      far_2d_median_px           25.759            26.154            1.535          0.061              0.010   0.119    24
M0_A->A1    F1_NO_RESPONSE      yaw_err_deg                7.752             7.701             -0.663         0.014              -0.002  0.050    24
M0_A->A1    F1_NO_RESPONSE      reproj_fixed_gt_px         57.699            57.723            0.041          0.046              -0.021  0.147    24
M0_A->A1    F1_NO_RESPONSE      pose_success               0.000             0.000             -              0.000              0.000   0.000    24
M0_A->A1    F1_NO_RESPONSE      gt_inframe_detection_rate  0.000             0.000             -              0.000              0.000   0.000    24
M0_A->A1    F2_CONFIDENT_WRONG  far_2d_median_px           44.020            43.763            -0.583         0.178              -0.071  0.820    35
M0_A->A1    F2_CONFIDENT_WRONG  yaw_err_deg                10.946            9.973             -8.888         -0.259             -0.508  0.005    35
M0_A->A1    F2_CONFIDENT_WRONG  reproj_fixed_gt_px         40.124            40.610            1.209          0.073              -0.040  0.369    35
M0_A->A1    F2_CONFIDENT_WRONG  pose_success               1.000             1.000             0.000          0.000              0.000   0.000    35
M0_A->A1    F2_CONFIDENT_WRONG  gt_inframe_detection_rate  1.000             1.000             0.000          0.000              0.000   0.000    35
```

[주의] 소표본(F2 N=35, F1 N=24)이라 CI 가 넓다.  gate 방향성과 session 일관성(`session_deltas.csv`, `session_delta.png`)을 함께 본다.


## 4b. FAIL 의 원인 분해

B1 이 **자기 학습 도메인에서** far 를 고쳤는가:

```
phase   subset  n    far_2d_median_px
─────────────────────────────────────
before  hard    100  4.566           
before  clean   100  1.917           
after   hard    100  4.546           
after   clean   100  1.951           
```

[확인] 학습 도메인(synthetic hard)에서조차 far 오차가 거의 안 줄었다.  residual head 가 위치를 못 고치는 게 아니라 **고칠 오차가 없다** — 학습 신호 부재다 (일반화 실패가 아니다).


A1 의 target semantics 가 F1 response 를 움직였는가:

- [확인] frame median peak paired delta = +0.00178 (평균 +0.00239), 20/24 프레임에서 상승 — **방향은 일관되게 양수**.
- [확인] 그러나 F1 의 median peak 는 0.099 이고 검출 임계값은 0.3 이다.  +0.002 수준의 상승으로는 임계를 넘길 수 없어 GT-in-frame 검출률이 0.2188 로 **비트 단위 동일**하게 유지된다.
- [판정] 즉 A1 의 primary gate 지표는 이 표본에서 **사실상 상수**라 판별력이 없었다.  A1 FAIL 은 '효과 0' 이 아니라 '이 지표·이 표본으로는 검출 불가' 다.


## 5. 그림

- `B1_far_error_curve.png`
- `B1_residual_magnitude.png`
- `A1_inframe_response_recovery.png`
- `failure_class_delta.png`
- `session_delta.png`
- `example_overlays/ (4 panels)`

예시 패널은 같은 frame 의 GT / ep57 / matched control / candidate 를 나란히 보여준다 (`example_overlays/`).


## 6. 남은 불확실성

- [확인] 1 seed, 400장, 최대 10 epoch 의 **스크리닝**이다.  통과 후보라도 새 공개 데이터셋에서 clean 3-seed 로 다시 검증해야 한다.
- [확인] F2 N=35 / F1 N=24 는 소표본이며 session 수도 적다.
- [확인] mechanism-val 은 학습에 쓰이지 않았지만 **모델 선택(best epoch)**에는 쓰였다.  따라서 여기 수치는 낙관적 상한이다.
- [확인] final-test 는 열지 않았다.

