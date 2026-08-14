# PAPER_S2 ep57 — Mechanism Diagnostic Report

진단 전용.  새 모델·새 학습·checkpoint selection 없음.  산출물은 (1) 실패 프레임의 최초 붕괴 단계, (2) 그에 대응하는 architecture/loss 우선순위.

- checkpoint `net_epoch_0057.pth` SHA `c0055fe7c4210f63…`
- git HEAD `97e1219a147e9e1681a4b0019cbd9a72fb2f95c5`
- primary population strict filter-val **N=87** (outside 44 / night 43, truncated 17 / non-truncated 70)
- cache key `8dfcd6c238dab4fa…`, model forwards this build: 87
- final-test open count **0** (sealed sessions fail-closed)


## 1. 관찰

[확인] baseline 재현 게이트 통과: GT-2D PnP 87/87, predicted local-softargmax PnP 70/87, yaw median 6.025°, fixed-GT reproj median 23.162 px.

[확인] 2D correspondence 를 GT 로 전부 바꾸면 87/87 가 풀린다 — solver·K·dimensions 는 병목이 아니다.


## 2. Failure class 분포

```
failure_class          frames  fraction
───────────────────────────────────────
F2_CONFIDENT_WRONG     35      0.402   
F1_NO_RESPONSE         24      0.276   
F5_MIXED               23      0.264   
F3_GEOMETRY_AMPLIFIED  4       0.046   
F4_SOLVER_SPECIFIC     1       0.011   
```

클래스별 프로필 (median):

```
failure_class          n   corners_detected  far_detected  frame_peak  matched_2d_px  far_2d_px  yaw_err_deg  pose_success_rate  truncated_rate
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
F1_NO_RESPONSE         24  0.000             0.000         0.105       24.441         27.885     7.697        0.292              0.500         
F2_CONFIDENT_WRONG     35  8.000             4.000         0.860       22.381         43.344     9.945        1.000              0.143         
F3_GEOMETRY_AMPLIFIED  4   8.000             4.000         0.863       7.653          12.544     8.347        1.000              0.000         
F4_SOLVER_SPECIFIC     1   8.000             4.000         0.685       15.698         15.698     9.243        1.000              0.000         
F5_MIXED               23  8.000             4.000         0.905       10.792         12.179     2.070        1.000              0.000         
```

[확인] F5_MIXED 23 프레임 중 20 개는 pose 성공 + yaw ≤ 5.0° 로 **사실상 실패가 아니다** (catch-all 이 정상 프레임을 흡수한다).  F5 를 '혼합 실패'로 읽으면 안 된다.


### 미검출 corner 는 화면 안이었나 밖이었나 (F1 해석의 전제)

```
failure_class          corner_observations  undetected  undetected_gt_in_frame  undetected_gt_off_image  undetected_gt_sentinel
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
F1_NO_RESPONSE         192                  157         125                     21                       11                    
F2_CONFIDENT_WRONG     280                  10          8                       2                        0                     
F3_GEOMETRY_AMPLIFIED  32                   2           2                       0                        0                     
F4_SOLVER_SPECIFIC     8                    0           0                       0                        0                     
F5_MIXED               184                  6           6                       0                        0                     
```

[확인] F1 프레임에서 검출되지 않은 corner 157개 중 **125개는 GT 가 화면 안**에 있었다 (화면 밖 21, sentinel 11).  프레임 단위로도 15/24 는 미검출 corner 가 **전부 화면 안**이다.
[확인] 즉 F1 은 '화면 밖이라 안 잡힌 것'이 아니라 화면 안에 보이는 corner 에서도 belief 가 안 뜨는 **진짜 response 실패**가 다수다.
[추정] 따라서 Candidate A 를 '화면 밖 supervision 문제'로만 좁히면 안 된다.  outside-aware head 가 직접 겨냥하는 것은 화면 밖 21 + sentinel 11 관측이고, 나머지는 appearance/response 문제다.


도메인·truncation 별:

```
failure_class          domain   is_truncated  frames
────────────────────────────────────────────────────
F1_NO_RESPONSE         night    False         3     
F1_NO_RESPONSE         night    True          8     
F1_NO_RESPONSE         outside  False         9     
F1_NO_RESPONSE         outside  True          4     
F2_CONFIDENT_WRONG     night    False         19    
F2_CONFIDENT_WRONG     night    True          4     
F2_CONFIDENT_WRONG     outside  False         11    
F2_CONFIDENT_WRONG     outside  True          1     
F3_GEOMETRY_AMPLIFIED  night    False         1     
F3_GEOMETRY_AMPLIFIED  outside  False         3     
F4_SOLVER_SPECIFIC     night    False         1     
F5_MIXED               night    False         7     
F5_MIXED               outside  False         16    
```

[확인] threshold sensitivity (×0.75 / ×1.0 / ×1.25):

```
threshold_scale  F2_CONFIDENT_WRONG  F1_NO_RESPONSE  F5_MIXED  F3_GEOMETRY_AMPLIFIED  F4_SOLVER_SPECIFIC
────────────────────────────────────────────────────────────────────────────────────────────────────────
0.750            46.000              24.000          15.000    1.000                  1.000             
1.000            35.000              24.000          23.000    4.000                  1.000             
1.250            28.000              24.000          28.000    5.000                  2.000             
```

## 3. First-break stage

```
first_break_stage            frames  fraction
─────────────────────────────────────────────
REPRESENTATION_LOCALIZATION  33      0.379   
IMAGE_OR_RESPONSE            24      0.276   
POSE_REFINEMENT              14      0.161   
DECODER                      10      0.115   
PNP_GEOMETRY                 3       0.034   
MIXED_OR_UNRESOLVED          3       0.034   
```

## 4. Decoder recovery

```
variant  pose_success  pose_success_delta_pp  yaw_median  delta_yaw_median  matched_2d_median  delta_matched_2d_median  decoder_is_primary_lever
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
D0       70            3.448                  6.025       -0.029            13.656             -0.793                   False                   
D1       70            3.448                  6.025       -0.029            13.656             -0.793                   False                   
D2       67            0.000                  4.760       0.000             15.290             0.000                    False                   
D3       67            0.000                  4.857       -0.255            11.822             -0.463                   False                   
D4       70            3.448                  6.796       0.185             14.279             0.025                    False                   
D5       70            3.448                  6.134       0.063             13.438             -0.300                   False                   
```

[확인] decoder 교체만으로 판정 기준(pose success +5%p / yaw −20% / 2D −5px)을 넘는 변형이 없다 → decoder 는 보조 원인.

[확인] 다만 frame 단위로는 10 프레임에서 어떤 decoder 하나가 baseline 을 회복한다.  population 수준 효과가 0 인데 frame 수준 회복이 존재한다는 것은 체계적 이득이 아니라 PnP candidate 선택의 불안정성에 가깝다 [추정].

[확인] D2(canonical eval decoder)는 D0 대비 pose success 가 3.4%p **낮고**(67 vs 70) yaw median 은 낮다 — 두 값은 서로 다른 프레임 집합 위의 median 이므로 직접 비교하면 안 된다.


## 5. Keypoint / group oracle recovery

```
variant                  recovered  lost  delta_yaw  delta_reproj
─────────────────────────────────────────────────────────────────
O11_all                  17         0     -8.347     -22.561     
O10_all_corners          17         0     -8.282     -21.990     
O4_top                   14         0     -5.526     -12.881     
O5_bottom                9          0     -5.177     -13.151     
O7_depth_right           17         0     -5.084     -14.617     
O6_depth_left            9          0     -3.919     -5.931      
O3_far                   16         0     -2.828     -5.192      
O1_kp7                   2          0     -0.935     -1.900      
O1_kp5                   1          0     -0.860     -0.232      
O1_kp4                   2          0     -0.851     -1.930      
O8_kp5_kp6               1          1     -0.776     -1.970      
A_drop_lowest_conf(kp1)  0          0     -0.544     -0.374      
A_drop_lowest_conf(kp2)  0          0     -0.392     -0.376      
A_drop_lowest_conf(kp3)  0          0     -0.181     -0.085      
O1_kp2                   1          0     -0.178     -5.063      
A_drop_lowest_conf(kp8)  0          0     -0.177     3.089       
A_drop_lowest_conf(kp5)  0          0     -0.156     -0.275      
LOO_kp5                  0          0     -0.120     0.613       
LOO_kp1                  0          0     -0.107     0.003       
A_drop_lowest_conf(kp0)  0          0     -0.104     -1.642      
O1_kp6                   1          0     -0.103     -0.748      
LOO_kp4                  0          0     -0.069     0.421       
LOO_kp2                  0          0     -0.068     0.080       
A_drop_kp5               0          0     -0.036     0.463       
LOO_kp3                  0          0     -0.033     0.272       
O1_kp0                   2          0     -0.031     -0.372      
A_drop_lowest_conf(kp6)  0          0     -0.030     -0.417      
LOO_kp6                  0          0     -0.016     -0.199      
LOO_kp0                  0          0     -0.006     0.232       
A_drop_kp6               0          0     -0.002     -0.190      
A_drop_centroid          0          0     -0.002     0.216       
A_drop_kp8               0          0     -0.002     0.216       
LOO_kp8                  0          0     -0.002     0.216       
A_drop_kp0               0          0     0.000      0.232       
A_drop_kp2               0          0     0.000      0.000       
A_drop_kp4               0          0     0.000      0.421       
O0                       0          0     0.000      0.000       
A_drop_kp1               0          0     0.000      0.000       
A_drop_kp3               0          0     0.000      0.265       
O1_kp1                   1          0     0.024      -4.409      
A_drop_kp7               0          0     0.026      0.000       
O1_kp3                   2          0     0.053      -0.373      
A_drop_lowest_conf(kp4)  0          0     0.061      -0.024      
O2_near                  10         0     0.102      -10.135     
O9_centroid              0          0     0.178      -0.496      
O1_kp8                   0          0     0.178      -0.496      
A_drop_lowest_conf(kp7)  0          0     0.185      -0.832      
LOO_kp7                  0          0     0.450      0.189       
A_drop_far               0          14    2.412      4.976       
```

### failure class 별 최초 회복 intervention

```
failure_class          variant                  recovered_frames  delta_yaw_median  delta_reproj_median
───────────────────────────────────────────────────────────────────────────────────────────────────────
F1_NO_RESPONSE         O11_all                  17                -7.697            -54.036            
F1_NO_RESPONSE         O10_all_corners          17                -7.525            -50.920            
F1_NO_RESPONSE         O7_depth_right           17                -5.084            -17.920            
F2_CONFIDENT_WRONG     O11_all                  0                 -9.945            -38.601            
F2_CONFIDENT_WRONG     O10_all_corners          0                 -9.734            -37.026            
F2_CONFIDENT_WRONG     O7_depth_right           0                 -5.445            -22.919            
F3_GEOMETRY_AMPLIFIED  O11_all                  0                 -8.347            -22.561            
F3_GEOMETRY_AMPLIFIED  O10_all_corners          0                 -8.282            -21.990            
F3_GEOMETRY_AMPLIFIED  A_drop_lowest_conf(kp0)  0                 -7.868            -5.399             
F4_SOLVER_SPECIFIC     O11_all                  0                 -9.243            -16.879            
F4_SOLVER_SPECIFIC     O10_all_corners          0                 -9.230            -16.706            
F4_SOLVER_SPECIFIC     O7_depth_right           0                 -7.684            -14.617            
F5_MIXED               O11_all                  0                 -2.070            -7.135             
F5_MIXED               O10_all_corners          0                 -2.057            -6.710             
F5_MIXED               O4_top                   0                 -0.881            -3.446             
```

## 6. Stage-wise progression (belief stage 1~6)

```
stage_label     group_near_far  n  
───────────────────────────────────
EARLY_WRONG     centroid        33 
EARLY_WRONG     far             184
EARLY_WRONG     near            125
LATE_DRIFT      centroid        12 
LATE_DRIFT      far             40 
LATE_DRIFT      near            28 
RECOVERED_LATE  centroid        36 
RECOVERED_LATE  far             80 
RECOVERED_LATE  near            64 
SHARPEN_ONLY    centroid        2  
SHARPEN_ONLY    far             19 
SHARPEN_ONLY    near            35 
STABLE          centroid        4  
STABLE          far             22 
STABLE          near            86 
UNRESOLVED      far             3  
UNRESOLVED      near            10 
```

stage 별 median 위치오차 / peak (detected keypoint):

```
group     stage  n    median_err_px  err_delta_vs_stage1_px  median_peak  peak_ratio_vs_stage1
──────────────────────────────────────────────────────────────────────────────────────────────
near      1      249  9.984          0.000                   0.545        1.000               
near      2      249  8.197          -1.787                  0.692        1.271               
near      3      249  8.004          -1.980                  0.718        1.319               
near      4      249  7.493          -2.490                  0.813        1.492               
near      5      249  6.869          -3.114                  0.848        1.556               
near      6      249  6.885          -3.099                  0.868        1.592               
far       1      272  31.888         0.000                   0.383        1.000               
far       2      272  24.305         -7.584                  0.481        1.256               
far       3      272  22.349         -9.540                  0.663        1.730               
far       4      272  21.419         -10.469                 0.761        1.987               
far       5      272  21.611         -10.278                 0.839        2.191               
far       6      272  22.077         -9.812                  0.855        2.232               
centroid  1      74   32.883         0.000                   0.344        1.000               
centroid  2      74   21.540         -11.343                 0.585        1.702               
centroid  3      74   20.255         -12.628                 0.745        2.166               
centroid  4      74   17.169         -15.714                 0.800        2.327               
centroid  5      74   16.757         -16.126                 0.884        2.571               
centroid  6      74   16.939         -15.944                 0.903        2.626               
```

[확인] EARLY_WRONG 342 vs LATE_DRIFT 80 (EARLY_WRONG = stage1 과 stage6 모두 오차 >20px, LATE_DRIFT = stage1 ≤10px 인데 stage6 >20px).  오류의 대부분은 refinement 가 **만들어내는** 것이 아니라 stage 1 에 이미 있고 끝까지 남는다.
[확인] far face median 오차: stage1 31.9px → stage4 21.4px (최저) → stage6 22.1px.  즉 stage 4 이후로는 더 줄지 않고 오히려 소폭 되돌아간다.  같은 구간에서 peak 는 0.76 → 0.86 로 계속 오른다 (stage1 대비 2.2배).
[확인] 최종 far 오차 22.1px 는 near 6.9px 의 3.2배다.  near 도 stage 5 이후 정체하지만 정체 수준 자체가 다르다.
[추정] 후반 refinement 는 far face 에서 **위치를 더 고치지 못하고 신뢰도만 올린다**.  이것이 far/rear 가 confidently-wrong 으로 나타나는 기전과 정합한다.


## 7. Same-image counterfactual

```
variant  n   corners_detected  pose_success_rate  matched_2d_median  far_2d_median  yaw_median
──────────────────────────────────────────────────────────────────────────────────────────────
C0       24  8.000             0.958              14.907             36.751         4.322     
C1       24  4.000             0.542              44.354             73.006         22.321    
C2       24  6.500             0.833              39.754             71.185         17.279    
C3       24  2.000             0.333              35.680             80.398         17.435    
C4       24  4.000             0.625              52.800             88.534         16.357    
C5       24  8.000             0.875              30.427             73.226         10.359    
C6       24  6.000             0.792              39.990             76.271         31.502    
C7       24  7.500             0.792              23.780             38.648         7.702     
C8       24  7.500             0.625              33.906             58.909         17.954    
C9       24  8.000             0.958              10.869             18.248         8.533     
C10      24  2.500             0.458              14.638             58.462         5.480     
C11      24  8.000             0.917              14.890             29.866         3.248     
```

[확인] 모든 counterfactual 의 affine/K/GT 정합 오차 최대 5.68e-14 px (< 0.1 px 게이트).


### same-source paired transition (동일 이미지, 한 요소만 변경)

```
depth         variant  kind          n_paired  corners_detected_median  delta_corners_vs_original  pose_success_rate  delta_pose_success_pp  delta_peak_median  delta_far2d_median
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
shallow_left  C1       crop_only     24        4.000                    -2.000                     0.542              -41.667                -0.406             3.238             
shallow_left  C5       reflect_pad   24        8.000                    0.000                      0.875              -8.333                 -0.056             14.147            
shallow_left  C7       constant_pad  24        7.500                    0.000                      0.792              -16.667                -0.089             4.770             
deep_left     C3       crop_only     24        2.000                    -6.000                     0.333              -62.500                -0.578             51.961            
deep_left     C6       reflect_pad   24        6.000                    0.000                      0.792              -16.667                -0.199             39.117            
deep_left     C8       constant_pad  24        7.500                    0.000                      0.625              -33.333                -0.272             24.305            
```

[확인] shallow_left: crop-only 는 원본 대비 corner 검출 -2개, pose success -42%p, peak -0.41.  같은 crop 을 원래 캔버스로 reflect-pad 하면 검출 +0개 (= 원본 수준 복귀), pose success -8%p, constant-pad 는 -17%p.

[확인] deep_left: crop-only 는 원본 대비 corner 검출 -6개, pose success -62%p, peak -0.58.  같은 crop 을 원래 캔버스로 reflect-pad 하면 검출 +0개 (= 원본 수준 복귀), pose success -17%p, constant-pad 는 -33%p.

[확인] 그러나 padding 이 회복하는 것은 **response** 이고, far-face 2D 오차는 모든 padding 변형에서 원본보다 나쁘다 (delta_far2d 전부 양수).
[주의] far2d/matched 2D 중앙값은 검출된 corner 수가 변형마다 달라 짝이 달라진다 — crop-only 처럼 corner 가 2개만 남은 경우의 2D 값은 검출된 소수 점에 대한 값이므로 변형 간 직접 비교는 제한적이다.  검출 수·pose success·peak 가 더 견고한 비교축이다.
[추정] 현재 모델은 실제 frame-edge truncation 에 취약하고 reflect-pad 가 그 실패를 가린다는 frozen-response 증거다.  matched retraining 전까지 '현재 augmentation 이 학습 실패의 인과 원인'이라고는 말할 수 없다.


[확인] 같은 이미지에서 blur 만 준 C10 은 corner 검출 median 2.5 로 붕괴하지만, luma 만 낮춘 C11 (8.0) 과 0.5배 축소한 C9 (8.0) 은 원본 수준을 유지한다 — 어둠·작은 물체 자체는 response 붕괴의 원인이 아니다.


## 8. 지지된 원인

- [확인] 2D correspondence 품질이 지배적 손실이다 — GT 치환으로 87/87 회복하고, 8 corner 만 GT 로 바꿔도 17 실패 프레임이 전부 회복된다.
- [확인] 실패는 두 개의 서로 다른 population 이다: response 자체가 없는 24 프레임(median corner 검출 0개, peak 0.10)와, response 는 강한데 far face 위치가 틀린 35 프레임(peak 0.86, far 2D 43px).
- [확인] 오차는 far/depth face 에 집중된다.  far·depth-right group GT 치환의 회복 프레임 수가 near group 보다 크다 (16/17 vs 10).
- [확인] frame-edge truncation 은 response 를 붕괴시킨다 (same-source crop-only counterfactual).


## 9. 반증된 원인

- [확인] solver·K·dimensions 단독 원인 — GT 2D 아래에서 87/87 이므로 기각.
- [확인] centroid 단독 원인 — O9_centroid 는 실패 프레임을 0개 회복하고 yaw 를 오히려 +0.18° 바꾼다 → 기각.
- [확인] decoder 선택 (training softargmax vs canonical eval vs argmax vs offset 제거) — population 수준에서 판정 기준 미달 → 주원인 아님.
- [확인] **'late-stage refinement 가 위치를 망친다(late drift)'는 가설 기각** — EARLY_WRONG 342 vs LATE_DRIFT 80.  far face 오차는 stage 1 에서 이미 32px 이고, refinement 로 21px 까지만 줄었다가 stage 6 22px 로 정체한다.  따라서 Candidate B 의 근거는 'late drift' 가 아니라 '초기부터 틀린 위치를 refinement 가 끝까지 못 고치고 신뢰도만 올린다' 로 수정되어야 한다.
- [확인] 저조도·작은 물체 자체 — C11(luma)·C9(downscale) counterfactual 에서 response 유지 → 기각.
- [추정] occlusion — true occlusion metadata 가 없어 인과 주장을 하지 않는다 (truncation 과 구분).


## 10-11. Architecture 후보 순위 / 보류·폐기

```
priority  candidate                            target_failure         supporting_frames  supporting_fraction  oracle_recovery_frames  target_class_delta_yaw  target_class_delta_reproj  stage_evidence                                       
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1         B_far_depth_structured_refinement    F2_CONFIDENT_WRONG     35                 0.402                17                      -2.526                  -15.717                    far err stage6/stage1=0.69 while peak x2.2           
2         A_visibility_partial_keypoint        F1_NO_RESPONSE         24                 0.276                17                      -7.611                  -52.478                    EARLY_WRONG fraction (response absent from stage 1)  
3         C_calibrated_uncertainty_robust_pnp  F3_GEOMETRY_AMPLIFIED  4                  0.046                0                       -0.116                  -0.026                     SHARPEN_ONLY fraction (confidence without correction)
4         D_direct_pose_gated_residual         F4_SOLVER_SPECIFIC     1                  0.011                0                       -                       -                          no stage-level signal isolated                       
```

[주의] `oracle_recovery_frames` 는 baseline 이 **실패**한 프레임의 회복 수인데, 실패 17 프레임은 전부 F1 에 있으므로 A·B 모두 17 로 포화된다.  두 후보를 가르는 값은 `target_class_delta_yaw` — 각 후보의 GT 치환이 **자기 target class 안에서** 만드는 yaw 개선량이다.


### 단일 keypoint GT 치환의 frame 단위 일관성

```
variant  n_paired  fraction_improved  median_delta_yaw  recovered_frames
────────────────────────────────────────────────────────────────────────
O1_kp0   70        0.514              -0.001            2               
O1_kp1   70        0.486              0.030             1               
O1_kp2   70        0.486              0.051             1               
O1_kp3   70        0.443              0.030             2               
O1_kp4   70        0.686              -0.850            2               
O1_kp5   70        0.614              -0.344            1               
O1_kp6   70        0.643              -0.355            1               
O1_kp7   70        0.671              -0.719            2               
O1_kp8   70        0.543              -0.039            0               
```

REJECTED / DEFERRED:
- REJECTED `fixed kp5 pixel correction` — kp5 단일 GT 치환은 실패 프레임을 1개만 회복하고, 프레임의 61% 에서만 yaw 가 개선된다.  상수 픽셀 보정으로 재현할 수 있는 일관된 부호가 아니다.
- REJECTED `fixed centroid upward shift` — O9_centroid 는 실패 프레임 0개 회복, yaw 중앙값 +0.18°.
- DEFERRED `raw covariance weighting` — local 7×7 covariance 는 calibration 없이 쓸 수 없다.  게다가 keypoint 제거 계열(A_drop_*, LOO_*) 중 실패 프레임을 회복시킨 변형은 하나도 없고 far face 를 통째로 빼면 14 프레임을 오히려 잃는다 → '나쁜 점을 빼거나 가중치를 낮추는' 접근만으로는 회복이 안 된다.
- REJECTED `solver sweep only` — solver 변경은 GT 2D 아래에서 이미 87/87 이므로 상류를 못 고친다.
- DEFERRED `unconditional pose residual` — F4 표본이 1 프레임뿐이라 direct-pose 근거가 약하다.


## 12. Micro-training 계획

`MICRO_TRAIN_PLAN.md` 참조 (이번 작업에서 학습은 실행하지 않음).


## 13. 남은 불확실성

- [추정] true occlusion metadata 가 없으므로 occlusion 인과 주장은 하지 않는다. 본 보고서의 `truncation` 은 프레임 경계 잘림만 뜻한다.
- [추정] counterfactual 은 frozen response 증거이며, matched retraining 전까지 '현재 augmentation 이 학습 실패의 인과 원인'이라고 말할 수 없다.
- [확인] primary N=87 은 소표본이다. failure class 별 subset 은 더 작으므로 class 별 결론은 예비적이다.
- [확인] manual36 은 exploratory PL-pool 이라 primary 결과와 합치지 않았다.


## 그림

- `failure_class_distribution.png`
- `first_break_stage_distribution.png`
- `oracle_recovery_matrix.png`
- `stage_progression_by_failure.png`
- `far_depth_oracle_recovery.png`
- `decoder_pose_recovery.png`
- `counterfactual_original_crop_reflect.png`
- `error_propagation_graph.png`
- `architecture_evidence_matrix.png`

정성 예시 (GT 초록 / 예측 빨강, stage 예시는 stage1 파랑 / stage6 빨강):
- `failure_class_examples/` — failure class 별 대표 프레임
- `late_drift_examples/` — LATE_DRIFT keypoint 가 많은 프레임
- `error_propagation_examples/` — first-break stage 별 대표 프레임

