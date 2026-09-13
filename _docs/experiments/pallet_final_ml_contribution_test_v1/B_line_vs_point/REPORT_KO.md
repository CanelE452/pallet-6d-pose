# 마지막 ML contribution 대조 실험

최종 architecture 판정: `LOCAL_REFINEMENT_ONLY_SIGNAL`. DEV-only이며 independent confirmation·novelty 주장은 없다.

## 무엇을 비교했는가

R0는 기존 frozen YOLO26n이다. L은 기존 `image_line_only` seed 1/2/3이며 재학습하지 않았다. P는 구조 정보를 쓰지 않는 새 point-local voting head 하나다. 기존 line 실험 primary `image_joint`의 실패는 변경하지 않았다.

P/L trainable params는 18,962/19,810개(-4.28%). 각각 8 roles × 221 non-null candidates × 32 samples × P3/P4로 113,152 spatial samples다. Null은 추가 sampling 없이 context를 사용한다. 파라미터·sample 수 일치는 FLOPs/latency 일치가 아니다.
같은 cached FP16 P3/P4, 같은 55,915 matched usable train rows, 같은 seed별 96,000 exposures, 6,000 updates × 3 fits, batch16, FP32 head/no AMP, AdamW/warmup/cosine/clip을 재사용했다. 학습 source·목록·최종 sampler 상태로 old/new order parity를 검증했다.

P의 isotropic 4×8 patch와 L의 along-line evidence encoder는 다르다. P target은 Gaussian point displacement, L target은 bilinear line geometry/null이다. 동일 point validity를 사용하지만 edge length/support를 point mask로 오용하지 않는다. 기존 L에 C2 min-over-permutation 학습이 없어서 정본 index assignment를 그대로 사용했다. 이 차이는 matching 한계이며 새 symmetry-invariance 주장을 하지 않는다.

## Synthetic 선택과 heldout

P T(seed1/2/3) = [1.0, 1.0, 1.0]; shared lambda=1.0, cap=0.01. 기존 T/lambda/cap grid와 tie rule만 사용했다. Selection은 기존 all-source-GT-denominator 8-corner capped/diagonal-normalized frame score이며, 아래 9kp heldout score로 바꾸지 않았다.
Train55980/cal1004/selection1031/heldout1985의 원래 분할을 유지했다. P selection artifact를 저장한 뒤에만 heldout accuracy를 계산했고, real 결과로 조정하지 않았다. 원래 R0·과거 probe의 validation 재사용 때문에 전체 연구에 독립적인 test라는 뜻은 아니다.

| 모델 | synth9 median px | synth9 P90 px | PCK10/allGT | 원래8 selection-objective heldout |
|---|---:|---:|---:|---:|
| R0 | 1.8969 | 6.8849 | 0.9382 | 0.00911675 |
| L1 | 1.7418 | 6.5449 | 0.9415 | 0.00889021 |
| L2 | 1.7466 | 6.5196 | 0.9421 | 0.00888746 |
| L3 | 1.7551 | 6.5000 | 0.9414 | 0.00890407 |
| P1 | 1.7356 | 6.6217 | 0.9414 | 0.00889591 |
| P2 | 1.7501 | 6.6550 | 0.9410 | 0.00892081 |
| P3 | 1.7368 | 6.6159 | 0.9417 | 0.00890396 |

Radius coverage, movement, null probability, point availability는 `SYNTH_HELDOUT_RESULTS.json`에 함께 보존했다. R0의 dormant P1 null 필드는 해당 없음으로, L의 null은 별도 계산해 `SYNTH_DIAGNOSTIC_FIELD_CLARIFICATION.json`에 명시했다. 초기 generic permutation 테스트의 C2 yaw 이름 오류와 학습 후 추가 정본 검사도 `C2_TEST_LABEL_CORRECTION.md`에 공개했다. 어느 정정도 학습/예측/선택을 바꾸지 않는다. 이 진단으로 radius/stencil/model을 다시 설계하지 않았다.

## Real DEV319: 고정한 일회 비교

P 3 seeds 각각 positive319+negative2689에서 실제 image inference를 수행했다. 모든 candidate의 box/score/order, 최고 score 선택, nonselected points와 center8 보존을 exact 검사했다. Negative detection/AP/ranking은 동일하다. 실제 negative P 점 출력도 별도 저장하되, 기존 canonical scorer는 negative point supervision이 없으므로 그 부분만 baseline raw cache를 재사용한다.

| 모델 | 9kp median px | P90 px | frame mean px | gross20 | rotation med ° | translation med cm | IoU3D med | ADDsym AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 6.6157 | 38.6700 | 21.6734 | 0.1720 | 2.2625 | 7.8969 | 0.6032 | 0.4285 |
| L1 | 6.0518 | 37.2041 | 21.1446 | 0.1607 | 2.1100 | 7.6092 | 0.6374 | 0.4500 |
| L2 | 6.0373 | 37.3017 | 21.0853 | 0.1593 | 2.1031 | 7.3644 | 0.6361 | 0.4519 |
| L3 | 6.1207 | 37.4530 | 21.1644 | 0.1600 | 2.1358 | 7.6719 | 0.6250 | 0.4450 |
| P1 | 5.8509 | 38.0254 | 20.9305 | 0.1557 | 2.0892 | 6.8647 | 0.6422 | 0.4535 |
| P2 | 5.9427 | 37.7279 | 21.0096 | 0.1589 | 2.1059 | 6.9167 | 0.6374 | 0.4561 |
| P3 | 5.9211 | 37.5460 | 20.9486 | 0.1542 | 2.0731 | 7.6775 | 0.6371 | 0.4541 |

Primary seed-mean L−P median difference = +0.165021px; paired13-session 95% CI [-0.009842, +0.381123], paired-frame CI [+0.042911, +0.286646]. 10,000 draws, seed=20260913. 각 draw에서 같은 frame/session ID를 모든 seed에 적용하고, seed별 metric(L)-metric(P)의 평균을 계산했다. R0를 3배 독립 표본으로 세지 않았다.
L/P seed-mean P90 = 37.319600/37.766432px; gross20 = 0.160015/0.156265. L/P median = 6.069932/5.904910px; R0=6.615678px.
Proj@5/10/20는 supervised keypoint 오차가 threshold 이내인 비율이다. 모든9점이 모든 프레임에서 supervised인 것은 아니다. Canonical evaluator/GT/reference를 그대로 사용하고 full-precision point 오류를 재계산해 기존 JSON 수치와 일치시켰다. Yaw·coverage·Proj는 `PER_SEED_REAL.json`에 포함했다. Lateral/depth는 기존 frozen per-frame pose export에 없어 새 지표를 추가하지 않았다.

| Gate | 결과 |
|---|---|
| G1 | FAIL |
| G2 | PASS |
| G3 | FAIL |
| G4 | PASS |
| G5 | FAIL |

G5는 6D 우월성 확증이 아니라 downstream 안전성이다. 아래 차이는 모두 L−P다.

| Downstream | seed-mean difference | session95% low | high | L better direction | clear L harm |
|---|---:|---:|---:|---|---|
| rotation_median_deg | +0.026932 | -0.085539 | +0.197677 | False | False |
| translation_median_cm | +0.395514 | -0.395917 | +0.839496 | False | False |
| iou3d_median | -0.006052 | -0.025673 | +0.013808 | False | False |
| add_sym_auc | -0.005573 | -0.019333 | +0.007051 | False | False |

## Mechanism 및 비용

GT-assisted edge-endpoint 평균 normal error의 L−P median delta=-0.012641px; tangent delta=+0.156822px. 이는 기전 진단일 뿐 primary/gates를 뒤집지 않는다. Role/day-night/material/session 및 사전 고정 R0 mean error ≤5 / 5–10 / >10 px subgroup을 전부 보존했다.
Visibility는 annotation code에 따른 기술적 subgroup이며 physical observed/occluded provenance를 독립 검증하지 못했다. 물리적 occlusion robustness 주장은 하지 않는다.

Runtime은 기존과 같은26장·5 warmup·3 repeats·batch1이며 baseline/integrated를 교대로 측정했다. BGR→reflect padding→YOLO→refinement→원본2D좌표, 동기화된 wall latency다. File decode/model load/PnP는 제외했다. Timing output과 accuracy output의 좌표도 exact 비교했다.

| 모델 | median ms | mean ms | P90 ms | paired added median ms | head used |
|---|---:|---:|---:|---:|---:|
| L1 | 19.680 | 19.274 | 22.047 | +10.333 | 1.000 |
| L2 | 19.548 | 19.587 | 22.139 | +9.542 | 1.000 |
| L3 | 17.643 | 17.988 | 20.035 | +8.036 | 1.000 |
| P1 | 13.822 | 14.094 | 16.409 | +4.273 | 1.000 |
| P2 | 12.984 | 13.928 | 16.369 | +4.227 | 1.000 |
| P3 | 13.853 | 13.978 | 16.301 | +3.974 | 1.000 |
| R0 paired observations | 9.547 | 10.444 | 12.806 | 0 | 0 |

비용 수치는 현재 GPU/환경의 측정치다. Runtime 시작부에 CPU 점수 계산이 병행되어 CPU 부하가 완전히 격리되지는 않았다(`RUNTIME_CONDITIONS_NOTE.json`). 따라서 관측 wall-time 차이를 통제된 인과적 속도 향상으로 주장하지 않는다. 반복을 골라내거나 더 빠른 값을 얻기 위한 재측정은 하지 않았다. FLOP 일치도 주장하지 않는다. GT-assisted mechanism 결과만으로 line 필요성을 증명할 수 없다.

## 감사와 종료

최종 source/체크포인트/seed-order/metric/bootstrap/runtime/출력 보존 감사는 `FINAL_AUDIT.json`을 따른다. Audit PASS는 performance PASS가 아니다. 원래 task-risk/active/line 결과 및 paper/final은 수정하지 않았다.

Line-specific gates 전체를 통과할 때만 새 independent real sessions confirmation이 다음 단계다. 그 외에는 현재 line-specific claim을 닫고 추가 DHT/Hough/module/score 구조 탐색은 하지 않는다. Generic P를 자동으로 새로운 proposed novelty로 승격하지 않는다.

Local refinement is useful, but the current evidence does not establish a line-specific inductive-bias contribution.
