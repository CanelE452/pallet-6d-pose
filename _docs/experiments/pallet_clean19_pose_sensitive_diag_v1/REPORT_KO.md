# Diagonal pose-sensitive EASY→HARD pilot

## 1. 결론

4 fits × 320 update = 1,280 update 완료. **SYNTH_REAL_GEOMETRY_TRANSFER_GAP**. routing: `SYNTH_ONLY`.
MODERATE CURRENT ADD AUC: 0.353494 → 0.354851. 후보 oracle AUC: 0.406374 → 0.404115. Oracle는 GT를 사용하는 사후 진단이며 배포 성능이 아니다.

## 2. 왜 v1 수식을 바꿨는가

기존 `rᵀHr`는 좌표 간 교차항으로 상쇄된다. H=[[1,1],[1,1]], r=[1,-1]이면 기존 값0, 대각 값2. 이번에는 `sum(diag(H)*r²*mask)/Nvalid`만 사용한다. 양의 가중치에서 각 residual의 직접 gradient는 0 방향이다. 공유 파라미터 때문에 다른 출력까지 보존된다는 뜻은 아니다. 기존 v1 파일은 변경하지 않았다.

## 3. gradient integrity

실제 Trainer seed42/deterministic 설정, TF32/AMP off, 새 R0 3개에서 독립 forward/backward 1회씩. 사전 허용오차 atol=1e-5, rtol=1e-4. OLD→OLD와 OLD→NEW λ0 모두 max_abs=0, max_rel=0, loss/component exact. optimizer0. 이전 v1의 bit-exact STOP을 방법 성능 실패로 해석하지 않는다.

![gradient](figures/01_gradient_repeatability.png)

## 4. synthetic geometry / weights

기존512/512 renderer binding, occurrence projection 최대0.00055742px를 재사용. 두 material 각각 synthetic2560 활성, real2560 비활성, 비활성 synthetic0. valid scalar 평균1 오차 최대1.79e-7. 실사·무시점 가중치/gradient는0, center8은 제외.

![weights](figures/02_pose_weight_by_corner.png)

## 5. lambda / training

λ=0.00299772979899; 기존 TOTAL loss coordinate-head gradient norm=17.931541, 대각=1495.426683; ratio=.25. 실제 head의 pose_head+kpts_head 모듈 객체에서 파라미터를 추출했다.

모두 원래 R0, seed42, 5epoch, batch/nbs16, 640, AdamW, lr1e-4, lrf.1, cosine. material별 기존 S1 RGB/order/target/mask와 정확히 같은5120 occurrences; 각 실사/합성2560. M0는 기존 criterion 그대로, M1만 대각 손실. 평가 중 학습·pseudo refresh·rescue 없음. final last만 사용; 학습 중 validation 호출은 무연산으로 차단했다.

## 6. MODERATE 및 전체 요약

| 집단 | arm | PCK10 | 맞은 코너/전체 | CURRENT ADD AUC | ORACLE AUC (진단) | 선택손실 | AxisAcc |
|---|---|---|---|---|---|---|---|
| ALL300 | M0 | 0.6289 | 1476/2347 | 0.4184 | 0.4656 | 0.0472 | 0.7800 |
| ALL300 | M1 | 0.6331 | 1486/2347 | 0.4148 | 0.4676 | 0.0527 | 0.7733 |
| CLEAN | M0 | 0.7152 | 751/1050 | 0.5977 | 0.6169 | 0.0192 | 0.9242 |
| CLEAN | M1 | 0.7286 | 765/1050 | 0.5885 | 0.6139 | 0.0254 | 0.9242 |
| MODERATE_OCCLUSION | M0 | 0.6632 | 445/671 | 0.3535 | 0.4064 | 0.0529 | 0.7241 |
| MODERATE_OCCLUSION | M1 | 0.6528 | 438/671 | 0.3549 | 0.4041 | 0.0493 | 0.7471 |
| SEVERE_OCCLUSION | M0 | 0.4473 | 280/626 | 0.1961 | 0.2828 | 0.0867 | 0.6049 |
| SEVERE_OCCLUSION | M1 | 0.4521 | 283/626 | 0.1962 | 0.2971 | 0.1009 | 0.5556 |
| PLASTIC | M0 | 0.6008 | 861/1433 | 0.3656 | 0.4402 | 0.0747 | 0.7228 |
| PLASTIC | M1 | 0.6050 | 867/1433 | 0.3661 | 0.4477 | 0.0815 | 0.7011 |
| WOOD | M0 | 0.6729 | 615/914 | 0.5023 | 0.5060 | 0.0036 | 0.8707 |
| WOOD | M1 | 0.6772 | 619/914 | 0.4921 | 0.4991 | 0.0070 | 0.8879 |

### 고정 분모 2D 및 6D 상세

| 집단 | arm | PCK5 | PCK20 | median/P90 px | gross20 | detected/matched |
|---|---|---|---|---|---|---|
| ALL300 | M0 | 0.3937 | 0.8130 | 6.90 / 38.68 | 0.1870 | 300/294 |
| ALL300 | M1 | 0.3894 | 0.8228 | 6.80 / 39.57 | 0.1772 | 300/294 |
| CLEAN | M0 | 0.5171 | 0.8924 | 4.75 / 20.63 | 0.1076 | 132/132 |
| CLEAN | M1 | 0.5162 | 0.9019 | 4.81 / 19.66 | 0.0981 | 132/132 |
| MODERATE_OCCLUSION | M0 | 0.3979 | 0.8167 | 6.50 / 100.96 | 0.1833 | 87/86 |
| MODERATE_OCCLUSION | M1 | 0.3875 | 0.8033 | 6.51 / 105.66 | 0.1967 | 87/85 |
| SEVERE_OCCLUSION | M0 | 0.1821 | 0.6757 | 11.65 / 66.71 | 0.3243 | 81/76 |
| SEVERE_OCCLUSION | M1 | 0.1789 | 0.7109 | 11.20 / 57.08 | 0.2891 | 81/77 |

| 집단 | arm | R median/P90 ° | yaw median/P90 ° | t median/P90 cm | IoU3D median/P90 | pose coverage |
|---|---|---|---|---|---|---|
| ALL300 | M0 | 2.471 / 84.959 | 1.305 / 84.895 | 6.639 / 42.812 | 0.619 / 0.843 | 1.0000 |
| ALL300 | M1 | 2.499 / 85.118 | 1.288 / 85.114 | 6.440 / 39.286 | 0.638 / 0.828 | 1.0000 |
| CLEAN | M0 | 1.567 / 4.264 | 0.605 / 2.636 | 4.080 / 13.033 | 0.704 / 0.878 | 1.0000 |
| CLEAN | M1 | 1.634 / 4.853 | 0.672 / 3.149 | 3.658 / 14.961 | 0.714 / 0.869 | 1.0000 |
| MODERATE_OCCLUSION | M0 | 2.761 / 84.176 | 1.454 / 84.081 | 7.037 / 45.507 | 0.578 / 0.819 | 1.0000 |
| MODERATE_OCCLUSION | M1 | 2.552 / 83.513 | 1.377 / 82.976 | 6.902 / 41.903 | 0.585 / 0.806 | 1.0000 |
| SEVERE_OCCLUSION | M0 | 5.585 / 88.357 | 4.464 / 88.354 | 13.288 / 108.948 | 0.511 / 0.781 | 1.0000 |
| SEVERE_OCCLUSION | M1 | 7.018 / 88.374 | 6.766 / 88.364 | 11.828 / 107.369 | 0.532 / 0.771 | 1.0000 |

M0 두 재질은 이전 S1 final 가중치와 bit-exact 일치한다. 모델 로딩 시 실행 모듈 `__main__.DiagModel` 이름 해석만 보완했으며, checkpoint 수정·학습 재시작은 없었다.

![moderate candidate](figures/03_m0_m1_moderate_candidate.png)

![moderate current](figures/04_m0_m1_moderate_current.png)

## 7. SEVERE

![severe](figures/05_m0_m1_severe.png)

## 8. CLEAN/source 및 TRAIN

![clean](figures/06_clean_preservation.png)

![source](figures/07_source_heldout.png)

합성 heldout은 기존256장 그대로이며 두 material 학생을 각각 같은256장에 평가했다. 보존기준1pp는 pilot 분기 기준이지 통계적 유의성 기준이 아니다.

| material | arm | heldout PCK10 | 맞은점/전체 | TRAIN Ldiag | TRAIN weighted error | TRAIN pose normalized |
|---|---|---|---|---|---|---|
| PLASTIC | M0 | 0.9147 | 1855/2028 | 104.5772 | 5.5694 | 0.0366 |
| PLASTIC | M1 | 0.9038 | 1833/2028 | 98.4826 | 5.4696 | 0.0373 |
| WOOD | M0 | 0.9137 | 1853/2028 | 200.2435 | 10.4156 | 0.0440 |
| WOOD | M1 | 0.9162 | 1858/2028 | 193.8385 | 10.5911 | 0.0407 |

![train](figures/08_train_diag_loss.png)

고정 synthetic TRAIN Ldiag 감소: {'PLASTIC': True, 'WOOD': True}. 이는 TRAIN fit 진단이며 일반화 성능이 아니다.

![candidate selection](figures/09_candidate_vs_selection_loss.png)

## 9. 성공/실패 사례

### 합성 heldout pose와 해석 보완

합성 heldout256도 원래 renderer pose/K/dimensions와 저장된 2D 정답의 투영을 직접 대조하여 모두 연결했다. 새 pose GT를 추정하지 않았다. 전체 pose 지표는 [SYNTH_HELDOUT_POSE_RESULTS.json](SYNTH_HELDOUT_POSE_RESULTS.json), 근거는 [SOURCE_GEOMETRY_BINDING.json](SOURCE_GEOMETRY_BINDING.json)이다.

| 모델 경로 | R0 ADD AUC | old S1 / M0 ADD AUC | M1 ADD AUC |
|---|---|---|---|
| Plastic | 0.613627 | 0.550686 | 0.550041 |
| Wood | 0.613627 | 0.567154 | 0.567105 |

`SYNTH_ONLY`는 여기서 **고정 synthetic TRAIN의 대각 loss 감소**를 뜻한다. 합성 heldout pose까지 개선됐다는 뜻이 아니다. TRAIN pose error도 plastic은 0.03662→0.03732로 악화, wood는 0.04403→0.04070으로 개선되어 혼재한다. 따라서 현재 결과로 대각 가중치가 pose 일반화를 해결했다고 주장할 수 없다.

CLEAN PCK10 correct는 751/1050→765/1050이며, R0에서 맞았던 점을 잃은 수는77→66이다. 하지만 plastic source heldout은1855/2028→1833/2028(-1.0848pp)이어서 사전 보존기준1pp를 소폭 위반했다. Wood source는1853/2028→1858/2028이다. 학습 TRAIN19 manual87점은 plastic44/48, wood39/39로 양쪽 모델이 같았다. 이 TRAIN 점수는 일반화 근거가 아니다.

노란선=학생 2D, 빨간선=현재 선택 pose, 하늘 점선=대안 W/D, 초록십자=평가 정답. 원본 RGB와 M0/M1 전체 영상을 동일 배율로 표시. 개선/손상은 사후 ADD로 선택했으며 랜덤6장을 별도 표시한다.

### moderate_candidate_improved — wood_day_01:027445

![moderate_candidate_improved](figures/case_01_moderate_candidate_improved.jpg)

### moderate_candidate_improved — wood_night_01:033721

![moderate_candidate_improved](figures/case_02_moderate_candidate_improved.jpg)

### moderate_candidate_improved — wood_night_01:032276

![moderate_candidate_improved](figures/case_03_moderate_candidate_improved.jpg)

### moderate_candidate_improved — wood_day_01:002141

![moderate_candidate_improved](figures/case_04_moderate_candidate_improved.jpg)

### moderate_candidate_improved — plastic_day_01:020955

![moderate_candidate_improved](figures/case_05_moderate_candidate_improved.jpg)

### moderate_current_improved — wood_night_01:033721

![moderate_current_improved](figures/case_06_moderate_current_improved.jpg)

### moderate_current_improved — eval_pallet07:1778652142480077056

![moderate_current_improved](figures/case_07_moderate_current_improved.jpg)

### moderate_current_improved — plastic_day_01:020954

![moderate_current_improved](figures/case_08_moderate_current_improved.jpg)

### moderate_current_improved — wood_day_01:027445

![moderate_current_improved](figures/case_09_moderate_current_improved.jpg)

### moderate_current_improved — wood_day_01:002141

![moderate_current_improved](figures/case_10_moderate_current_improved.jpg)

### moderate_harmed — wood_183705:001263

![moderate_harmed](figures/case_11_moderate_harmed.jpg)

### moderate_harmed — wood_night_01:032494

![moderate_harmed](figures/case_12_moderate_harmed.jpg)

### moderate_harmed — wood_day_01:004294

![moderate_harmed](figures/case_13_moderate_harmed.jpg)

### moderate_harmed — plastic_day_01:011067

![moderate_harmed](figures/case_14_moderate_harmed.jpg)

### moderate_harmed — wood_183705:000828

![moderate_harmed](figures/case_15_moderate_harmed.jpg)

### severe_improved — eval_night09:1779449602689248000

![severe_improved](figures/case_16_severe_improved.jpg)

### severe_improved — eval_night08:1779449496875356416

![severe_improved](figures/case_17_severe_improved.jpg)

### severe_improved — eval_outside:1778651650570397184

![severe_improved](figures/case_18_severe_improved.jpg)

### severe_improved — eval_night09:1779449638581035008

![severe_improved](figures/case_19_severe_improved.jpg)

### severe_improved — eval_pallet09:1778653804674198784

![severe_improved](figures/case_20_severe_improved.jpg)

### severe_harmed — eval_outside:1778653367706938112

![severe_harmed](figures/case_21_severe_harmed.jpg)

### severe_harmed — eval_night09:1779449643284402176

![severe_harmed](figures/case_22_severe_harmed.jpg)

### severe_harmed — eval_outside:1778653508779767808

![severe_harmed](figures/case_23_severe_harmed.jpg)

### severe_harmed — eval_pallet09:1778653713962971904

![severe_harmed](figures/case_24_severe_harmed.jpg)

### severe_harmed — eval_night09:1779449575470221824

![severe_harmed](figures/case_25_severe_harmed.jpg)

### random_control — eval_cad:1778653056137140480

![random_control](figures/case_26_random_control.jpg)

### random_control — plastic_day_01:013476

![random_control](figures/case_27_random_control.jpg)

### random_control — wood_night_01:029837

![random_control](figures/case_28_random_control.jpg)

### random_control — wood_night_01:030771

![random_control](figures/case_29_random_control.jpg)

### random_control — wood_night_01:032781

![random_control](figures/case_30_random_control.jpg)

### random_control — wood_night_01:032890

![random_control](figures/case_31_random_control.jpg)

## 10. 객관적 판정

PRIMARY_BOTTLENECK_AFTER: `SYNTH_REAL_GEOMETRY_TRANSFER_GAP`

SECONDARY_BOTTLENECK_AFTER: `RECOVERY_PRESERVATION_TRADEOFF`

## 11. 다음 딱 한 실험

평가300 밖 실사에서 신뢰 가능한 pose/relative geometry 감독을 확보·검증하는 단일 provenance 실험. 기존 DEV를 학습에 추가하지 않음.

## 12. 한계

single seed; 같은 세션에서 재사용한 DEV300; 독립 물리적 pose GT가 아닌 geometry-resolved reference; diagonal local approximation; Linear-Covariance 완전 재현 아님; 좌표 분리 loss는 공유 신경망 출력 보존을 보장하지 않음. PCK 상승을 pose 성공으로 치환하지 않으며, oracle을 실제 성능으로 보고하지 않는다.

## 13. 재현

HEAD_BEFORE: `e77944c460d852eda3bf48622b21222b39875079`

입력 hash: [INPUT_BINDINGS.json](INPUT_BINDINGS.json). 손실/gradient: [GPU_GRADIENT_PARITY.json](GPU_GRADIENT_PARITY.json). 전체 material×severity/session/common-matched 및 2D/6D 분포: [RESULTS.json](RESULTS.json). 코너 유지/손실: [TRANSITIONS.json](TRANSITIONS.json). 학습: FIT_*.json 및 [TRAINING_PARITY.json](TRAINING_PARITY.json). 이 문서를 포함하는 Git 커밋이 재현 코드 버전이다. 대용량 checkpoint/cache는 push 대상에서 제외했다.
