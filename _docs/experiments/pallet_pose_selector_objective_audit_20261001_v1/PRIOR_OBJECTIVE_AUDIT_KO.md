# 후보 선택 목적함수 선행 감사

2026-10-01. 저장소의 기존 코드·보고서만 읽었다. 이번 감사가 실행한 fit, 이미지 forward, 실사 GT 열람·채점은 모두 0회다. 단일 RGB와 물리 치수를 사용하는 현재 입력 계약을 유지한다. 근거 파일의 실제 SHA256은 [PRIOR_OBJECTIVE_AUDIT.json](PRIOR_OBJECTIVE_AUDIT.json)에 기록했다. 과거 로컬 산출물은 공개 범위가 달라 아래에서 코드 경로로 표기한다.

**비용 회귀·상대 이득 회귀·soft-target 순위 학습·보존 regret는 이미 여러 번 실행됐다. 이를 일반적으로 미시도 방법이라고 부르면 안 된다.** 다만 조사한 실행에서 현재 R0/DIVERSE의 네 물리 pose와 동일한 `max(T/sT,R/sR)` 비용을 사용한 비용 차이 가중 순위 손실은 찾지 못했다. 이는 저장소 안의 제한된 확인이며 학술적 신규성 주장이나 성공 예측이 아니다.

현재 결과는 [원래 union 실험 보고서](../pallet_pose_union_selection_20261001_v1/REPORT_KO.md)에 따른다. TRAIN oracle gate는 통과했으나 6개 최종 selector의 source VAL gate는 45개 중 42개만 통과해 전체 FAIL이다. seed3의 T 중앙값 동률과 작은 R 악화를 허용 오차로 지우거나 좋은 seed만 선택할 근거가 없다. 새 실사 routing은 허용되지 않았다.

## 실제 실행된 가장 가까운 목적함수

| 선행 | 코드와 실제 학습 목표 | 완료된 결과와 해석 범위 |
|---|---|---|
| Stage2 W/D shared scorer | `pallet_selector_recovery_v1/models.py:14–32`, `train_scorer.py:10–16`. 두 후보 score 차이의 **비가중 BCE**, target은 exact W/D parity. Linear94 또는 MLP. | TRAIN4096/VAL1024/TEST1024. 합성 TEST 정확도 90.9180→93.3594%. 물리 pose의 T/R 비용 차이를 학습한 결과가 아니며, source 성공 자체는 존재한다. |
| Stage4 S0/S1 router | 같은 `models.fit(..., pairwise=False)`, `build_router_dataset.py:82–96`. 먼저 각 expert에서 D9 W/D를 고른 뒤, 두 전체 pose 중 작은 C2 ADDnorm expert를 **비가중 BCE**로 분류. | synthetic TEST mean ADDnorm S0 0.09621724, S1 0.09840939, routed 0.09768815, oracle 0.09031495. oracle 여유가 있어도 더 좋은 S0를 이기지 못했다. 실사 판정도 `CLEAN_RECOVERY_BUT_HARD_LOSS`. 현재 R0/PoseFix 네 후보 공동 순위와는 다르다. |
| Structured DHT v2 | `pallet_dht_structured_v2/data_ops.py:76–103`. 후보별 `log1p(mean 2D error/(0.01×image diagonal))`의 softmax를 target으로 하는 listwise CE + 후보 cost SmoothL1 + 코너 cost 회귀. 실제 temperature 0.1, regression weight 0.5, corner weight 0.25. | 3arm 각각 seed1·2000updates 완료. source pilot 통과 뒤 실사에서 full arm은 319장 전부 identity 유지. segment의 개선은 관측 가능한 309장 중 단 1장, point-only는 다른 1장을 악화. source calibration의 모든 0.5 미만 margin이 clean 보호 조건을 위반했다. 비용 감독 자체는 이미 실행됐고 일반적인 전이 해결책이 아니었다. |
| Point-line v4 | `pallet_point_line_v4_bundle/pointline_v4/objective.py:12–51`. 위 2D candidate cost의 soft-target CE + scalar SmoothL1 + corner 회귀를 계승. | 4arm×3seed×2000updates 실제 완료. 주 H의 source primary가 baseline 0.005755 대비 0.005789/0.005436/0.005436; seed1 악화로 `NO_SYNTHETIC_ADVANCEMENT_SIGNAL`. 실사와 후속 전체 네트워크 학습은 미실행. 소수 C4 번호 선택 오류의 회복·손상이 결과를 좌우했다. |
| Frozen Hough P/Q gain selector | `pallet_hough_gain_selector_v1/core.py:13–43`, `experiment.py:119–129,206–237`. **같은 전체 물체 GT 대응**에서 `P mean 2D error−Q mean 2D error`를 구해, 1000×diagonal-normalized signed gain을 SmoothL1로 회귀. 런타임은 전체 P 또는 전체 Q. | 3seed×2000updates. oracle primary gain 2.179%였지만 실제 전체 mean gain은 +0.016280/0/−0.019861px. `GAIN_SELECTOR_SYNTHETIC_FAIL`, 실사 미실행. 점 단위 혼합이 아니라 whole-layout **상대 비용 회귀도 이미 실패한 선행**이다. |
| N2/Replay utility selector | `pallet_posefix_utility_selector_v1/train.py:16–18`. `100×(N2 point error−Replay point error)/predicted box diagonal`의 masked SmoothL1. RGB 패치·기하·heatmap 요약 입력. 양 끝점의 예상 gain>0인 수직 모서리쌍을 수락. | 53,233 parameters, seed1·1500updates. 합성 clean PCK10 95.233→93.618%(9점 회복/51점 손상), GREEN150 81.351→78.267%. 실제 4.34→168.30px 오보정을 수락했다. 물리 whole-pose ranking은 아니지만 상대 gain 회귀의 clean 손상 반례다. |
| Student-relative geometry trust, Track D | `pallet_paper_contribution_screen_v1/track_d/mechanism.py:83–97,129–169`. 학생 점과 teacher line의 동일 normal 방향 오차 차이를 diagonal로 나누고 TRAIN RMS로 정규화한 **signed gain MSE**. GT 없는 불일치·불확실성 특징의 MLP64. | 3seed×1500updates. 15개 사전 calibration 조합 모두 coverage/harm 제약 실패. 세 seed 모두 abstention, TEST 선택 coverage0, `D_COMPLEMENTARITY_NOT_PREDICTABLE`. Student fit은 0이지만 **trust fit은 3개 완료**다. |

Structured-v2 근거는 `data/pallet/results/pallet_dht_structured_v2/{PROTOCOL.json,TRAINING_COMPLETION.json,PILOT_RESULTS.json}`와 `provenance/cost_gap_diagnosis_seed1/README.md`다. Point-line-v4의 정정된 `data/pallet/results/pallet_point_line_v4/REPORT_KO.md`를 사용했으며, 철회된 “무작위 사분면 선택” 해석은 사용하지 않았다. 나머지는 각 실험의 기존 한국어 보고서와 JSON에 근거한다.

## 치수 ranker와 보존 손실의 정확한 구분

`pallet_type_selftrain_v1/identity_rank.py:112–118`은 두 C2 대응 후보의 2D 오차 차이가 box diagonal의 1%보다 큰 source 행만 hard label로 학습한다. 원래 MLP는 CE, `identity_rank_linear.py:34–51,78–87`은 두 descriptor 차이에 대한 **비가중 BCE + L2**이며 full-batch L-BFGS strong-Wolfe, 최대200iteration을 사용했다. 규제값은 source TRAIN의 scenario 3-fold logloss로 골랐다. 비용 크기별 가중이나 expected regret가 아니다.

`identity_rank_dimensions.py:59–65,85–97`은 같은 loss에 알려진 치수 비율과 descriptor의 상호작용을 추가했다. 이 실행도 실제 완료됐고 실사 screen은 SYN/MIX 모두 실패했다. full194 PCK20은 R0 77.86%, SYN 75.35%, MIX 74.82%였으며, SYN은 hard8점 회복과 good15점 손상이 함께 있었다. 기존 선형 ranker도 R0 77.86% 대비 SYN76.40%/MIX75.35%였다. **치수·선형화·수렴 optimizer만으로 안전한 전이가 보장된다는 근거는 없다.** 이 후보들은 같은 예측 위치의 역할 번호를 바꾼 것이며 현재 물리 R0/PoseFix pose pool과 다르다.

보존 regret 역시 새 개념이 아니다. `pallet_direct_dimension_v1/model.py:45–58`의 `relu(new_error−R0_error)`와 `smooth_pilot.py:18–27`의 그 제곱, `pallet_direct_dimension_frozen_v1/model.py:45–51`의 `KD + .25GT + 2 squared regret + .1 easy consistency`는 실제 좌표 헤드 학습에 쓰였다. frozen dimension 후속은 각3seed·4000updates 뒤 대표 seed1의 주 지표 유지와 모든 지표 유지가 모두 False였다. 이는 **새 좌표를 만드는 손실**이며, 고정된 후보의 score만 학습하는 선택 손실과 같지는 않다.

`pallet_posefix_full_preserve_v1`은 source TRAIN normal에서 PRIOR1 오차≤5px인 코너를 teacher→student spatial KL로 보호했다. TRAIN gradient ratio0.25로 λ를 한 번 정하고 reset 뒤300updates를 실행했다. 결과는 `INCONCLUSIVE`: PRIMARY PCK10 순변화0, source clean PCK10은 BASE94.02%, FULL91.35%, FULL_PRESERVE92.24%로 source_clean gate 실패였다. 이 증거로 “보존 loss 추가가 미시도이며 clean을 해결한다”고 말할 수 없다.

반대로 `large_corner_selector.py`의 source-relative RandomForest 설계는 TRAIN coverage16, VAL coverage2로 최소3 기준에 미달해 **fit 전에 중단**됐다. 코드에 regression이 있다는 이유만으로 실패한 학습 실행에 포함하지 않는다. heatmap의 spatial soft target도 후보 간 비용 순위 목적함수와 분리했다.

## 현재 CE가 버리는 정보와, 그 사실만으로 할 수 없는 주장

[현재 train.py](../../../scripts/research/pallet_pose_union_selection_20261001_v1/train.py)의 `targets()`는 완전한 한 후보의 `c=max(T/sT,R/sR)`를 계산한 뒤 최소 index만 남긴다. exact cost tie에서는 Pareto·R0·가설 이름 순을 적용한다. `masked_loss()`는 `CE(-score, target)`로, 정답 index가 같으면 비용 차이가 작든 크든 같은 loss다. near tie의 expert 선택과 큰 W/D 실수의 비용 차이는 직접 반영하지 않는다. shared score의 공통 bias와 공통 특징 항은 후보 차이에서 상쇄되므로 유효 정보량도 95라는 명목 parameter 수와 동일하지 않다.

다른 담당자가 기존 checkpoint만 분석한 [TRAIN_DIAGNOSTIC.json](TRAIN_DIAGNOSTIC.json)에 따르면 UNION의 상위10% regret 행은 전체 regret의 약95.4–95.7%지만 CE의 약19.2–20.1%였다. 따라서 **margin-insensitive CE와 평균 scalar regret 사이의 불일치가 실제로 있다.** 한편 마지막5epoch online loss는 약0.40–0.47% 계속 감소했고, 최종 full-TRAIN gradient L2는 약0.053–0.057이었다. 이것은 충분한 수렴을 입증하지 않으며, 표현력 부족이나 optimizer 실패의 확정 진단도 아니다. 이 감사는 그 배열을 다시 채점하지 않았다.

Root의 [VAL oracle 진단](VAL_ORACLE.json)은 세 seed 모두에서 whole-pose oracle의 T/R 여유가 남는다고 보고했다. 동시에 learned seed3의 **평균 scalar regret가 가장 낮은데도 해당 seed만 중앙값 gate를 실패**했다. 따라서 expected scalar regret를 더 낮추면 두 population 중앙값이 반드시 좋아진다는 명제는 현재 관측과 맞지 않는다. 잘못된 W/D 비용이 약80인 쌍에 원시 cost-gap을 곱하면, 같은 분기의 작은 T/R 개선보다 큰 회전 오류가 학습을 지배할 수도 있다. 그 현상을 실제 학습으로 확인하기 전에 원인으로 확정하지 않는다.

조사 범위의 코드에서 `cost-sensitive`, `expected_regret`, `cost_gap_weight`, `pairwise_loss`, `ranking_loss`, `RankNet`, `LambdaRank` 및 관련 실제 loss를 검색하고 가까운 구현을 읽었다. 현재 frozen 물리 후보의 exact scalar cost gap을 가중치로 쓰는 순위 학습 실행은 찾지 못했다. **미발견은 곧 최우선 실행 근거가 아니다.** 단순 soft-label CE와 signed gain regression을 재명명해서 새 방법으로 제안하지 않는다.

## 다음 개입 한 가지에 대한 권고

**현재는 새로운 regret loss보다, whole-pose CE와 같은 Linear94를 사용하되 명시적 L2를 갖는 source-only 수렴 통제를 우선한다.** 후보·GT 비용·tie·source 정규화 통계·frame membership을 그대로 둔다. 30epoch 결과에서 학습이 아직 움직이는 만큼, regret 등 다른 통계적 목표까지 추가하지 않고 수렴한 규제 선형 모델을 먼저 확인하는 편이 다음 해석에 유리하다. 선행 identity-rank의 L-BFGS 사용을 발명으로 주장하지 않는다.

Root가 제시한 한 가지 후속안의 개념 검토는 다음과 같다. 이것은 **제안 검토**이며 새 fit 결과나 실행 영수증이 아니다.

| 항목 | 제안된 고정값 / 검토 |
|---|---|
| 학습 목적 | 전체 source TRAIN2598 frame 평균 masked CE + `(1e-4/2) × ||w||²`. exact whole-pose target 유지. 0/1-valid 후보 행 CE0 및 전체 frame 분모 유지. |
| 모델 | shared Linear94, 공통 bias는 모든 후보에서 상쇄되므로0 고정. 94개 weight 모두 명시적 L2 적용. |
| fit 수 | 동일 R0_ONLY를 한 번, DIVERSE251_s1/s2/s3와 각각 UNION을 한 번: **총4fit**. 모두 zero init, optimizer seed 탐색 없음. 세 반복은 이제 refiner seed 차이이며 selector 초기값 산포가 아니다. |
| solver / 예산 | CPU FP64 SciPy L-BFGS-B, bounds 없음, maxiter1000/maxfun2000, ftol1e-15/gtol1e-8. 사후 연장 없음. 실제 함수 호출 횟수도 기록하고, line search의 내부 평가 때문에 maxfun 표시를 넘길 수 있는 구현이면 strict cap을 별도로 보장해야 한다. |
| 수렴 인증 | solver success **그리고** full objective의 raw gradient에 대해 `||g||²/(2λ) ≤ 1e-6`. 실패하면 미수렴으로 중단하며 단순 success 표시로 대체하지 않는다. |
| 평가 | 모든4checkpoint를 먼저 동결한 뒤 기존 source VAL45개 조건·세 comparator를 그대로 적용. 모든 seed 통과 전 실사 routing0. 원래 실사 안정성 기준 유지. |

Bias를 제거하고 모든 weight에 λ>0 L2를 적용했으므로 이 목표는 λ-strongly convex다. `F(w)−F(w*) ≤ ||∇F(w)||²/(2λ)`는 그 full objective의 전역 최적값 차이에 대한 상계다. 계산은 frame 전체의 정확한 평균과 penalty를 포함해야 하며, stochastic batch gradient나 weight decay를 빠뜨린 CE-only gradient를 넣으면 이 인증이 아니다. 수치적으로 작은 gap을 달성해도 일반화나 population median 개선의 인증은 아니다.

λ의 숫자1e-4가 이전 weight_decay와 같아도 **AdamW의 decoupled decay와 explicit L2는 동일 objective가 아니다**. FP64와 optimizer도 바뀌므로 순수 optimizer 단독 인과 실험이라고 쓰지 않는다. 기존 source mean/std를 사용하되 FP32로 만들어졌던 정규화 배열을64bit로 올리는지, FP64에서 다시 계산하는지 실제 구현을 사전에 명시해야 한다. VAL을 보면서 tolerance·예산·규제값·정규화 경로를 고르지 않는다. 새 RGB forward·refiner fit은0이며, 본 감사는 이 실행의 사전 protocol을 대신하지 않는다.

기존6fit 및 source gate 실패를 보존하고, 이후 통제도 모든 세 seed와 원래 VAL 45개 조건으로 판정한다. train loss가 더 낮아져도 VAL의 공동 T/R·tail·failure 조건을 통과하지 못하면 real routing은 진행하지 않는다. 이미 공개된 VAL은 재사용 개발 평가이며 새로운 독립 검증으로 부르지 않는다. 이 통제와 cost-gap weighting/soft targets/새 특징을 동시에 실행하는 다중 탐색은 권고하지 않는다.

**현재 감사로 확인된 개선은 없다.** 후보의 GT 최선값, loss와 regret의 불일치, 수렴 여지는 각각 진단이며 단일 RGB+치수 입력의 최종 T/R 개선 실적과 구별한다.
