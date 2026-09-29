# 과거 시도와 표준 해법 감사: 최소 clean-to-pose 구성

## 결론과 감사 범위

새 학생을 먼저 학습할 이유는 없었다. R0·기존217 REF·clean78 RAW/REF CLEAR/OCC의 예측과 후보가 이미 있었고, 같은 frozen GEO를 붙이는 대조가 CPU 재사용으로 완성됐다. **현재 clean78 OCC는 가장 강한 단순 대조인 기존217 REF+GEO를 T/R 중앙값에서 함께 이기지 못한다.** 후속 1축 후보는 학생 재학습보다 **현재 모델 출력 분포에 맞춘 합성 전용 공통 GEO 1회 학습**이다. 이는 조건부 실험 권고이지 성공 판정이나 실행 기록이 아니다.

이 감사의 신규 학생/선택기 fit, optimizer update, GPU 사용은 모두 0이다. 같은 이름의 과거 실험을 같은 계약으로 간주하지 않았다. 결과를 보고 추가한 적응적 가설 선택이며 과거 사전등록으로 소급하지 않는다. 연결된 JSON은 읽은 저장소 입력의 SHA와 원문/공식 구현 접근 범위를 보존한다. 원장의 SHA는 **감사 시점 관측값**으로, 후속 실행으로 갱신될 수 있다.

## 1. 지금 가능한 강한 단순 대조

[CONTROL_RESULTS.json](CONTROL_RESULTS.json)의 자연 가림99장(Moderate21+Severe78) 전체 프레임 집계다. 아래 행의 valid는 모두99/99다. 표는 중앙값 차이를 설명하며 프레임별 변화의 중앙값이나 독립 반복의 신뢰구간이 아니다. Clean29·전체128·촬영 기록·paired 결과는 원본 결과 및 후속 robustness 분석을 함께 봐야 한다.

| 고정 구성 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° |
|---|---:|---:|---:|---:|
| R0_D9 | 12.14792 | 6.03794 | 120.47137 | 89.21826 |
| R0_GEO | 12.40325 | 5.21758 | 120.47137 | 88.92186 |
| OLD_REF_GEO | 11.41011 | 4.95526 | 127.14307 | 88.82035 |
| RAW_CLEAR_S42_GEO | 11.73011 | 5.19628 | 119.11880 | 89.47477 |
| REF_CLEAR_S42_GEO | 11.49986 | 5.20260 | 128.30350 | 89.06627 |
| RAW_OCC_S42_GEO | 11.81832 | 5.37691 | 119.44548 | 89.50165 |
| REF_OCC_S42_GEO | 11.36961 | 5.26350 | 128.64498 | 89.01503 |
| RAW_OCC_S43_GEO | 11.98750 | 5.21813 | 119.07545 | 89.31480 |
| REF_OCC_S43_GEO | 11.26432 | 5.07466 | 128.38763 | 89.03088 |

핵심 해석은 다음과 같다.

- R0+D9→R0+GEO는 R 감소/T 증가다. 선택기만 붙이면 모든 목적이 좋아지는 것은 아니다.
- 기존217 REF+GEO는 R0+GEO보다 T/R 중앙값이 모두 낮지만 T P90은 높다. clean78의 필요성을 주장하려면 이 대조를 반드시 포함해야 한다.
- REF OCC42/43−기존217 REF+GEO는 각각 T −0.04050/−0.14579cm(−0.405/−1.458mm), R +0.30824/+0.11940°다. joint improvement가 아니다.
- RAW CLEAR42+GEO는 RAW OCC42+GEO보다 T/R 중앙값이 모두 낮다. REF CLEAR42→REF OCC42는 T 감소/R 증가다. 입력 가림의 필수성은 확인되지 않았다.
- REF CLEAR42−RAW CLEAR42는 T 감소/R +0.00632°이며, REF OCC−RAW OCC는 두 seed에서 두 중앙값 감소다. 보정의 추가 가치는 입력 조건에 따라 다르고 REF의 T tail 손상도 남는다.
- seed43에는 OCC RAW/REF만 존재한다. CLEAR43을 있는 것처럼 채우거나, OCC 반복으로 CLEAR/OCC 효과의 재현을 주장하면 안 된다.
- 기존217→clean78은 membership·고유 장수·반복 노출·post-affine 공통 support 계약도 바뀐다. 순수한 clean 여부의 효과가 아니다.

## 2. 실제 과거 결과: 재실행 금지와 남은 질문

상세 행별 artifact/code 근거는 [기존 시도 지도](../pallet_oracle_mechanism_followup_v1/PRIOR_ATTEMPTS.md)와 [재사용 판정 JSON](../pallet_oracle_mechanism_followup_v1/REUSE_AND_RETRY_DECISIONS.json)을 이용하되, 그 뒤 실행된 C2/C3·objective v2·clean78 결과로 상태를 갱신했다. 특히 과거 지도에서 “affine-off 미실행”이던 상태는 지금은 사실이 아니다.

| 축·시도 | 실제로 바뀐 계약과 관측 | 이번 판정 |
|---|---|---|
| 320→640 update | 동일 pose-only 레시피의 추가 최적화. verified66에서 RAW/REF 둘 다43→44여서 REF 우위가 생기지 않았고 기존 REF 전체 PCK10은507→504/985. | 유효한 음성. 같은 반복을 새 가설로 쓰지 않는다. 모든 최적화 범위를 배제하지는 않는다. |
| source-off | source512+real512를 real 반복1024로 바꾸어 실사 노출도2배. 기존 DEV194 결과는 혼합. | source 간섭만 분리한 실험이 아니다. 같은 재학습보다 cached score 재사용 우선. |
| 실제 파라미터 방향 진단 | 실제 true-ignore·공통 normalizer·RLE gate를 유지한0-step 분해. REF 실사 코너119의 TOWARD87→combined87, lost16/gained16; 네 전역 cosine 모두 양수. | 지배적 source opposition, source 제거, PCGrad 필요성의 근거가 아니다. 국소 간섭과 실제 AdamW 장기 효과는 별개다. |
| AGREE vs COUNT | 동일 프레임/점 수로 교사 일치 기반 subset 대조. AGREE PCK20이 baseline/COUNT보다 낮고 큰 보정6개를 모두 버렸다. | 일치도=정확도라는 현재 cue는 지지되지 않음. threshold sweep 금지. |
| multi-teacher | point oracle는 좋아졌지만 median/medoid 실제 융합은 P90 손상. uncertainty 적합성 문제로 일부 student/adapter는 미실행. | 융합 음성과 미실행을 구별. 낮은 분산·도메인 구분력이 정확성 보증은 아니다. |
| N2/N3·PoseFix Replay | N3의 N2 대비 작은/없는 이득; PoseFix Replay9/38의 TRAIN 과적합은 DEV/clean 보존으로 이어지지 않음. | 새 이름으로 같은 교사·refiner 재훈련 금지. Clean19 교사와 current9/38을 혼동하지 않는다. |
| zero-init adapter·hard8 | zero-init S1 잔차 모듈은 severe/AUC 손상. hard8 direct36은 TRAIN36/36 및 일부2D/oracle 이득이나 oldGEO 최종 AUC 하락. | 모든 adapter·수동 신호가 무효라는 결론은 아님. 이번 새 수동 좌표 추가는 승인 범위 밖. |
| Hough/DHT | 합성의 약한/불안정 이득 또는 line이 point보다 낮음; 일부 real/student 단계 미실행. | 현재 팔레트 T/R 음성으로 잘못 표기하지 않되, 대형 새 표현 학습으로 확대할 근거도 부족. |
| C1 Huber D9 | 고정 예측에서 robust 후보 비용 변경. 두 재료/모든 팔의 선택 변화0. | 유효한 zero-fit 음성. Huber 계수 소탐색 금지. |
| C2 real affine-off | source/HSV 유지, 실사 translate/scale만0. Plastic PCK10 507→503/985, AUC 감소. 단 Full128 T/R 중앙값은9.186/3.766→8.988/3.661로 작게 개선. | 모든6D 지표가 실패했다고 쓰지 않는다. 이미 실행된 축이며 새 fit보다 cached99+sameGEO 재평가가 가능한 후순위 대조다. 큰 TRAIN 평균 오차의 주범은 한 frame의 최고점수 검출 instance mismatch였다. |
| C3 direct manual38 | 동일9장/38점 RAW9 vs MANUAL9,320 update. TRAIN PCK10 32→35/38이나 한 Wood frame3점 >20px 잔존. Plastic +5/985, Wood −22/346, verified66 −3. | 배정된 anchor에서의 부분 적합만 확인. 물리 좌표 identity 정답성과 일반화는 미해결. |
| objective v2 B | 실제 RLE clamp/좌표 gradient 감사 후 normalized SmoothL1 β1, λ0.1650038776 추가. 기존 REF 대비 T/R 둘 다 소폭 악화. | “포화에 L1만 더하면 해결”은 해당 설정에서 음성. clamp 존재만으로 전체 실패 원인 확정 금지. |
| objective v2 A/C | 입력 가림/노출확률 증가의 조건부 손익. C nominal seed43은 loader bug로 실제 trace/checkpoint 동일. | 당시 반복은 무효, 학습 비용은 보존. 이후 clean78의 실제 seed42/43 OCC 반복과 구별. |
| clean78 2×2+OCC 반복 | support intersection·실제 seed 흐름을 검증한6 fits. 기본 D9 공동 이득 없음. sameGEO 보정 OCC의 paired signal은 있으나 강한217 REF+GEO와 trade-off. | 새로 완성한 단순 대조를 숨기지 않고 불필요한 가림을 제거할 수 있어야 한다. |
| 모델별 GEO 재적합 | H_MANUAL에서 old→own GEO AUC .358570→.373285, 당시 S1+old .365035 상회. 하지만 합성 TEST 정답936/1024<old952이고 old pooled2모델/own1모델로 학습행 수도 달랐음. | 현재 분포 재적합의 직접적인 가능성 근거. 현재99 T/R 개선 증명은 아니며, 동일 정보 공통 scorer와 학습행 차이를 이번에 통제해야 한다. |

관련 최신 원자료: [gradient 방향](../pallet_gradient_transfer_diagnostic_v1/REPORT_KO.md), [C2](../pallet_oracle_mechanism_followup_v1/cycles/C2_REAL_AFFINE_OFF/REPORT_KO.md), [C3](../pallet_oracle_mechanism_followup_v1/cycles/C3_MANUAL38_CAPABILITY/REPORT_KO.md), [objective v2](../pallet_pose_objective_followup_v2/REPORT_KO.md), [loss 감사](../pallet_pose_objective_followup_v2/LOSS_SIGNAL_AUDIT.md), [모델별 selector](../pallet_model_conditioned_selector_v1/REPORT_KO.md).

## 3. 표준 원문과 공식 구현: 무엇을 가져올 수 있는가

2026-09-29 검색·접근. 아래 논문은 방법 원리 근거이며 현재 결과를 보증하지 않는다. 공식 repository/README를 확인했지만 전체 외부 구현을 설치·재현하거나 현재 Ultralytics와 동일하다고 인증하지 않았다. RLE/EPro-PnP의 CVF 직접 open 일부는 fetch 실패하여 접근 가능한 원문/저자 repository를 함께 사용했다.

| 방법 | 원문·공식 구현 | 현재 문제와의 연결 및 경계 |
|---|---|---|
| RankNet, ICML2005 | [저자 원문](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) | 쌍의 점수 차이로 선호 확률을 학습하는 원리. 현재 shared Linear94의 BCE(score0−score1)와 연결된다. 원래 RankNet 공식 실행 코드는 이번 감사에서 확인하지 않았고, 제3자 구현을 공식이라 하지 않는다. 실제 재사용 구현은 저장소의 검증된 models.py다. |
| DSAC, CVPR2017 | [원문](https://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html), [공식 코드](https://github.com/cvlab-dresden/DSAC) | pose hypothesis의 생성·평가·선택을 분리하고 downstream pose 목적을 고려하는 선례. DSAC는 확률적 선택/기대 loss를 미분하는 구조라 현 deterministic Linear94를 DSAC 구현이라고 부를 수 없다. 이번에는 새 solver나 end-to-end 학습을 도입하지 않는다. |
| RLE, ICCV2021 | [원문](https://arxiv.org/abs/2107.11291), [공식 코드](https://github.com/jeffffffli/res-loglikelihood-regression) | residual likelihood와 flow 기반 분포 추정. 현재 loss에 이미 해당 계열이 있어 “RLE 새 도입”은 새 가설이 아니다. 논문 원리로 현재 aggregate clamp의 제거나 sigma의 실제 오차 보정을 정당화할 수 없다. |
| PoseFix, CVPR2019 | [원문](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html), [공식 코드](https://github.com/mks0601/PoseFix_RELEASE) | RGB+초기 pose, 경험적 오류를 합성해 refiner 학습. 현재 Replay에서 이미 대응 시도. 사람 관절의 오류분포가 팔레트 role/자기가림 오류와 같다고 가정할 수 없다. |
| Mean Teacher, NeurIPS2017 | [원문](https://arxiv.org/abs/1703.01780), [공식 코드](https://github.com/CuriousAI/mean-teacher) | EMA 가중치 교사와 consistency 학습. 고정 보정 교사를 online teacher로 바꾸면 타깃 생성·학습 동역학이 달라지는 별도 개입이다. 일치도가 좌표 정답이라는 보장은 없다. |
| Random Erasing, AAAI2020 | [원문](https://ojs.aaai.org/index.php/AAAI/article/view/7000), [공식 코드](https://github.com/zhunzhong07/Random-Erasing) | 무작위 사각형 입력 지움의 원리. 이번 정책은 covered≥1/remain≥2와 canonical REF 계획을 추가한 팔레트 변형이다. 분류/검출에서의 효과가 amodal 팔레트 좌표 감독의 정확성이나 OCC 필수성을 증명하지 않는다. |
| BPnP, CVPR2020 | [원문](https://arxiv.org/abs/1909.06043), [공식 코드](https://github.com/BoChenYS/BPnP) | PnP를 통한 미분으로 geometric 목적을 연결. 실제 대응점/카메라/학습 목표 계약이 필요하다. DEV6D를 fit target으로 쓰거나 W/D discrete 선택을 무시한 단순 loss drop-in은 허용되지 않는다. |
| EPro-PnP, CVPR2022 | [공식 원문 연결 repository](https://github.com/tjiiv-cprg/EPro-PnP), [논문 페이지](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html) | 확률적인 pose 분포/대응점 가중치 학습. 현 sparse 직접 회귀/고정 후보선택과 목적·출력 계약이 달라진다. 후보 선택 1회라는 작은 축과 동시에 바꾸지 않는다. |

학습 범위·optimizer는 원문 유행보다 실제 코드가 우선이다. 현재 pose-only trainer는 pose/flow 관련132 tensor만 학습하고 보호747개를 정확히 유지한다. neck을 풀면 detector head 가중치가 고정이어도 공유 feature 때문에 box/score가 변할 수 있다. 과거 full-model 음성은 다른 LR·데이터·평가였으므로 모든 부분 unfreeze를 반박하지 않지만, 지금은 frozen detector/assignment 계약을 바꾸는 추가 복잡성이 있다. 현재0-step 방향 진단은 실제 AdamW update나 장기 성능 예측이 아니며 optimizer 변경 근거로 과장할 수 없다.

## 4. 권고하는 한 번의 다음 확인: current-domain common GEO

가설: old GEO가 학습한 S0/S1의 합성 candidate-feature 분포와 현재 모델들의 분포가 달라, 기존 후보의 이득을 선택 단계에서 일부 잃는다. **분포 차이 자체는 원인 증명이 아니므로**, frozen candidate의 전체-pose oracle에서 선택 가능 여지가 있는지 먼저 확인하고 그 이력이 새 명세에 연결돼야 한다.

한 번만 학습하는 공통 selector는 승인 범위 안이며 새 학생0회/새 수동좌표0개다. 기존 잠긴 합성 TRAIN4096/VAL1024/TEST1024의 RGB/K/치수/GT·split을 그대로 재사용한다. 이미 조회된 synthetic TEST는 새로운 봉인 TEST가 아니다. 새 실사 membership/타깃이나 평가 참조를 학습에 넣지 않는다.

### 실행 전 반드시 고정할 계약

1. **현재 feature donor 모델**과 checkpoint SHA를 결과 조회 전에 고정한다. 가장 좁은 후보는 RAW_CLEAR42+REF_CLEAR42의 두 모델 pool이다. old S0/S1도 두 모델 pool이므로 전체6모델 pool보다 학습행/epoch-update 차이를 줄인다. 이는 권고이며 실제 donor 선택은 별도 protocol이 결정한다. 모든 valid 수·누락·학습 노출 수 차이를 공개한다.
2. 두 모델의 frozen 합성 inference→94 feature와 실제 final 후보 freeze를 완료한다. 기존 train_scorer.py의 타깃은 y=np.tile(labels['parity'],2)인 **renderer의 물리적 W/D parity**다. 같은 합성 image의 label은 donor 모델이 바뀌어도 동일하게 유지한다. 현재 후보별 T/R로 타깃을 새로 만드는 것이 아니다. feature·valid pair·선택 결과의 정확도는 바뀔 수 있고, 선택된 물리 parity가 맞아도 연속 T/R 오차는 클 수 있다.
3. 기존 shared Linear(94,1), TRAIN 양 후보 공유 mean/std floor1e−6, BCEWithLogits(score0−score1), AdamW lr1e−3/wd1e−4, batch256, seed42, max30/patience5, earliest-best synthetic VAL을 재사용한다. invalid pair 처리, GT tie 처리와 양 후보 순서/이름 tie-break도 고정한다. 새 feature/threshold/MLP/loss sweep은 없다.
4. **하나의 동일 checkpoint와 동일 normalization**을 R0, OLD_REF217, RAW/REF CLEAR/OCC42 및 실제 존재하는 OCC43 모두에 적용한다. RAW/REF마다 따로 fit하지 않는다. 점수가 낮은 whole-pose 후보 하나만 선택하고 invalid pair는 기존 D9 fallback이다.
5. candidate renderer/solver를 바꾸지 않는다. 현 selector_compat.py처럼 feature extractor의 중간 production_pose가 아니라 이미 잠긴 **최종 D9 후보 pose**를 선택한다. 후보명/순서 교환 불변성, feature 치수, source hash, 모든 frame/실패 분모를 검사한다.
6. 합성 학습·checkpoint 선택에는 실사 참조/오차를 읽지 않는다. 실사 선택을 먼저 freeze한 뒤 별도 score에서 참조를 결합한다. “실사 결과를 보고 가설을 정함”과 “실사 GT를 fit/threshold에 사용함”은 구별한다.
7. old GEO와 새 GEO를 모든 동일 arm에 비교한다. 특히 R0+새 GEO와 OLD_REF+새 GEO보다 못한 학생을 최종 개선으로 승격하지 않는다. T/R 중앙값·P90·valid·paired·촬영기록·난도와 기존 seed 반복 범위로 판단하며 AUC/parity accuracy로 T/R 손상을 대체하지 않는다.
8. 한 번의 fit 뒤 결과가 나쁘다고 donor·특징·epoch 기준을 바꿔 재시도하지 않는다. 새 selector1회 한도에 계산하고 추론 비용과 학습 비용은 분리 기록한다.

### 강한 반론과 판정 한계

- 잘못된 2D/깊이·detector mismatch 때문에 두 후보가 모두 나쁘면 선택기만으로 회복할 수 없다. T-optimal 후보와 R-optimal 후보를 합성한 가상 pose는 해법이 아니다.
- old GEO의 간접 감독/학습분포 편향도 있지만 현재 pseudo targets도 안정적으로 틀릴 수 있다. 분포 적합은 물리적 정확성 보증이 아니다.
- 현재 사례는 반복 사용 DEV, 참조는 기하 재구성이다. 새 prediction lock과 old2seed가 독립 실측 정답이나 독립 평가를 만들지 않는다.
- 기존 H_MANUAL의 own GEO 성공은 ADDsym AUC 기준이며 합성 TEST는 오히려 낮았다. 이는 새 calibration 성공의 보증이 아니라 실제 배포 분포 호환성이 미해결이라는 근거다.
- 합성 parity classification 목적을 그대로 유지하면 실제 T/R 최적화와 완전히 일치하지 않는다. 이번에는 목적까지 동시에 변경하지 말고 그 한계를 보고한다.
- 새 scorer를 current self-trained 학생의 출력으로 학습한 뒤 R0에 붙이면 **배포 키포인트 모델에서 학생을 제거**할 수는 있지만 **전체 학습 계보에서 자기학습을 제거**했다고 말할 수 없다.
- old GEO를 재사용하는 파이프라인의 수동 감독 합집합은19장/86코너다. 새 scorer가 old checkpoint/정규화/feature 출력 중 무엇을 실제로 이어받는지에 따라 ancestry를 다시 추적해야 한다. 같은 feature 정의 코드를 재사용했다는 사실만으로19/86을 자동 상속하거나 자동 제거하지 않는다.

## 5. 후순위·배제·미검증을 구분

CLEAR 반복은 최종 구성 기여 판정에 필요한 경우 RAW/REF43 두 fit으로 보완할 수 있지만 새 방법의 발견이 아니다. 현재 selector 개입이 더 싼 구분을 제공하므로 그 전에 잔여4학생을 소모하지 않는 편이 낫다. 기존 C2 affine-off 등의 saved 예측 재평가는 신규 fit가 아니며 필요하면 별도 adaptive analysis로 연결할 수 있다.

독립 TRAIN 신뢰성 cue 없는 agreement/variance gating, 같은640update, 같은SmoothL1 λ 탐색, 가림확률 미세 sweep, Huber 비용 sweep은 현재 근거상 우선순위에서 제외한다. 추가 인간 레이블·교사 감독·새 데이터 풀은 별도 승인 대상이다. BPnP/EPro-PnP·dense voting head·공유 feature unfreeze는 전제가 달라지는 미검증 가설이지 검증된 실패도, 이번 작은 selector 변경과 함께 실행할 축도 아니다.

## 6. 공개·사례·감독 경계

clean78은 assistant RGB-only 외부 가림/마커판 기준 감사이며 모든 코너 수동 정답78장이 아니다. 제외137장을 모두 자연적인 심한 가림으로 표현하지 않는다. 신규manual0과 전체감독9/38은 동의어가 아니며 [감독 계보](../pallet_clean_to_pose_transfer_v1/SELECTOR_SUPERVISION_PROVENANCE.json)의 current9/38+oldPlastic10/48=19/86을 구분한다.

[이번 사례 선택 규칙](ANALYSIS_LOCK.md)을 따른다. 같은99 중 공동 개선/공동 악화를 T 변화로 정렬하고, 최종 T/R 최대 사례와 모든 실패도 남긴다. public에는 ID/오차/선정기준/숫자 그림, unrestricted topcase RGB·좌표 overlay는 private가 기본이다. 새 원본 RGB·비공개 좌표·checkpoint 공개는 금지한다.

이미 공개된 RGB만 재사용할 때에는 [기존 승인 manifest](../pallet_selftraining_paper_closure_v1/FIGURE_MANIFEST.json) 및 [보존 manifest](../pallet_pose_objective_followup_v2/RETAINED_FIGURE_MANIFEST.json)의 정확한 ID·image SHA·범위를 연결한다. 승인-ID subset의 예시는 전체99의 최악 사례를 대표한다고 주장하지 않는다. 공개 가능한 해당 범주가 없으면 NA 사유를 남긴다.

기존 report.py의 overlay(ax, im, pred, gt, title, color)는 **native 2D 점**용이다. green은 legacy2D 참조, 색 점은 native prediction이고 GEO 선택의 6D 재투영이 아니다. GEO 효과를 그림으로 보일 때 native 점은 같을 수 있다. 별도 selected whole-pose 재투영을 쓰면 구분된 색·범례·해당 pose T/R·기존 image bound clipping을 명시해야 한다. v2의 사례 정렬 기준을 이번 기준으로 조용히 대체하지 않는다.
