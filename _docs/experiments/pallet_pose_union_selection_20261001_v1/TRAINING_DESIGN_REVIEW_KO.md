# R0·DIVERSE 전체 pose 후보 통합: 학습 설계 독립 검토

2026-10-01. 기존 source/selector 코드와 과거 router 결과를 읽은 설계 검토다. 신규 feature 추출·fit·실사 평가·GT 기반 routing을 실행하지 않았다. 아래는 source 메타데이터 검증을 반영한 최종 학습 설계다. TRAIN feasibility PASS와 별도 TRAIN_PROTOCOL 봉인 전에는 fit하지 않는다. 실행 완료 또는 개선 결과를 뜻하지 않는다.

**조건부로 타당한 최소 비교다. 다만 후보 합집합에 좋은 pose가 있다는 사실만으로 Linear94가 그것을 식별할 수 있다고 가정하면 안 된다.** source에서 식별·전이 신호가 없으면6개의 작은 fit 뒤 실사 결과를 구제하는 feature/threshold/seed 탐색 없이 중단해야 한다. 이번 개입은 기존 pose 중 하나를 선택하는 새 방법이며, DIVERSE 좌표 자체의 오류가 줄었다고 보고할 수 없다.

## 무엇을 학습할 수 있고 무엇이 없는가

`scripts/research/pallet_selector_recovery_v1/features.py`의94개 특징은 pose/형상/재투영 잔차·bbox·confidence다. RGB embedding, patch의 시각적 증거, 표면 edge와의 정합은 없다. 특징은 detector/refiner의 이미지 조건부 출력에서 유도됐지만 RGB 자체를 보는 ranker는 아니다.

현재 DIVERSE는 R0의 선택 bbox, score, keypoint confidence와 center8을 보존한다. 따라서 bbox/confidence 직접 특징은4후보에서 공통이며 shared linear score의 후보 차이에서 상쇄된다. center residual은 후보 pose가 다르므로 공통값이 아니다. bbox 정규화·confidence 가중 residual의 간접 영향도 남는다. 이런 pose/잔차 차이로 source에서 품질을 예측할 수는 있지만, **잘못된 pose에 기하적으로 일관된 점**을 올바른 pose와 구별할 독립 시각 정보는 없다.

기존 `pallet_selector_recovery_v1/stage4_clean_preservation/STAGE4_REPORT_KO.md`는 더 풍부한 cross-expert 특징과 frozen S1 GAP448을 쓴 MLP64/32 router였다. synthetic TEST mean ADDnorm은 S0 0.096217, S1 0.098409, routed 0.097688, oracle 0.090315였다. oracle 여유가 있어도 best fixed expert를 이기지 못했다. 실사에서는 clean AUC +0.004207을 얻으면서 Severe −0.003218을 남겼다. 새 union은 expert별 기존 W/D 선택까지 동시에 바꿀 수 있고 T/R target을 사용한다는 차이가 있지만, “router를 아직 안 해 봤다”는 주장은 틀리다.

## 비교군과 후보 계약

각 s∈{1,2,3}에 대해 다음 한 쌍만 만든다.

| arm | 후보 | 후보 generator / optimizer seed |
|---|---|---|
| R0_ONLY_s | R0 long, R0 short | 동결 R0 / s |
| UNION_s | R0 long, R0 short, DIVERSE251_s long, DIVERSE251_s short | 같은 동결 R0와 기존 해당 DIVERSE checkpoint / s |

두 arm은 동일 source frame·split·94개 feature 함수·정규화·초기 파라미터·shuffle·optimizer·update 수·선택 규칙을 공유한다. UNION의 후보 두 개 추가가 비교 대상이다. DIVERSE3seed를 한꺼번에 합쳐8후보로 만들거나 사후 best seed를 고르지 않는다. 각 쌍에서 refiner seed와 optimizer seed가 함께 변하므로, 결과를 두 변동 원인의 분리된 요인 실험으로 해석하지 않는다.

95-parameter `Linear(94,1)` 하나를 모든 후보에 공유하고 score가 작은 pose를 고른다. expert ID, recording/severity, 평가 성적 또는 refiner seed ID를 feature로 추가하지 않는다. bias는 후보 차이에서 상쇄돼 순위 학습에 실질적으로 기여하지 않는다. 선택은 전체 pose (R,t)와 그 생성 expert/가설을 함께 보존한다. 서로 다른 후보의 R와 t, 코너를 혼합하지 않는다.

기존 `models.fit()`는2후보 logit와 parity target, `manual_seed(42)`가 하드코딩되어 있다. 이 함수를4후보에 그대로 호출하면 union 목적함수를 구현한 것이 아니다. 새로운 명시적 seed·가변 후보 mask·listwise loss가 필요하며 기존 파일은 수정하지 않는다.

## 권고 목적함수 하나

이전 long/short parity label은 같은 W/D의 R0와 DIVERSE 중 어떤 pose가 더 정확한지 말해주지 않는다. source exact renderer pose로 각 완전한 후보의 **T와 물리 대칭을 적용한 R**를 따로 계산해야 한다.

공통 scale은 frozen R0+기존 GEO의 적격 source TRAIN 오류에서 한 번만 계산한다.

```text
sT = median(TRAIN R0_operational translation_cm)
sR = median(TRAIN R0_operational rotation_deg)
cost(i,c) = max(T(i,c)/sT, R(i,c)/sR)
regret(i,c) = cost(i,c) - min_valid_candidate cost(i,·)
target(i) = one whole candidate with minimum cost
loss(i) = cross_entropy(-shared_score(valid candidates), target(i))
```

sT/sR는6fits에 공통이고 finite·양수여야 한다. 0이나 비정상 값이면 임의 floor를 찾아 구제하지 않고 준비 실패로 남긴다. 단위 변환은 scale에도 동일하게 적용하므로 cm/deg 선택이 결과를 바꾸지 않는다. 이는 균형을 정의하는 **source 기반 설계 선택**이며 실제 joint 개선의 수학적 보장이 아니다. R0_ONLY와 UNION에 서로 다른 scale을 사용하면 비교가 흐려진다.

최소 cost가 정확히 동률이면 그 집합 안의 Pareto-dominated 후보를 먼저 제거하고, 남은 후보 중 R0 우선·가설 이름 사전순으로 결정한다. 최종 inference score 동률에도 동일한 고정 expert/가설 순서를 적용한다. 근접 동률을 평가 성적에 맞춰 넓히는 epsilon sweep은 하지 않는다. failure 후보는 label/softmax에서 제외하되 frame 전체를 성능 분모에서 삭제하지 않는다.

**Pareto-only 쌍 loss를 주 objective로 권하지 않는다.** T 개선/R 손상처럼 서로 지배하지 않는 쌍에는 label이 없어, 실제 expert 선택 문제를 학습하지 못할 수 있다. 대신 TRAIN/VAL에서 cross-expert Pareto 개선·손상·tradeoff의 수를 반드시 보고한다. 위 scalar cost 역시 한 축 손상을 허용할 수 있으므로 아래 축별 gate가 필수다. 최저 T와 최저 R를 별도 후보에서 골라 하나의 target으로 만들지 않는다.

## source 학습 전 go/no-go

실사 GT를 읽지 않는 source 준비 단계에서 다음을 모두 확인한다.

1. 기존 renderer-group-disjoint 입력 pool TRAIN4096/VAL1024의 모든 이미지와 후보는 보존한다. 실제 ranker 적격성은 사전에 선언된 C2이며 canonical cuboid→Xcf가 proper rigid transform인 행으로 한정한다. 전체5120의 메타데이터 검증 뒤 최종 적격 TRAIN2598/VAL1024가 확정됐다. C1은 camera-facing와 물리 방향의180° 모호성을 기존 label로 해소하지 못해 분리 진단하고, C2 TRAIN 한 행 G38__G__f26406은 det−1 reflection이라 SO(3) pose를 만들 수 없는 label-schema 결함으로 적격 학습에서 제외한다. 이는 예측 오차 기준 필터가 아니며 원래 FAIL, 제외 사유와 행은 보존한다. 세부 계약은 `SOURCE_CONTRACT.json`과 metadata amendment에 있다. TEST1024는 이번 최저 비용 설계에서 열거나 선택에 쓰지 않는다. 이전 base/refiner가 일부 source를 이미 학습했다는 노출 이력을 공개한다. 이전 전체 split 감사에서 refiner source order와 TRAIN/VAL/TEST의 교집합은105/26/26장이었으므로(새 적격 부분집합의 재계수로 오해하지 않음) VAL을 모든 구성요소에 미노출인 TEST로 부르지 않는다.
2. 모든 frame에 대해 후보/feature를 먼저 잠근 뒤 source renderer label을 연결한다. prediction bbox를 GT bbox로 대체하지 않는다. source crop/pad/K, physical frame 변환, C2 또는 실제 객체 대칭 계약을 검증한다. synthetic perfect-coordinate fixture에서 target T/R가0에 가까워야 한다. 기존 parity나 ADD audit만으로 새 R label의 좌표계가 검증됐다고 가정하지 않는다.
3. 세 DIVERSE generator 모두에서 원 R0_ONLY보다 후보 수가 늘고, source TRAIN에 양 방향의 cross-expert Pareto 우세 사례가 존재하는지 확인한다. 한 expert가 언제나 선택되는 target이라면 learned routing이 필요한 증거가 약하므로 fixed-expert 결과부터 보고한다. 같은 feature에 상충 target이 붙는 중복·near-duplicate, 후보별 유효율과 collapse를 기록한다.
4. 위 cost로 선택한 **동일 whole-pose oracle**이 source TRAIN에서 R0_ONLY cost-oracle 대비 T/R 중앙값을 모두 낮추고, 각 P90이1.05배 안에 드는지 세 seed 모두 확인한다. 원 R0/DIVERSE operational baseline도 함께 비교한다. TRAIN에서 label 자체가 joint 방향을 보여주지 못하면 이 objective의6fit을 진행하지 않는다. 다른 cost scale을 사후 탐색하지 않는다.

위 gate는 추가 학습의 최소 의미를 확인하는 조건이지 linear 함수로의 식별 가능성을 증명하지 않는다. 명목 oracle gap만 크고 feature가 품질을 식별하지 못하는 문제는 VAL에서 걸러야 한다.

## 제한된 학습과 VAL 중단 조건

권고 고정값은 기존 small-scorer 범위를 따른다: AdamW lr0.001, weight_decay0.0001, betas(0.9,0.999), eps1e−8, source frame batch256, **30epochs의 final checkpoint만** 사용한다.최종2598 TRAIN이면 epoch당11 batch, arm당330 update, 총 최대6fits/1,980 update다. 마지막 batch도38행을 보존한다. 초기 제안의4096 TRAIN·480 update/fit·총2,880 update는 C1/invalid-schema 발견 전의 계산으로 남기는 이력이며 현재 실행 예산이 아니다. 학습 단계수에 wall time 추정을 붙이지 않는다. patience 또는 best-VAL epoch 선택을 쓰지 않아 paired 두 arm의 업데이트 수를 유지한다.

공통 mean/std는 적격 R0 source TRAIN의 유효한2후보 특징에서만 계산하고 기존 std floor1e−6을 그대로 사용한다. DIVERSE 특징도 같은 변환을 적용한다. transformed feature의 finite/범위를 준비 단계에서 보고한다. 씨앗 s로 한 번 만든 초기 Linear 상태와 frame 순서를 해당 두 arm에 복사한다. bias·고정 feature가 동일함, 실제 parameter 변화, seed별 stream 차이를 기록한다.

후보 결손이 있는 source frame에서 frame ID를 arm별로 다시 sampling하지 않는다. 모든 frame을 같은 순서로 보내고, 적격 후보2개 이상이면 CE를 계산한다.1개면 선택만 가능하고 loss0,0개면 target−1/loss0다. inference에서0개인 경우는 두 arm 모두 같은 저장된 R0+GEO whole pose가 available이면 사용하고, 그것도 없으면 실패와 양축+∞로 보존한다. frame 평균의 분모를 동일하게 유지하고 각 arm의 유효 ranking supervision 수 차이는 공개한다. invalid feature를0으로 채워 정상 후보처럼 점수를 주지 않는다.

최종6모델이 저장된 뒤 VAL을 한 번 채점한다. **세 UNION seed 모두**에서 matched R0_ONLY ranker, 기존 frozen R0+GEO, 해당 DIVERSE+GEO 대비 다음을 만족해야 실사173장으로 진행한다.

- T 중앙값과 R 중앙값이 각각 엄격히 작다.
- T/R P90은 각각 baseline의1.05배 이하다.
- frame 실패 수가 늘지 않는다. conditional과 실패 포함 full-population을 둘 다 제공한다.
- source가 clean/applied-occlusion 쌍을 이미 포함하는 설계라면 양 조건에 보존 gate를 별도 적용한다. 이번 최소안이 clean source만 사용한다면 occluded source 전이를 검증했다고 주장하지 않는다.

VAL 기준 하나라도 실패하면 해당 seed만 버리거나 fallback model을 골라 계속하지 않고,3seed 비교 전체를 `SOURCE_RANKING_NO_JOINT_SIGNAL`로 끝낸다. VAL 통과도 실제 자연 가림 전이의 증거가 아니다. 과거 VAL/TEST가 각각 renderer group1개였다는 분포 제약도 유지한다.

## 실사 평가를 진행할 경우

source 통과 후6개 checkpoint·normalization·모든 code/target binding·고정 tie/failure rule을 먼저 잠근다. 기존173장과 후보 입력을 그대로 사용해 모든 whole-pose 선택을 저장한 뒤 실사 참조로 평가한다. 여기서는 feature 구성·scale·epoch·seed·적용 여부를 바꾸지 않는다. missing candidate 또는 scorer failure를 원래 분모에서 보존하며, 이용 가능한 유일 후보는 그대로 고른다.0후보는 두 arm 모두 저장된 R0+GEO whole pose가 available이면 그 pose를 쓰고, 아니면 failure로 기록하는 규칙으로 확정한다.

natural99/clean29/wood45, 모든3seed, 원래 R0/PRIOR1/FULL125/SINGLE/DIVERSE 및 matched R0_ONLY를 함께 제공한다. joint T/R·P90·failure·recording 재표본/제외 기준을 유지한다. UNION이 R0_ONLY보다 낫지 않으면 “보정기를 회복했다”고 할 수 없다. R0_ONLY 자체가 좋아진 경우는 R0 selector calibration의 결과로 별도 해석한다.

기존고정 pool의 T 하한은 model 간 합집합에 그대로 적용되지 않는다. 반대로 union은 detector bbox나 후보 밖의 pose를 만들지 않으므로 원래 잘못 선택된 객체/후보 pool에 없는 팔레트를 해결하지 못한다. 전체 목표가 이 구조만으로 달성된다는 전제를 두면 안 된다.

## 사용 가능한 구현과 준비 한계

- `scripts/research/pallet_selector_recovery_v1/{features,feature_contract,models,synth_labels}.py`: feature94 정의, 기존 Linear 계산, exact renderer 연결의 참고 구현이다.9점 feature/projection/residual과 corner8 SQPnP/LM fitting을 구분한다.
- `scripts/research/pallet_pose_stable_improvement_20261001_v1/infer.py` 및 기존 refiner `CORE.predict(...,cap_fraction=None)`: 동결 DIVERSE 적용 계약. source TRAIN GT를 받는 `SourceData.item()`을 inference 입력으로 대신 사용하지 않는다.
- `data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer/{SYNTH_INPUTS.json,SYNTH_RECORDS.json}` 및 `challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz`: 기존 source 분할·K·치수·renderer pose. hash/row 대응과 source-only 읽기 guard 필요.
- 기존 감사 기준 R0 full-confidence source cache 재사용은 TRAIN129/VAL49장이다. TRAIN+VAL5120장 전체를 준비하려면 누락4942장 R0 forward와, DIVERSE3개×5120=15360 frozen-refiner forward가 필요하다. 이는95-parameter fits 비용과 별개이며 smoke·실패 처리·runtime parity를 별도 예산에 포함해야 한다. 이번 감사에서는 이 forward를 실행하지 않았다.

## 구현 검산과 실행 경계

`train.py selfcheck`는 CPU의 분석용 배열만으로 paired 초기값·seed별 순서 차이·330 update 계산, masked CE의 부호·원래 batch 분모·invalid gradient0,0/1후보 loss0, 정확한 cost/Pareto/R0/가설 tie, 전체 pose tradeoff 선택, inference score parity를 통과했다. 실제 source fit은 실행하지 않았다. trainer는 SHA로 봉인된 feasibility protocol, `SOURCE_TRAIN_GATE.PASS`, 별도 TRAIN_PROTOCOL, feature/label/contract/runtime amendment 연결이 모두 일치해야 실행된다. TRAIN process는 실사 참조·VAL quality·원본 source GT container 접근을 차단하고 적격2598행만 담긴 SOURCE_TRAIN_LABELS를 읽는다.

최대6개의 이름이 고정된 fit에 START, step별 TRACE, final checkpoint, 완료 receipt를 남긴다. 부분 실행은 자동 재개/재시도하지 않고, 이미 완료된 fit은 입력·trace·상태 hash를 검증한 뒤 재학습 없이 반환한다. 두 arm의 초기값, 정규화, 각 epoch/batch source row 순서가 같음을 완료 시 다시 검사한다. 이 검산은 source gate나 T/R 개선을 대신하지 않는다.

**최종 권고:** 통합4후보+shared Linear94+공통 source worst-axis cost를 한정된 가설로 검토할 수 있다. source 데이터/좌표계/whole-pose target/축별 gate를 먼저 고정한 뒤에만6fit 비교가 해석 가능하다. 이미지 증거 부재와 기존 router의 음성 결과 때문에, source gate를 생략한 직접 실사 최적화 또는 추가 feature sweep으로 진행하는 설계는 권하지 않는다.
