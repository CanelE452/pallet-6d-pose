# R0·PoseFix 전체 pose 통합 선택: 선행 실험과 방법 감사

2026-10-01. 기존 코드·고정 보고서만 검토했다. 이 감사에서 신규 이미지 추론, selector fit, 실사 채점은 실행하지 않았다. 근거 파일의 SHA와 확인 범위는 [PRIOR_AND_METHOD_AUDIT.json](PRIOR_AND_METHOD_AUDIT.json)에 보존한다. 과거 로컬 경로는 공개 여부가 서로 달라 코드 표기로 적었다.

**조사한 저장소에서 현재 R0와 동결 DIVERSE251_s의 W/D 네 후보를 source T/R로 공동 순위화한 실행은 찾지 못했다.** 따라서 정확한 설정의 반복은 아니다. 다만 합성으로 학습한 expert router는 이미 실행됐고, 기본 expert를 이기지 못한 부정적 결과가 있다. 후보 합집합에 좋은 pose가 있다는 사실을 선택기의 실현 가능한 개선으로 바꾸어 말하면 안 된다.

## 가장 가까운 선행: S0/S1 Stage4 router

`scripts/research/pallet_selector_recovery_v1/build_router_dataset.py:14–17,76–96`과 `router_features.py:45–55`는 먼저 각 expert의 W/D를 선택한다. 실제 고정값은 **PRODUCTION_D9**였다. Stage3의 합성 GEO가 S0 Moderate를 손상시켜 양쪽 D9로 돌아갔기 때문이다. 이후 이미 선택된 S0 pose 하나와 S1 pose 하나, 총 두 개 중 router가 하나를 고른다. 네 개의 W/D 후보를 동시에 순위화하지 않는다.

두 expert는 현재 checkpoint SHA `970a0913…`의 R0와 그 좌표를 보정하는 PoseFix 조합이 아니라 과거 S0/S1 YOLO 학생이다. 입력은 confidence·잔차·selector margin·두 expert의 box/점/pose 불일치와 frozen S1 GAP448이다. `router_features.py:7–43`의 더 풍부한 입력을 MLP64/32가 처리한다. source clean/occluded 쌍 TRAIN8192/VAL2048/TEST2048, seed42, 최대30epoch/patience5, VAL expert-choice accuracy로 earliest best를 골랐다. 라벨은 두 pose 중 작은 **exact synthetic C2 ADDnorm**이며, 공동 T/R 손실이 아니다.

| 기존 결과 | S0 | S1 | ROUTED | 사후 GT oracle |
|---|---:|---:|---:|---:|
| synthetic TEST 전체 mean ADDnorm | 0.09621724 | 0.09840939 | 0.09768815 | 0.09031495 |
| synthetic TEST clean mean ADDnorm | 0.08806852 | 0.09147516 | 0.08941924 | 0.08453126 |
| synthetic TEST occluded mean ADDnorm | 0.10436596 | 0.10534363 | 0.10595706 | 0.09609864 |
| 실사 CLEAN29 ADD AUC | 0.711879 | 0.698638 | 0.702845 | 0.724879 |
| 실사 SEVERE78 ADD AUC | 0.144782 | 0.195397 | 0.192179 | 0.206641 |

근거: `_docs/experiments/pallet_selector_recovery_v1/stage4_clean_preservation/STAGE4_REPORT_KO.md:3–16,20–42`. VAL 정확도0.552246, best epoch5, TEST 정확도0.5415였다. ROUTED의 source 전체 평균 ADD는 더 좋은 고정 expert S0보다 나빴고, occluded에서는 두 expert 모두보다 나빴다. 실사 판정도 **CLEAN_RECOVERY_BUT_HARD_LOSS**였다. 이 결과는 “router가 없어서 개선을 놓쳤다”는 설명을 지지하지 않는다. 새 시도는 후보 구조·실제 checkpoint·T/R 감독이 달라야 하며, 더 작은 Linear94가 성공한다고 전제하지 않는다.

## 혼동하면 안 되는 다른 선행

| 선행 | 실제로 선택한 대상과 학습 | 실제 결과 / 이번 방법과의 차이 |
|---|---|---|
| model-conditioned GEO | S1/H_MANUAL 각각의 long/short 두 후보,94-feature shared linear, exact W/D parity | HMAN ALL128 AUC0.358570→0.373285, S1 자기 출력 보정은 혼합·악화. 현재 R0/PoseFix도 아니고 expert 간 T/R 선택도 아니다. |
| clean-pose current-domain calibration | RAW/REF YOLO 학생 출력의 두 W/D 후보에 공통 Linear94, exact parity | seed에 따라 손익이 갈렸고 안정적 공동 T/R 회복이 입증되지 않았다. 같은 특징의 재보정만으로 전이가 보장되지 않는 선행이다. |
| N2/Replay utility selector | source RGB+점 특징 CNN/MLP가 코너별2D utility 예측, 수직 모서리쌍 단위 선택 | 합성 clean PCK10 95.233→93.618%,9점 회복/51점 손상. GREEN150 81.351→78.267%. 전체6D pose 선택이 아니지만 보정 수락의 clean 손상 위험은 실제 음성 근거다. |
| frozen R0 six-view medoid | 6개 실제 예측의 C2 quotient 2D 거리 medoid, fit 없음 | hard corner23회복/281, good corner24손상/511로 damage gate 실패. learned physical-pose ranker는 아니다. |
| large-corner source selector | 6 R0 view×native/reindexed90,183개 RGB/예측 특징, source2D utility RandomForest 설계 | TRAIN16≥10이나 VAL identity coverage2<3으로 **fit 전 중단**.1,280 source case 후보는 생성했지만 selector fit·실사194 선택/채점은0이다. 실패한 학습 성능으로 인용하면 안 된다. |
| four-expert whole-pose oracle | R0/teacher/RAW/REF 각자의 이미 D9-selected pose를 GT ADD로 선택 | Plastic AUC0.359016→0.441824는 사후 진단이다. W/D union도 학습된 runtime selector도 아니다. |

각 근거 경로는 JSON에 묶었다. 특히 `large_corner_selector.py:43–55,161–180`의 구현 가능성이 실제 학습 완료를 뜻하지 않는다. 보고서의 FIT_STOP을 우선했다. `ORACLE_COORDINATE_REPORT_KO.md:48–55`도 네 expert oracle가 이미 선택된 각 pose 하나씩임을 명시한다.

## 네 후보와94개 특징의 적용 계약

각 s∈{1,2,3}의 pool은 `[R0:long, R0:short, DIVERSE251_s:long, DIVERSE251_s:short]`다. 기존 selected detection, raw R0 RGB 경로, refiner crop·inverse affine, K, 물리 치수, corner index를 그대로 보존한다. DIVERSE 세 seed를 합쳐8후보로 만들거나 평가에서 좋은 seed를 고르지 않는다. 선택 결과는 한 후보의 완전한 `(R,t)`와 생성 expert/가설이다. R와 t를 따로 고르거나 점을 섞지 않는다.

`pallet_selector_recovery_v1/features.py:11–37`의 `F.extract(pred,K,dims,hw)`는 동일 함수로 각 출력의 long/short를 만들므로4×94로 쌓을 수 있다. 다만 pair 전체 valid를 요구하는 현재 wrapper가 한 가설 실패 때 다른 정상 후보까지 버릴 수 있으므로, 새 프로토콜은 per-candidate validity를 명시해야 한다. 유효하지 않은 특징을0으로 채워 정상 후보처럼 점수를 매기지 않는다. 실제 pose 객체와 특징을 동일 가설 이름으로 연결하고, 현재 diagnosis의 최종 pose 계약과 수치적 동등성을 먼저 검증한다.

**PnP fitting은 corner8점, 특징 입력·투영·잔차는 중심점까지9점**이다. `challenge/evaluation_v2/pnp_selector.py:405–429`의 SQPnP+LM은 `object_points[:8]`, `points[:8]`만 fit한 뒤9점을 투영한다. 기존9점 특징을9점 PnP라고 부르지 않는다. 새 ranker가 solver나 대칭 규칙을 바꾸는 개입이 되어서는 안 된다.

R0와 DIVERSE는 box·confidence·center8을 공유한다. Shared Linear94에서 동일 box/confidence 직접 항은 후보 간 score 차이에서 상쇄된다. 반면 투영/잔차·pose·extent 차이, confidence로 가중된 residual은 남는다. 특징에는 실제 RGB edge/patch 정합 증거가 없다. 기하적으로 일관되지만 틀린 팔레트 pose나 잘못 선택된 물체를 독립 시각 증거로 식별할 수 있는 구조는 아니다. 현재 detector 후보 밖의 팔레트를 생성하지도 않는다.

## 물리 source T/R 감독

old parity label은 R0-long과 DIVERSE-long 중 어느 pose가 좋은지 알려주지 못한다. prediction/feature를 모두 잠근 뒤 exact renderer `Xcf,R,t`를 연결하여 각 **완전한 후보**의 center translation과 physical rotation 오류를 계산해야 한다. source2D keypoint 오차나 ADD만으로 T/R label을 대신하지 않는다.

`synth_labels.py:7–29`와 `features.py:39–46`의 source convention은 주의가 필요하다. 기존 source pose는 `R_cf @ Q @ S`, `S=diag(1,-1,-1)`를 renderer `R_table`에 비교했다. 현재 real convention 후보가 `R_cf @ Q`이면 참조를 `R_table @ S`로 옮겨 비교하는 방식이 동등하다. C2 `diag(-1,1,-1)`와 S는 commute하며, geodesic rotation의 양쪽 right action으로 거리는 보존된다. 이 대수 관계만으로 renderer native corner의 물리 축 대응이 모두 검증됐다고 선언하지 않는다. source 계약 감사의 exact-coordinate SQPnP/LM fixture와 투영 검산을 통과한 변환을 최종 프로토콜에 고정해야 한다.

source 객체에 C1/C2가 섞일 수 있으므로 old `addnorm()`의 C2 하드코딩을 무조건 복사하지 않는다. 실제 source symmetry mapping을 따로 확인하고 현재 ordinary-plastic real C2와의 목적 차이를 공개한다. 같은 pose에 대해 가장 작은 T와 가장 작은 R를 서로 다른 후보에서 가져온 가짜 target은 금지한다. 실사 GT·오류·난도·recording은 fit 또는 runtime feature가 아니다.

## 권고하는 제한된 비교

[학습 설계 독립 검토](TRAINING_DESIGN_REVIEW_KO.md)의 **2arm×3seed, 최대6fit**이 원인 해석에 적절하다. 이는 실행 결과가 아니라 최종 protocol로 확정할 제안이다.

| arm | 후보 수 | 학습 반복 |
|---|---:|---|
| R0_ONLY_s | R0의2개 W/D | optimizer seed s=1,2,3 |
| UNION_s | 같은 R0의2개 + 해당 DIVERSE251_s의2개 | 같은 optimizer seed·초기 상태·source frame 순서 |

동일 shared Linear(94,1), R0 TRAIN-only 공통 정규화, 동일 batch256·AdamW 설정·30epoch final checkpoint, frame별 동일 가중치를 사용하면 최대 arm당480 update, 총2,880 update다. 기존 `models.fit()`는 seed42·2후보 parity BCE가 하드코딩돼 있으므로 그대로 호출하지 않고 새 namespace에서 가변 후보 masked loss를 구현해야 한다. arm별 후보 수가 다르다는 이유로 source frame 수 또는 update 수를 늘리지 않는다. refiner seed와 selector seed가 함께 바뀌는 세 쌍이므로 두 변동 원인을 분리한 요인 실험은 아니다.

제안된 단일 target은 source TRAIN의 frozen R0+GEO T/R 중앙값을 공통 scale로 둔 `max(T/sT,R/sR)`의 최소 **whole-pose 후보**다. shared score에 masked listwise CE를 적용한다. 이는 source만으로 축의 상대 가중치를 고정하는 설계이지 공동 개선의 보장이 아니다. exact cost tie는 Pareto 지배 후보 제거 후 R0 우선·가설 이름 순으로 고정하고, inference score tie도 사전 고정한다. 실제 선택에 GT Pareto 검사를 적용하지 않는다. 다른 weight/threshold/feature를 결과에 맞춰 추가 탐색하지 않는다.

source에서 cross-expert 개선과 손상이 모두 있는지, 같은 feature에 상충 label이 붙는지, 후보가 실제 구분되는지 먼저 보고한다. TRAIN whole-pose target 자체의 T/R 방향성과 P90·failure를 확인한 뒤에만6fit을 진행한다. 모든 checkpoint를 잠근 뒤 VAL을 한 번 채점하고,3seed 모두에서 matched R0_ONLY와 기존 operational baseline 대비 T/R·P90·failure의 사전 기준을 만족해야 실사로 진행하는 보수적인 source gate를 권한다. 미달 seed를 숨기거나 좋은 seed만 평가하지 않는다. clean source만 사용했다면 자연 가림의 source 전이를 이미 검증했다고 주장하지 않는다.

실사에 진행하더라도 원래 자연99/clean29, recording bootstrap·leave-one-recording-out,5% 보존·full failure 규칙을 바꾸지 않는다. 기존 SINGLE/DIVERSE/FULL125/PRIOR1/R0와 새 R0_ONLY를 함께 보여야 한다. **UNION이 R0_ONLY보다 나아야 PoseFix 후보의 기여를 주장할 수 있다.** R0_ONLY만 좋아지면 raw R0 선택기 보정 결과로 분리해서 보고한다. 이미 반복 열람한 실사 DEV를 독립 확인으로 부르지 않는다.

## 비용과 진행 판단

기존 인증 캐시는 TRAIN129+VAL49=178장만 confidence를 포함한다. 최소 source TRAIN4096/VAL1024 준비에는 누락 **4,942 R0 forward**, 고정 smoke 최대8장, **DIVERSE3×5,120=15,360 frozen refiner forward**가 필요하다. 신규 이미지 모델 학습은0이며6개의95-parameter selector fit과 별도 비용이다. TEST1024와 새로운 실사 image forward는 이 최소안에서 필요하지 않다. RGB SHA·pad/K·동일 runtime 및 물리 라벨 계약을 먼저 검증해야 한다.

이전 DIVERSE 고정 pool의 자연 T P90 하한은 후보가 늘어난 이 union에 그대로 적용되지 않는다. R0 후보가 있다는 사실 역시 clean/tail 보존을 자동 보장하지 않는다. **정확한 설정은 미실행으로 확인돼 제한된 source gate를 시작할 근거는 있으나, 과거 router와 utility의 음성 결과 때문에 성공 가능성을 확인된 사실로 제시할 근거는 없다.** source gate 실패 시 같은 DEV를 보며 ranker를 더 바꾸는 반복 대신 실패 원인과 이미 실행한 범위를 보존해야 한다.
