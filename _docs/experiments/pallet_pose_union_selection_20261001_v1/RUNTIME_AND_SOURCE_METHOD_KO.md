# R0·PoseFix 전체 pose 선택: 입력 계약과 실행 재현 검토

기준 시각: **2026-10-01 08:45 KST**. 이 문서는 source 후보 생성 및 학습 가능성 검사의 방법·예산·중단 이력을 설명한다. 작성자는 기존 코드·계약·실행 receipt를 읽고 SHA와 정합성을 확인했으며, 추가 이미지 추론·학습·품질 채점을 실행하지 않았다.

**확인된 결과는 입력·물리 좌표계 검사와 R0 재현 smoke 통과다. 새로운 T/R 개선은 아직 확인되지 않았다.** 아래의20,480개 출력은 동결할 전체 대상이며, 기준 시각에는 전체 예측 완료 lock이 없었다. 일부 파일 생성이나 준비 완료를 학습 성공·최종 목표 달성으로 부르지 않는다.

## 이번 방법과 대조군

각 seed에 대해 같은 R0 원본 출력의 두 W/D pose와 해당 동결 PoseFix 보정 출력의 두 W/D pose를 비교한다.

| 조건 | 선택 가능한 완전한 pose | 반복 |
|---|---|---|
| R0_ONLY | R0 long-face-front, R0 short-face-front | selector seed1/2/3 |
| UNION | 위 R0 두 pose + DIVERSE251_s의 같은 두 W/D pose | 해당 refiner seed와 짝지은 selector seed1/2/3 |

후보마다 동일한94개 기하 특징을 만들고 shared Linear(94,1)이 낮은 점수의 후보 하나를 고르는 설계다. expert ID, recording, 난도, 실사 GT 점수를 입력하지 않는다. 선택된 후보의 R와 t를 함께 유지하며, 서로 다른 후보의 T/R 최솟값이나 코너를 섞지 않는다. DIVERSE3seed를 한꺼번에 합치거나 실사에서 가장 좋은 seed만 고르지 않는다.

R0_ONLY는 “보정 후보를 추가한 효과”를 별도 selector 재보정 효과와 구분하는 대조군이다. UNION이 이것보다 나아야 PoseFix 후보가 기여했다고 주장할 수 있다. R0_ONLY만 좋아지면 raw R0 선택기의 보정 결과로 보고한다. 원래 FULL/PoseFix 연구 목표를 다른 목표로 바꾸지 않는다.

이전 DIVERSE-only W/D pool의 T 하한은 후보가 늘어난 UNION에 그대로 적용되지 않는다. 반대로 R0가 후보에 있다는 사실만으로 clean/tail 보존이 보장되지는 않는다. 선택기는 현재 박스에 없는 팔레트나 후보 밖의 새로운 pose를 만들지 않는다.

## 선행 실험과 구별되는 점, 남아 있는 부정적 근거

[선행·방법 감사](PRIOR_AND_METHOD_AUDIT_KO.md)와 [근거 SHA 목록](PRIOR_AND_METHOD_AUDIT.json)은 정확한 현재 R0/PoseFix의 네 후보를 source T/R로 공동 순위화한 실행 이력을 찾지 못했다고 기록한다. 이는 저장소 조사 범위의 결론이며 일반적인 router 발상의 최초성 주장이 아니다.

가장 가까운 기존 Stage4 router는 old S0/S1 각각에서 D9로 W/D를 먼저 고르고, 남은 두 pose 중 하나를 source ADDnorm 감독·MLP64/32·GAP448로 골랐다. source TEST mean ADDnorm은 더 좋은 고정 S0가0.096217, routed가0.097688이었다. 실사도 CLEAN AUC +0.004207과 SEVERE −0.003218이 함께 나타나 **CLEAN_RECOVERY_BUT_HARD_LOSS**였다. 현재 네 후보 공동 선택·checkpoint·T/R 감독과는 다르지만, 단순히 router를 추가하면 회복된다는 설명에 반하는 결과다.

N2/Replay의 코너쌍 utility 선택기는 합성 clean PCK10을95.233%→93.618%, GREEN150을81.351%→78.267%로 낮췄다. R0 six-view medoid도 hard corner23개를 회복하면서 good corner24개를 손상시켜 보존 기준을 통과하지 못했다. 별도 source RandomForest 선택기 제안은 검증 identity 사례가2<3으로 부족하여 **학습 전 중단**했으므로 학습 성능 실패로 인용하지 않는다.

새 Linear94에는 실제 RGB patch나 영상 경계와 pose의 정합 특징이 없다. R0와 refiner가 공유하는 box/confidence의 직접 항은 후보 간 선형 점수 차이에서 상쇄된다. 기하적으로 일관되지만 잘못된 pose를 구분할 독립 영상 증거가 부족하다는 제약을 유지한다.

## 데이터 적격성: 전체 입력과 학습 대상의 구분

기존 renderer-group 분할과 RGB·K·치수·패딩을 그대로 유지한다. 상세 검산과 한계는 [source 계약 설명](SOURCE_CONTRACT_KO.md), [최종 source 계약](SOURCE_CONTRACT.json), [물리 형상 적격성 개정](SOURCE_CONTRACT_AMENDMENT_01.json)에 있다.

| source 구분 | 전체 입력·후보 동결 | T/R ranker 학습/선택 적격 | 별도 진단으로 유지 |
|---|---:|---:|---:|
| TRAIN | 4,096 | 2,598 | C1 1,497 + 부적격 C2 1 |
| VAL | 1,024 | 1,024 | 0 |
| 합계 | 5,120 | 3,622 | 1,498 |

기존 C1 라벨을 C2로 바꾸지 않는다. 일부 C1의 front/back 부호는 현재 camera-facing 출력으로 faithful한 C1 물리 회전을 구분할 수 없어, 이번 C2 대상 학습·모델 선택 범위에서 제외하고 진단용으로 유지한다. 이는 실사 평가 프레임을 버리는 규칙이 아니다.

추가로 `G38__G__f26406`의 기존 Xcf는 canonical cuboid에 대한 대응이 det=−1인 반사였다. proper rotation SO(3)로 표현할 수 없는 입력 계약 오류이며 신경망 예측 오차가 아니다. 원래5,120행 모두에서 치수·중심·least-squares residual·AᵀA·det(A)=+1을1e−6 조건으로 확인하는 **메타데이터 규칙**을 먼저 고정했다. 원본 파일과 해당 행을 보존했고, 행 대체·좌표 수정·GT 오차 기반 필터를 적용하지 않았다. 개정 전 실패는 [보존된 source 계약](SOURCE_CONTRACT_PRE_AMENDMENT_01.json)에 남아 있다.

적격3,622행의 정확한 renderer 점을 현재 실제 candidate API에 넣은 기하 검사에서 최대 T는 약3.16×10⁻⁶cm, 최대 R은 약1.708×10⁻⁶°였다. 이것은 정확한 입력에서 좌표계·solver 계약을 검증한 결과다. 신경망의 실제 예측 성능이나 source 학습 성공을 의미하지 않는다.

기존 refiner가 source 일부에 이미 노출됐고 VAL/TEST의 renderer group도 각각1개다. 분할·과거 노출은 새로 고르거나 숨기지 않는다. source VAL을 전체 frozen pipeline에 완전히 미노출인 최종 TEST로 부르지 않는다. 이번 source 단계에서는 TEST1024를 사용하지 않는다.

## 전체 source pool과 물리 라벨을 연결하는 순서

[source 생성 코드](../../../scripts/research/pallet_pose_union_selection_20261001_v1/source_infer.py)는 동결된 R0와 DIVERSE251_s1/s2/s3에 같은5,120 입력을 사용한다. **5,120×4=20,480개 모델별 출력행**을 잠그는 대상이다. 각 출력에서 long/short 두 pose를 만들므로 후보는 최대40,960개지만, 실패한 후보를 정상 pose로 세지 않는다.

기존 인증 full-candidate R0 캐시178행은 점·box에100px를 다시 더해 prepared canvas 좌표로 복원한다. 점 confidence, 검출 score, 후보 순서, selected index를 유지한다. 이미 패딩된 RGB에100px를 추가로 붙이지 않는다. 나머지4,942행은 같은 frozen R0 경로로 생성한다. refiner는 기존 crop1.25·288×384·inverse affine·center8 보존 및 원래 box/confidence 계약을 유지한다.

[특징·TRAIN 검사 코드](../../../scripts/research/pallet_pose_union_selection_20261001_v1/source_features.py)의 두 CLI는 별도 프로세스다.

1. `freeze`: 전체 예측 lock과 네 completion receipt, 모든 예측 파일·checkpoint·protocol SHA를 확인한다. source 라벨 파일 접근을 audit hook으로 차단한 상태에서4모델의94개 특징과 기존 whole-pose 후보를 계산하고 `SOURCE_FEATURES.npz`, `SOURCE_POSES.json`, `SOURCE_FEATURE_LOCK.json`을 저장한다.
2. `train_gate`: 특징 lock과 사전 [TRAIN 가능성 검사 계약](TRAIN_FEASIBILITY_PROTOCOL.json)을 검증한 다음, source 적격 TRAIN2,598행에만 renderer 참조를 연결한다. VAL/C1의 모델 T/R 배열은 만들지 않는다.

현재 physical pose가 `R_cf @ Q`이면 source 참조는 `R_table @ diag(1,-1,-1)`, translation은 renderer t 그대로다. C2의 전체 회전 오차를 사용한다. PnP fit은 corner8점이고,94개 특징의 입력·투영·잔차에는 중심점까지9점이 들어간다. 이 둘을9점 PnP solver로 혼동하지 않는다.

후보 validity는 GT 없이 finite feature와 기존 pose available을 함께 검사한다. valid=false인 특징은 무시한다. 실패의 T/R는 둘 다+∞이며 frame 분모를 유지한다. NaN은 실패로 조용히 누락하지 않고 오류로 중단한다. 결과에는 full population과 conditional 통계를 구분한다.

## TRAIN gate와 이후 학습의 제한

동결된 R0+기존 GEO의 적격 TRAIN T/R 중앙값을 각각 sT/sR로 사용한다. 두 값은 finite·양수여야 한다. 각 whole-pose 후보의 cost는 `max(T/sT,R/sR)`이고, 같은 후보를 골라 T와 R를 모두 계산한다. exact cost 동률에서만 Pareto 지배 후보를 제거한 뒤 R0 우선·가설 이름 순으로 결정한다. **이 source target 처리의 GT/Pareto 비교는 실제 inference에 들어가지 않는다.** runtime 동률은 예측 score의 고정 순서만 사용한다.

각 UNION seed의 cost-best whole pose가 R0_ONLY cost-best whole pose보다 TRAIN full-population T/R 중앙값을 모두 엄격히 낮추고, T/R P90은 각각1.05배 이내이며 실패 수가 늘지 않아야 한다. 세 seed가 모두 통과해야 다음 작은 학습을 진행하는 계약이다. 조건은 후보를 만든 뒤 임계값을 바꾸기 전에 고정됐다.

이 검사는 해당 scalar cost와 source pool에 대한 가능성 검사다. 실패하면 이 고정된 cost 설계를 중단한다는 의미이며, 모든 UNION 선택기가 수학적으로 불가능하다는 뜻은 아니다. 통과도 특징으로 선택을 학습할 수 있거나 자연 가림에서 개선된다는 증거가 아니다.

통과 후 계획은 같은 source frame·정규화·초기 상태·shuffle·optimizer 예산의 R0_ONLY/UNION 두 arm×3seed, shared Linear94다. 적격 TRAIN2,598, batch256,30epoch이면 arm당11×30=**330 update**,6fit 최대**1,980 update**다. 이전 선행 감사의 전체 TRAIN4096 가정480 update/fit은 초기 제안이었고, 실제 후속 계획은 현재 적격성 계약의330을 따른다. 후속 학습 프로토콜·VAL 기준을 따로 잠그기 전에는 source 준비 예산을 selector fit 허가나 실행 완료로 해석하지 않는다.

## R0 smoke 실패와 명시적 runtime 개정

[첫 실패 기록](R0_SMOKE_FAILURE_01.json)은 고정 smoke의 첫 이미지 `G38__G__f0337`에서 keypoint 최대 차이 **0.00390625px**를 남겼다. 허용0.0001px를 넘었으므로 **1회 forward 뒤 대량 pool 추론 전에 중단**했다. 실패 attempt와 당시 코드·원래 protocol은 보존했다. 실패 직전 전체 prediction은 직렬화되지 않았으므로 존재하지 않는 출력을 사후 생성하지 않았다.

원인은 새 wrapper의 fusion 순서 차이였다. 과거 `FrozenYoloFeatures`는 CPU에 로드한 모듈을 Ultralytics backend에 넘겼고, 이 non-Jetson 경로는 **CPU Conv/BN fusion 후 GPU 이동**이다. 새 wrapper는 **GPU 이동 후 fusion**을 수행했다. fusion에는 행렬곱·sqrt/div로 새 weight/bias를 만드는 연산이 있어 같은 원 checkpoint라도 계산 위치에 따른 반올림 차이가 뒤의 좌표에 반영될 수 있다.

독립 코드 확인 근거는 기존 `pallet_dim_conditioned_p_v1/paper_evaluate.py:60–66`, `pallet_line_pose_v1/features.py:37–38,60–62`, 설치된 Ultralytics `nn/backends/pytorch.py:52–57`와 `utils/torch_utils.py:285–296`이다. 외부 설치 코드의 당시 사본·SHA도 개정 문서에 묶었다.

[runtime 개정](SOURCE_RUNTIME_AMENDMENT_01.json)은 CPU에서 `fuse()`한 뒤 CUDA로 옮기도록 과거 순서를 복원했다. **smoke8개 ID, 좌표·box 허용1e−4px, score/confidence 허용1e−6, rtol=0, 후보 수·순서·selected index 일치 조건을 그대로 유지했다.** GT나 실사 결과를 보고 threshold를 완화한 재시도가 아니다. 같은 smoke가 다시 실패하면 추가 재시도 없이 중단하도록 정했다.

[개정 후 smoke receipt](R0_SMOKE.json)는 같은8개 전부에서 keypoint·box·score·keypoint confidence의 최대 차이가 **0**임을 기록한다. 후보 수·순서·selected index도 유지됐다. 여기서 확인한 것은 과거 R0 입력·수치 경로의 재현성이다. T/R 개선 결과가 아니다.

## 정확한 예산

| 구분 | 최대 새 이미지 forward / fit | 설명 |
|---|---:|---|
| 보존한 첫 실패 smoke | 1 forward | 삭제·미계상하지 않음 |
| 같은 고정 smoke 재실행 | 8 forward | 원래 ID·허용오차 유지, receipt 통과 |
| 누락 R0 pool | 4,942 forward | 인증 캐시178행은 재사용 |
| 동결 refiner3개 | 15,360 forward | 각5,120 입력, 입력 불가 시 실제 forward 수 별도 기록 |
| source 준비 총 상한 | **20,311 forward** | R0총4,951 + refiner최대15,360 |
| 이 source 준비 단계의 새 fit | **0** | 원 이미지 모델 가중치 학습0 |
| 후속 조건부 작은 ranker 계획 | 최대6fit /1,980 update | TRAIN gate 이후 별도 프로토콜 필요 |
| source TEST / 신규 실사 이미지 forward | **0 /0** | 현재 source 준비 범위 |

입출력 행 수, 신경망 forward 수, optimizer update 수는 서로 다른 단위다. 완료 시간이나 GPU peak 사용량을 이 상한에서 추정해 실측처럼 보고하지 않는다.

## 개정·binding·읽기 경로 검토

[common.protocol 구현](../../../scripts/research/pallet_pose_union_selection_20261001_v1/common.py)은 원래 [SOURCE_PROTOCOL](SOURCE_PROTOCOL.json)의 SHA를 먼저 검증하고, 별도 runtime amendment의 SHA·원래 protocol binding·실패 smoke binding을 확인한다. 원본 JSON의 code SHA를 덮어쓰지 않고 effective code 목록과 실패 smoke 추가 예산만 메모리에서 적용한다. R0 receipt와 최종 prediction lock은 원본 protocol 및 runtime amendment를 따로 기록하도록 구현돼 있다.

검토 시 effective code binding **109개**, 원래 protocol·runtime amendment·TRAIN gate protocol SHA, 원래 코드 사본과 runtime 근거 사본, 동일8개 smoke ID/허용오차, 총 R0 예산4,951의 일치를 확인했다. 엄격한 source-label guard를 켠 읽기 검토에서119개 고유 경로를 접근했으며 source/real target 파일의 내용 접근이나 추론·fit은 없었다. 원본 protocol은 보존 기록이고 실제 재현에는 amendment까지 함께 적용해야 한다.

개별 source prediction의 `protocol_sha`가 원본 protocol SHA인 것은 원본을 바꾸지 않는 기록 방식이다. 그러므로 **개별 파일만 떼어 실행 경로 전체를 인증할 수 없다.** 최종 SOURCE_PREDICTIONS_LOCK → 모델별 completion receipt/파일 목록 → 원본 protocol+runtime amendment+effective code 목록의 묶음을 함께 검증해야 한다. 첫 실패 때 pool 출력0이었고, 실패 smoke attempt는 새 smoke8회와 별도 경로로 보존되므로 실패 출력이 정상 pool에 섞였다는 근거는 없다.

## 결과 대기: 기준 시각의 실행 상태

| 단계 | 확인된 상태 | 무엇을 아직 주장할 수 없는가 |
|---|---|---|
| source RGB·K·물리 형상 계약 | PASS, 적격 TRAIN2598/VAL1024 고정 | 신경망의 T/R 성능 |
| 첫 R0 smoke | FAIL,1회·대량 pool0회 상태에서 중단 기록 | 실패를 없었던 실행으로 처리 |
| CPU-fusion 복원 후 smoke8 | PASS, 네 필드 최대 차이 모두0 | 좌표/pose 개선 |
| R0 및 refiner 전체 source 예측 | **전체 completion receipt/lock 대기** | 20,480 출력행 완료 |
| 특징·pose 동결 | **미실행/lock 없음** | 실제 source 후보 오류 분포 |
| TRAIN cost gate | **미실행/결과 없음** | gate 통과 또는 실패 |
| ranker6fit·VAL·실사 평가 | **미실행/결과 없음** | T/R 공동 개선·실사 일반화·목표 완료 |

이 상태표는08:45 KST의 receipt 확인 기록이다. 이후 완료 여부와 실제 수치는 후속 receipt·결과 문서로 확인해야 하며, 이 방법 설명의 계획 수치를 측정값으로 인용하지 않는다.
