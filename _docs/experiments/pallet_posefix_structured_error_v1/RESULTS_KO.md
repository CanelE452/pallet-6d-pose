# 구조적 입력 오류 합성 PoseFix — 300-step 통제 실험

사전 screen 통과: **False**. 기존 최종 모델 변경 없음. 추가 학습은 한 번 300 step만 실행했다.

## 변경한 것 / 보존한 것

- 기존 Replay와 같은 PRIOR1에서 독립 시작. seed1, 300step, TFAdam1e-4, 실사8+합성8, micro2, loss, BN, sample index/order, supervision count 동일.
- 기존 raw 입력 branch는 그대로. 기존 교란 branch만 radial / 인접 모서리 근처로 한 점 이동 / 인접 코너 한 쌍 교환을 1:1:1로 배합했다.
- 이 비율은 실사 평가 오류에서 추정하지 않은 사전 고정 pilot이다. 원 논문의 오류 통계 전체 재현이 아니다. 오류 종류뿐 아니라 오차 크기와 변경 코너 수도 달라진다.
- partial real GT에서 알려진 manual 코너만 사용. GT target은 바꾸지 않으며 전체 C2/C4 회전을 오류로 만들지 않는다. 다른 사람/물체 관절 swap을 그대로 이식하지 않는다.
- 평가 split은 기존 촬영분리 DEV72 + GREEN150 고정. green은 manual-only가 주 지표, all-known은 proxy 진단. train9/38코너 이외 새 실사 정답 없음.
- 주 출력은 raw. cap이나 좋은 checkpoint를 골라 주 결과로 대체하지 않는다. 이번 실행은 학생 self-training이 아니다.

## 2D 실제 평가

| 집합 | 모델 | PCK10 % ↑ | median px ↓ | P90 px ↓ | 복구 / matched hard | 훼손 / matched good |
|---|---|---:|---:|---:|---:|---:|
| DEV72 | R0 | 53.514 | 9.381 | 51.375 | 0/156 | 0/138 |
| DEV72 | REPLAY_RAW | 60.180 | 7.370 | 52.982 | 4/156 | 1/138 |
| DEV72 | STRUCT_RAW | 57.477 | 7.700 | 52.845 | 0/156 | 1/138 |
| DEV72 | STRUCT_CAP1PCT | 57.477 | 7.785 | 51.222 | 0/156 | 1/138 |
| GREEN150_MANUAL | R0 | 82.232 | 4.580 | 9.213 | 0/4 | 0/344 |
| GREEN150_MANUAL | REPLAY_RAW | 77.386 | 3.970 | 11.065 | 1/4 | 14/344 |
| GREEN150_MANUAL | STRUCT_RAW | 78.708 | 3.894 | 10.941 | 0/4 | 14/344 |
| GREEN150_MANUAL | STRUCT_CAP1PCT | 80.470 | 3.842 | 10.043 | 0/4 | 4/344 |
| GREEN150_ALL_KNOWN_PROXY | R0 | 80.333 | 4.237 | 10.113 | 0/36 | 0/651 |
| GREEN150_ALL_KNOWN_PROXY | REPLAY_RAW | 75.917 | 3.893 | 12.682 | 2/36 | 31/651 |
| GREEN150_ALL_KNOWN_PROXY | STRUCT_RAW | 77.167 | 3.736 | 12.177 | 0/36 | 25/651 |
| GREEN150_ALL_KNOWN_PROXY | STRUCT_CAP1PCT | 79.000 | 3.715 | 10.941 | 0/36 | 6/651 |

復구: R0>20→≤10px. 훼손: R0<5→>10px. green matched hard는 4개뿐이므로 단독 복구율 주장 금지.

## DEV72 동일 PnP 6D

| 모델 | R median ° ↓ | t median cm ↓ | IoU3D median ↑ | ADDsym AUC ↑ |
|---|---:|---:|---:|---:|
| R0 | 4.6509 | 9.5884 | 0.556279 | 0.291208 |
| REPLAY_RAW | 4.4962 | 9.1417 | 0.579577 | 0.314514 |
| STRUCT_RAW | 4.7605 | 9.5389 | 0.559833 | 0.311674 |
| STRUCT_CAP1PCT | 4.7665 | 9.4218 | 0.560607 | 0.297778 |

동일 prediction-only selector/SQPnP/LM; 참조는 기하 복원 6D이며 독립 물리 GT가 아니다. green K mismatch로 green 6D는 새로 계산하지 않는다.

## 합성 heldout256 보존

| 모델 | 입력 | PCK10 % | P90 px |
|---|---|---:|---:|
| Replay | clean | 90.950 | 9.320 |
| Replay | stress | 68.843 | 24.716 |
| Structured | clean | 92.878 | 7.571 |
| Structured | stress | 54.204 | 33.237 |

## 사전 판정 조건

- DEV72_damage_not_increased: True
- DEV72_PCK10_not_lower: False
- GREEN150_MANUAL_damage_not_increased: True
- GREEN150_MANUAL_PCK10_not_lower: True
- DEV72_more_recovery: False
- source_PCK10_retained: True
- source_P90_retained: True

## 실제 입력 종류 / 검증

{"real": {"raw_unchanged": 1160, "radial": 409, "pair_swap": 414, "near_corner": 417}, "source": {"pair_swap": 393, "raw_unchanged": 1189, "near_corner": 418, "radial": 400}}

실제 업데이트 300회, training/probe 경과 184.9초, peak allocated 2182.9MiB.
코너 교란 회귀검사, BN 불변, 감독 수 parity, 222장 검출/중심 보존888회, 기존 평가 parity744회, 기존 R0/Replay pose parity144회 확인.
원본 source/checkpoint/prediction 해시는 전후 검증한다. 제한된 1seed/300step 결과이며 성공/실패를 모든 구조적 오류 합성의 일반화로 확대하지 않는다.

재현: pallet-yolo26 환경에서 `python scripts/research/pallet_posefix_structured_error_v1/run.py {prepare,train,evaluate,finish,audit}`. 신규 경로에만 기록하며 기존 결과는 덮어쓰지 않는다.
