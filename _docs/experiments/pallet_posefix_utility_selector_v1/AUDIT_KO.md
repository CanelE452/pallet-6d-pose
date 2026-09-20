# Utility selector v1 — 독립 구현·산출물 감사

검토일: 2026-09-20. 검토자는 잠금된 실행 코드를 수정하지 않았고 GPU를 사용하지 않았다. CPU 단위검사, 독립 NumPy 재계산, 체크포인트·파일 해시 검사와 CPU 추론 재현을 수행했다. 이 감사는 새 독립 평가 데이터로 수행한 성능 검증을 의미하지 않는다.

## 판정

**구현 및 결과 집계 감사 PASS. 그러나 최종 모델 교체를 지지하는 성능 결과는 아니다.**

| 고정 평가 집합 | 기존 N2 정답/분모 (10px) | 학습 선택기 정답/분모 | N2 대비 개선 / 손상 |
|---|---:|---:|---:|
| DEV72 기존 참조점 | 330 / 555 | 333 / 555 | 12 / 9 |
| GREEN150 수동 점 | 554 / 681 | 533 / 681 | 4 / 25 |
| GREEN150 전체 기존 점, 보조 proxy | 964 / 1,200 | 931 / 1,200 | 8 / 41 |

DEV72의 참조점은 수동 GT로 검증되지 않은 기존 라벨이다. 초록 수동 점에서 21개가 순감소했으므로 이 선택기를 기존 N2보다 우수한 최종 모델로 발표하면 안 된다. 재학습·임계값 조정·새 체크포인트 선택 없이 결과를 보존한다.

## 학습 입력과 분리

- 합성 원본 1,616장: TRAIN 1,286장 / 1,003개 시나리오, validation 330장 / 251개 시나리오. 같은 원본의 clean/stress 두 변형은 항상 같은 분할에 속한다. 총 TRAIN 2,572개, validation 660개 예제다.
- 시나리오 및 이미지 해시의 train/validation 교집합이 없다. 기존 Replay probe의 249개 시나리오 전체가 제외되어 있다. 실사 TRAIN 9장, DEV72, GREEN150 이미지 해시와도 겹치지 않는다.
- 원래 N2/PRIOR의 가중치 학습 행과 selector 원본 pool은 겹치지 않는다. 다만 R0의 과거 validation/checkpoint 선택 및 합성 개발 평가 이력은 존재한다. 따라서 완전히 처음 보는 후보 생성 데이터라고 표현하지 않는다.
- 원본 pool은 C1 448장, C2 1,168장, C4 0장이다. 정사각형 초록 도메인을 학습했다고 주장할 수 없다.
- 검토한 feature API와 실제 train 배치는 RGB, 점 주변 패치, 수치 특징, 전체 물체 특징만 모델에 전달한다. 실사 GT는 학습·선택 입력에 들어가지 않는다. 실제 222장 선택 출력을 먼저 잠근 후 GT로 점수를 계산한다.

## 합성 utility 독립 재계산

`SOURCE_COMPLETE.json`에 바인딩된 `source_cpu_geometry_v2/SOURCE.npz`를 검사했다. 원본 cache, 예측 box, 원본 GT와 승인된 대칭 순열로 직접 다시 계산했으며 실행 코드의 `utility_targets` 함수를 호출하지 않았다.

- 전체 3,232개 예제, 유효 corner utility 25,592개가 정확히 일치했다. 최대 target 차이는 **0.0**이다.
- N2에서 한 번 고른 전체 물체 대칭 branch, 같은 branch에 정렬된 두 후보의 GT, supervision mask가 모두 일치했다.
- utility 부호 및 `100 × (N2 오차 − Replay 오차) / 예측 box 대각선` 정규화가 맞다. 양수는 Replay 후보의 오차 감소를 뜻한다.
- 원본 행별 SeedSequence로 stress를 독립 재생성했다. 두 vertical pair의 이동, centroid 보존, 정규화된 점·변위 특징과 64차원 전체 특징이 저장된 값과 정확히 일치했다.
- 1,616개 clean N2 좌표가 과거 저장된 N2 좌표와 bit-exact였다.

## 수치 실행 정정 검토

최초 FP32/batch1 재현이 사전 0.001px 검사를 넘겨 학습 전에 중단되었다. 과거 N2의 cuDNN TF32, 원래 validation batch16 구성, CPU decode를 복원하는 별도 wrapper를 추가했다. 첫 wrapper에서 GPU/CPU box norm 차이로 강화된 bit-exact 검사가 멈춘 기록도 보존했다. 이때 차이 0.0000400543px는 원래 0.001px 기준 이내였다.

최종 v2 wrapper는 CPU box norm/displacement 및 raw→prepared 덧셈 순서까지 복원한다. 원래 잠금 코드·프로토콜·0.001px 허용치·split·utility·학습량·임계값은 바뀌지 않았다. 과거 부분 chunk는 보존되고 최종 source는 별도 하위 경로에서 다시 생성되었다.

`N2_RUNTIME_CORRECTION_V2.json`과 wrapper는 최종 source에 해시 바인딩되어 있다. 실제 clean 1,616개는 logits와 좌표 모두 bit-exact, 차이 0.0px였다. stress도 고정 N2를 실제 forward하며 해당 batch의 대상 행만 바꾼다. N2에는 batch 통계나 예제 간 상호작용 층이 없고 GT tensor가 전달되지 않는다. Replay는 기존 TF32=False 설정을 유지한다.

## 체크포인트·실사 출력·점수

- 새 선택기 53,233개 파라미터만 고정 1,500 step 학습했다. seed 1 초기화를 CPU에서 독립 생성해 계산한 가중치 변화 L2는 **19.473770141601562**로 FIT 기록과 일치한다.
- 마지막 step 체크포인트가 고정되어 있고 평가 기반 checkpoint 선택이 없다. 기존 R0/N2/Replay 체크포인트와 원래 잠금된 실행 코드 해시가 변하지 않았다.
- 222장 모두 두 끝점의 예측 utility가 엄격히 양수이고 유효할 때만 해당 pair의 Replay 좌표를 사용한다. 나머지는 정확히 N2 좌표다. 혼합 좌표 생성이나 GT를 사용한 재배정은 없다.
- 채택 pair는 DEV72 110개, GREEN150 227개다. 모든 출력의 centroid, detection 수·순서·box·score·keypoint confidence가 보존된다.
- 저장된 feature와 마지막 체크포인트로 CPU batch32 재추론한 222장 선택이 GPU batch1의 선택과 모두 같았다. utility의 장치·batch 수치 차이 최대값은 **0.000009059906**이며 선택 부호를 바꾸지 않았다. 이는 추가 재현 관측치이며 실험 임계값 변경이 아니다.
- 기존 3개 비교군에 대해 기록된 baseline parity 검사는 1,116개다. 별도로 학습 선택기의 **372개 frame×평가모드**를 실제 GT 좌표부터 독립 재계산했다. branch, 매칭, 고정 분모, canonical corner 오차, 관측점 오차가 일치했고 frame 평균 오차 차이는 0.0px였다.
- PCK 5/10/20, 정답 개수, 평균 오차와 gain/damage를 저장된 per-frame 결과로 독립 집계하여 최종 표와 일치함을 확인했다. N2→학습 선택기의 평가 대칭 branch 변경은 모든 평가모드에서 0건이다.

이 실험은 고정된 N2/Replay 두 후보 중 선택하는 작은 모델의 제한된 검증이다. 새 좌표를 찾아내는 보정기나 실사 self-training을 학습한 결과가 아니며, 여기서의 미개선을 모든 머신러닝 접근의 불가능성으로 일반화할 수 없다.
