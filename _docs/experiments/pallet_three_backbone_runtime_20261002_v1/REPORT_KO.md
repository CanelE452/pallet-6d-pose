# YOLO·DOPE·ResNet-18 통합 속도 측정

## 측정 결론

세 기반 추정기를 **같은 26개 DEV 영상 바이트**에서 다시 실행했다. 모든 행은 batch 1, arm별 20회 warmup, 5회 반복이며 시작점은 RAM에 디코딩된 native BGR이다. 아래 수치는 모델 로드와 파일 디코딩을 제외하고 2D 출력까지, 그리고 같은 prediction-only MAIN PnP까지를 각각 잰 값이다.

## 논문 본문용 주 비교

| 방법 | 2D median ms | 2D P90 ms | PnP median ms | 전체 median ms | 전체 P90 ms | pose 상태 |
|---|---:|---:|---:|---:|---:|---|
| YOLO R0 | 9.554 | 11.594 | 1.444 | 11.092 | 13.184 | OK:130 |
| YOLO + P1 | 13.174 | 15.468 | 1.409 | 14.779 | 16.959 | OK:130 |
| DOPE | 62.894 | 76.766 | 2.172 | 64.585 | 78.870 | CANONICAL_SELECTOR_EXCEPTION:15, FEWER_THAN_SIX_FINITE_CORNERS:10, NO_DETECTION:15, OK:90 |
| DOPE + P1 | 66.291 | 80.003 | 1.420 | 67.411 | 81.361 | CANONICAL_SELECTOR_EXCEPTION:15, FEWER_THAN_SIX_FINITE_CORNERS:10, NO_DETECTION:15, OK:90 |
| ResNet-18 FULL | 7.835 | 10.432 | 1.452 | 9.453 | 12.146 | OK:130 |
| ResNet-18 + P0 (seed 1) | 10.520 | 13.565 | 1.455 | 12.193 | 14.982 | OK:130 |

주 비교의 ResNet 보정기는 합성 source 선택 규칙을 고정한 `P0_S1` 한 개다. 속도 측정에서 12개 평가 head를 순차 실행하지 않았다.

## ResNet 보조 ablation

| 방법 | 2D median ms | 2D P90 ms | PnP median ms | 전체 median ms | 전체 P90 ms | pose 상태 |
|---|---:|---:|---:|---:|---:|---|
| ResNet-18 + D0 (seed 1) | 10.104 | 13.410 | 1.477 | 11.886 | 14.838 | OK:130 |
| ResNet-18 + P5_CONSTANT (seed 1) | 11.442 | 13.790 | 1.475 | 13.132 | 15.228 | OK:130 |
| ResNet-18 + P5 (seed 1) | 11.302 | 13.977 | 1.488 | 12.963 | 15.342 | OK:130 |

`P5_CONSTANT_S1`과 `P5_S1`은 같은 구조에서 correction head의 5차원 치수 문맥만 바뀌는 보조 비교다. FULL backbone 자체는 이미 이미지와 치수를 함께 사용한다.

## 대응 추가 비용

| 보정 방법 | 대응 baseline | 추가 2D median ms | 추가 2D P90 ms | 추가 전체 median ms | 추가 전체 P90 ms |
|---|---|---:|---:|---:|---:|
| YOLO + P1 | YOLO R0 | 3.288 | 6.035 | 3.419 | 6.112 |
| DOPE + P1 | DOPE | 2.886 | 7.774 | 2.463 | 7.148 |
| ResNet-18 + P0 (seed 1) | ResNet-18 FULL | 2.544 | 5.525 | 2.436 | 5.500 |
| ResNet-18 + D0 (seed 1) | ResNet-18 FULL | 2.412 | 5.502 | 2.507 | 5.540 |
| ResNet-18 + P5_CONSTANT (seed 1) | ResNet-18 FULL | 3.361 | 5.932 | 3.322 | 5.931 |
| ResNet-18 + P5 (seed 1) | ResNet-18 FULL | 3.314 | 6.005 | 3.407 | 5.937 |

추가 비용은 같은 repeat·같은 frame의 보정 arm에서 해당 baseline 시간을 뺀 대응 차이다. 음수 표본도 그대로 포함했고 가장 빠른 반복을 고르지 않았다.

## 재현·동일성 검사

- warmup 180회와 본 측정 1170회를 모두 수행했다.
- 저장된 DEV 2D 출력 재현의 전체 최대 절대차는 `0.000265` px다.
- 저장된 pose 상태·축·지표 재현의 전체 최대 절대차는 `0.0002`다.
- parity 계산과 GT 기반 수치 대조는 timer 종료 뒤에만 수행했다. GT는 추론이나 PnP 입력에 들어가지 않았다.
- CPU thread는 Torch intra-op 1, inter-op 1, OpenCV 1로 고정했다.
- 저장 출력 재현을 위해 CUDA 수치 설정도 source 실험대로 복원했다: YOLO cuDNN TF32 on, DOPE·ResNet-18 cuDNN TF32 off다. 설정 전환은 timer 밖이다.
- 측정 순서는 repeat마다 arm을 순환 이동하고 홀수 repeat에서 뒤집었다.

## 해석 범위

이 표는 동일 데스크톱 GPU에서 얻은 기술적 latency 비교다. 26장은 재사용 DEV이며 accuracy의 독립 반복이 아니다. Jetson 속도, 처리량 최적화, 동시 요청 성능을 뜻하지 않는다. 보정 전후 정확도 주장은 각 backbone의 별도 DEV 결과와 함께 읽어야 한다.

## 원자료

- [고정 측정 계약](PROTOCOL.json)
- [전체 결과와 1,170개 측정 행](RESULTS.json)
- raw JSONL: `data/pallet/results/pallet_three_backbone_runtime_20261002_v1/runtime/attempt_0001/ROWS.jsonl`

GPU 기록: `NVIDIA GeForce RTX 3080, 580.178.04, 806 MiB, 10240 MiB, 48, 11 %`
