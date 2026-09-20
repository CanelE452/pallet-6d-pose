# IoU 0.9 선별 후 일반 플라스틱 self-training

2026-09-20 KST. 일반 플라스틱만 재학습. 이전 결과·모델은 보존했고 자동 최종 모델 교체는 하지 않았다.

## 학습 조건

- 249장 후보 중 보정 전후 hull IoU≥0.9 AND 원본/변형 보정 hull 최소 IoU≥0.9인 167장으로 제한했다.
- 기존과 같은 seed9021 복원추출로 실사 512 slots를 구성했다. 실제 고유 노출은 162장(기존 학생 217장)이다. 추가 5장 탈락이 아니라 미샘플링이다.
- 합성 replay는 이전과 동일한 512장, epoch당 합성512+실사512. 5 epoch, batch16/nbs16, 실제 optimizer update320. 고정 last.pt만 평가했다.
- 이전 학생을 이어 학습하지 않고 동일 R0에서 독립 시작했다. AdamW·학습률·증강·true-ignore loss·seed는 이전과 같다.
- 보정 좌표/박스/학습 라벨 바이트는 기존과 동일하다. 수동 검토 JSON 및 코너 제외 마스크는 사용하지 않았다.
- 후보 선별 집합만 바뀌지만 고유 이미지 수와 중복 노출 분포도 함께 바뀐다. 같은 수 무작위 선별 대비가 없어 IoU 기준만의 인과적 우월성 주장은 하지 않는다.

## 평가 1 — F0 예측 품질

| 모델 | 유지 | 중앙 오차 px↓ | P90 px↓ | 모든 감독점≤20px 이미지 | Precision↑ |
|---|---:|---:|---:|---:|---:|
| R0 합성만 | 194 | 7.689 | 55.65 | 102 | 0.5258 |
| 기존 보정 self-training | 194 | 7.573 | 42.20 | 98 | 0.5052 |
| IoU 0.9 선별 self-training | 194 | 7.740 | 46.20 | 98 | 0.5052 |

F0는 기존 기본 적격성(검출 및 충분한 confident corner)만 적용한다. Precision은 AP precision이 아니라 모든 감독점이20px 이내인 이미지의 비율이다.

## 평가 2 — 검출·오검출·keypoint

| 모델 | AP50–95↑ | AP50↑ | AUROC↑ | FPR95↓ | kp med px↓ | kp P90 px↓ | IoU50 매칭 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 합성만 | 0.7194 | 0.8949 | 0.9890 | 0.0439 | 7.275 | 39.36 | 186/194 |
| 기존 보정 self-training | 0.7000 | 0.8795 | 0.9870 | 0.0658 | 7.546 | 40.09 | 190/194 |
| IoU 0.9 선별 self-training | 0.6998 | 0.8870 | 0.9876 | 0.0610 | 7.715 | 42.70 | 190/194 |

전체 194장 일반 플라스틱 + 동일 NEG2689. 보정기/IoU/LOO로 평가 이미지를 걸러내지 않은 학생 단독 추론이다. kp 오차는 IoU50 매칭 이미지의 감독 keypoint pooling이며 F0와 집계 조건이 다르다. FPR95는 각 모델 TPR95% 지점의 배경 FP 비율이다.

## 평가 3 — frozen MAIN pose

| 모델 | PoseCov↑ | AxisAcc↑ | R med°↓ | Yaw med°↓ | t med cm↓ | IoU3D↑ | ADDsym AUC↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 합성만 | 1.000 | 0.7062 | 2.133 | 1.194 | 10.472 | 0.58572 | 0.34476 |
| 기존 보정 self-training | 1.000 | 0.7165 | 2.287 | 1.274 | 10.169 | 0.57654 | 0.33829 |
| IoU 0.9 선별 self-training | 1.000 | 0.6804 | 2.279 | 1.274 | 11.904 | 0.56392 | 0.31712 |

기존 frozen prediction-only selector + SQPnP/RefineLM. GT axis oracle가 아니다. 기준6D는 2D annotation·K·치수 기반 기하 복원 reference이며 외부 측정6D GT가 아니다.

## 보조 — 전체 분모 symmetry-aware PCK20

| 모델 | PCK20 |
|---|---:|
| R0 합성만 | 77.86% |
| 기존 보정 self-training | 77.26% |
| IoU 0.9 선별 self-training | 76.60% |

허용 whole-object symmetry, 첫8코너, 검출/매칭 실패 penalty 포함. 위 논문 fixed-index9-keypoint median과 다른 지표이다.

## 한계·재현

- 단일 seed의 짧은 후속 실험, 반복 사용한 DEV 평가이며 독립 최종 확인이나 통계적 유의성 주장은 없다.
- 평가 결과를 보고 threshold/step/last checkpoint를 다시 선택하지 않는다. 고유 라벨 감소 및 노출 분포 변화도 결과에 포함된다.
- 기존 replay에는 empty-label negative가 없다. 이번에도 추가 negative 학습은 하지 않았다.
- 학습 완료: 77.4초, 실제320steps, R0 초기화 exact parity와 가중치 변경 검사 통과.
- 원시 결과: `data/pallet/results/pallet_type_selftrain_v1/iou_selftrain/SUMMARY.json`, `metrics/`, `SYMMETRY_METRICS.json`.
- 새 모델: `data/pallet/results/pallet_type_selftrain_v1/iou_selftrain/runs/IOU90/weights/last.pt`.
- 실행: `python -m scripts.research.pallet_type_selftrain_v1.iou_selftrain {prepare,train,eval,negative,score}` 순서. GPU phases는 각각 별도 process로 실행.
- commit/push, 기존 모델 덮어쓰기, 컴퓨터 재부팅 또는 드라이버 변경은 하지 않았다.
