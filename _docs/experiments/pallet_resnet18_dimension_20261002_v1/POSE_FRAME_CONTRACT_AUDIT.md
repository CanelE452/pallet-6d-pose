# ResNet18 pose frame 계약 감사

## 감사 이유

같은 FULL 2D 예측을 사용했는데 direct 3-arm 보고서와 frozen-refiner 보고서의 T 중앙값은 같고 R·ADD-S AUC가 달랐다.

| 산출물 | T median cm | R median ° | normalized C2-aware ADD AUC |
|---|---:|---:|---:|
| direct 3-arm pose | 11.007430 | 3.581331 | 0.316284 |
| refiner/cross-backbone canonical MAIN | 11.007446 | 3.252385 | 0.312959 |

두 cache의 점 좌표 최대 차이는 0.001163px로 수치적으로 일치한다. 따라서 큰 R 차이는 2D 모델 출력이나 PnP translation의 차이가 아니라 회전을 평가하는 좌표계 계약의 차이다.

## 두 계약

direct 평가는 `scripts/research/pallet_dim_conditioned_p_v1/pose.py`의 `infer()`와 `metric()`을 사용한다. 선택된 camera-facing 회전 `R_cf`에 W/D 선택에 따른 `Q`를 곱해 `R_physical=R_cf@Q`로 바꾼 뒤, GT도 같은 고정 physical canonical frame으로 바꾸어 C2 대칭 회전을 계산한다. 따라서 잘못 선택된 W/D parity가 R·yaw·ADD에 포함된다.

refiner와 세 백본 통합 평가는 `scripts/research/pallet_resnet18_dim_refiner_20261002_v1/evaluation.py`에서 canonical MAIN 경로를 사용한다. `R_cf`를 `R_gt_representative`와 직접 비교하고, 선택된 W/D parity는 별도의 `axis_accuracy`로 보고한다. 이 계약의 FULL axis accuracy는 234/319, 73.354%다. 두 경로의 ADD는 선언된 proper C2 group에서 대응점 거리를 최소화하는 지표이며 unrestricted nearest-neighbor ADD-S가 아니다. IoU 계약도 direct는 선택된 예측 extents와 GT extents를 비교하고 canonical 경로는 GT camera-facing extents를 양쪽 cuboid에 사용하므로, W/D 축 오류에서는 차이가 날 수 있다.

## 행 단위 대조

두 산출물의 319개 frame ID는 완전히 같았다.

- translation 절대차의 전체 최대값: 0.001218 cm
- axis-correct 234장: R 절대차 중앙값 0.000007°, 최대 0.000153°
- axis-wrong 85장: R 절대차 중앙값 77.074638°, 최대 89.175012°
- axis-wrong 85장: normalized ADD 절대차 중앙값 0.431127, 최대 0.687440

따라서 R·ADD의 집계 차이는 85개 W/D 선택 오류를 회전 오차에 포함하는지, 별도 axis 지표로 분리하는지에서 생긴다.

## 사용 규칙

- direct `CONSTANT/SHAPE/FULL` 내부 비교는 모두 동일한 fixed-physical-frame 계약이므로 그 실험 안의 T·R ablation에만 사용한다.
- YOLO·DOPE·ResNet18 보정기의 주 전후 비교와 세 백본 통합 표는 모두 canonical MAIN camera-facing-frame 계약을 사용한다.
- 두 계약의 R 또는 ADD 절대값을 같은 열에서 섞거나, direct R을 canonical MAIN R로 대체하지 않는다.
- canonical MAIN의 R은 W/D parity를 포함한 완전한 physical canonical rotation으로 부르지 않고, camera-facing-frame C2 rotation으로 명시한다. W/D 선택 성능은 `axis_accuracy`와 함께 읽는다.

## 결론

FULL의 두 pose 표는 서로 다른 좌표계 질문에 답한다. 주 논문 표에는 세 estimator와 동일한 canonical MAIN 계약을 사용하고, direct 치수 ablation의 pose 수치는 별도 evaluator임을 표시한다.
