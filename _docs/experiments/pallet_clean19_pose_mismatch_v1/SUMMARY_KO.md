# 진단 요약

**중간 가림의 하락은 W/D 선택 오류만으로 설명되지 않는다. 두 후보에서 정답 기반 최선을 골라도 S1/S2의 ADD가 S0보다 낮아, 예측점의 상대 배치·pose fit 문제가 남는다. 심한 가림에서는 W/D 선택 오류의 영향이 더 크다.**

PRIMARY: CASE B — KEYPOINT GEOMETRY BOTTLENECK (MODERATE; not candidate-absence claim)

SECONDARY: CASE A — W/D SELECTOR BOTTLENECK (especially SEVERE)

NEW_TRAINING: 0. LOO: blocked (finite9 input contract).

| 난도 | 모델 | PCK10 % | Axis % | ADD AUC | wrong | alternate parity-correct | both parity-bad | oracle Axis Δpp | oracle ADD Δ |
|---|---|---|---|---|---|---|---|---|---|
| MODERATE_OCCLUSION | S0 | 63.34 | 77.01 | 0.3846 | 20 | 20 | 0 | 8.05 | 0.0461 |
| MODERATE_OCCLUSION | S1 | 66.32 | 72.41 | 0.3535 | 24 | 24 | 0 | 16.09 | 0.0529 |
| MODERATE_OCCLUSION | S2 | 65.13 | 74.71 | 0.3277 | 22 | 22 | 0 | 10.34 | 0.0547 |
| SEVERE_OCCLUSION | S0 | 42.17 | 53.09 | 0.1463 | 38 | 38 | 0 | 35.80 | 0.0986 |
| SEVERE_OCCLUSION | S1 | 44.73 | 60.49 | 0.1961 | 32 | 32 | 0 | 30.86 | 0.0867 |
| SEVERE_OCCLUSION | S2 | 46.17 | 53.09 | 0.1942 | 38 | 38 | 0 | 40.74 | 0.1057 |

PCK_UP_POSE_DOWN: {'CLEAN': {'S1': 7, 'S2': 6}, 'MODERATE_OCCLUSION': {'S1': 6, 'S2': 6}, 'SEVERE_OCCLUSION': {'S1': 5, 'S2': 7}}

TOP_INFLUENTIAL_CORNERS (GT oracle count): [(5, 32), (6, 28), (4, 22)]

TOP_INFLUENTIAL_EDGES (descriptive frequency, not causality): [{'severity': 'MODERATE_OCCLUSION', 'model': 'S2', 'edge': '4-5', 'n': 87, 'direction_delta_median': -0.0074059913030369295, 'PCK_up_direction_worse': 13}, {'severity': 'MODERATE_OCCLUSION', 'model': 'S2', 'edge': '4-7', 'n': 87, 'direction_delta_median': 0.05414469026833357, 'PCK_up_direction_worse': 10}, {'severity': 'MODERATE_OCCLUSION', 'model': 'S2', 'edge': '5-6', 'n': 87, 'direction_delta_median': -0.1032395207000869, 'PCK_up_direction_worse': 10}, {'severity': 'MODERATE_OCCLUSION', 'model': 'S1', 'edge': '4-7', 'n': 87, 'direction_delta_median': 0.14084336853229518, 'PCK_up_direction_worse': 9}]

NEXT_ONE_EXPERIMENT: Independent-session paired audit with existing checkpoints: validate corner and W/D references independently, then compare S0/S1/S2 on the same frozen frames; no training/selector tuning. Not executed.

[이미지 포함 전체 보고서](REPORT_KO.md)
