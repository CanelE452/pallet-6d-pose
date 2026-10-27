# DOPE 종류별 필터 제거 실험: 통합372장

주평가 모집단은 PLASTIC128/985점, WOOD125/963점, GREEN119/943점, 합계372장/2891점이다. WOOD는 기존1280×720 45장과 추가640×480 80장을 합쳤다. 원본 영상에서 5/10/20 px 이내인 점 수를 전체 참조점으로 나누며 미검출은 실패다. 해상도가 달라 같은 px가 같은 정규화 허용오차를 뜻하지 않는다. 기존45장 PCK10 60.12%와 새125장의 차이를 모델 개선으로 해석할 수 없다. 모델 간 비교는 동일125장에 짝지었다. conditional_precision10의 분모는 관측된 주석점이다. 점 오차 통계는 관측 표본 조건부이며 SD는 모집단 기준(ddof=0)이다. WOOD 수동 클릭330점과 GREEN 수동 클릭600점은 별도 보조 패널이다. 추가80은 원 FINAL 역할과 명시 승인 이력을 유지하며 학습 풀과 REC_001/002를 공유하므로 녹화 독립 TEST가 아니다. 같은 인덱스의 2D 데이터셋 채점 규약 호환성을 사용했으며 물리적 모서리 정확성이나 독립 실측6D 정답을 보증하지 않는다. 추가80의 K quality는 UNKNOWN, signed pose는 UNCONFIRMED다. 이 출력은2D 주평가이며 원 DEV292/WOOD45 6D 표는 역사적 보조 입력으로 명시한다. 통합125장6D는 별도 검증된 재구성 단계에서만 만들며80을0으로 채우지 않는다. 단일 학습 seed이며 프레임 bootstrap은 학습 seed 분산이 아니다. 다중비교 보정은 없다.

## PLASTIC

| 조건 | R | 상태 | PCK5 % | PCK10 % | PCK20 % | Coverage % | 조건부 Precision10 % | 관측/참조 점 | 오차 평균 ± SD px |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 0 | available_cached | 7.21 | 26.19 | 44.87 | 77.36 | 33.86 | 762/985 | 35.03 ± 45.84 |
| control | 1 | available_cached | 19.90 | 36.35 | 48.73 | 70.36 | 51.66 | 693/985 | 30.24 ± 53.40 |
| control | 2 | available_cached | 22.23 | 43.05 | 56.04 | 80.00 | 53.81 | 788/985 | 28.78 ± 52.64 |
| full | 1 | available_cached | 19.09 | 38.07 | 53.50 | 82.03 | 46.41 | 808/985 | 30.13 ± 46.12 |
| full | 2 | available_cached | 21.12 | 39.49 | 59.39 | 93.60 | 42.19 | 922/985 | 33.02 ± 51.73 |
| no_flip | 1 | available_cached | 15.03 | 33.81 | 48.02 | 81.22 | 41.62 | 800/985 | 36.64 ± 53.29 |
| no_flip | 2 | available_cached | 15.33 | 35.13 | 51.27 | 86.29 | 40.71 | 850/985 | 34.89 ± 51.08 |
| no_confidence | 1 | available_new | 17.97 | 36.75 | 54.01 | 87.31 | 42.09 | 860/985 | 34.27 ± 52.10 |
| no_confidence | 2 | available_new | 17.97 | 40.20 | 57.46 | 91.98 | 43.71 | 906/985 | 33.58 ± 52.88 |
| no_ransac | 1 | available_new | 17.56 | 37.26 | 53.60 | 86.70 | 42.97 | 854/985 | 32.56 ± 45.09 |
| no_ransac | 2 | available_new | 20.20 | 40.61 | 57.87 | 96.04 | 42.28 | 946/985 | 31.57 ± 43.48 |
| no_loo | 1 | available_new | 19.09 | 38.07 | 53.50 | 82.03 | 46.41 | 808/985 | 30.13 ± 46.12 |
| no_loo | 2 | available_new | 19.90 | 38.98 | 57.77 | 92.79 | 42.01 | 914/985 | 33.57 ± 52.16 |
| no_final_support | 1 | available_new | 16.75 | 38.17 | 55.03 | 85.58 | 44.60 | 843/985 | 32.61 ± 47.82 |
| no_final_support | 2 | available_new | 17.16 | 38.48 | 56.45 | 91.88 | 41.88 | 905/985 | 33.30 ± 50.61 |
| confidence_only | 1 | available_new | 17.16 | 34.72 | 49.85 | 81.83 | 42.43 | 806/985 | 37.20 ± 54.67 |
| confidence_only | 2 | available_new | 18.07 | 36.95 | 53.60 | 88.12 | 41.94 | 868/985 | 38.28 ± 54.73 |

## WOOD

| 조건 | R | 상태 | PCK5 % | PCK10 % | PCK20 % | Coverage % | 조건부 Precision10 % | 관측/참조 점 | 오차 평균 ± SD px |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 0 | available_cached | 4.78 | 31.26 | 64.28 | 81.62 | 38.30 | 786/963 | 23.73 ± 60.38 |
| control | 1 | available_cached | 26.17 | 55.76 | 69.99 | 78.50 | 71.03 | 756/963 | 15.59 ± 39.70 |
| control | 2 | available_cached | 26.06 | 54.00 | 70.20 | 81.00 | 66.67 | 780/963 | 19.48 ± 50.66 |
| full | 1 | available_cached | 15.06 | 53.06 | 79.23 | 88.37 | 60.05 | 851/963 | 17.37 ± 47.74 |
| full | 2 | available_cached | 21.08 | 62.31 | 82.87 | 92.73 | 67.19 | 893/963 | 18.24 ± 52.53 |
| no_flip | 1 | available_cached | 16.30 | 55.76 | 83.18 | 94.60 | 58.95 | 911/963 | 19.18 ± 48.81 |
| no_flip | 2 | available_cached | 22.85 | 62.31 | 85.36 | 97.51 | 63.90 | 939/963 | 18.75 ± 48.69 |
| no_confidence | 1 | available_new | 14.75 | 52.23 | 79.96 | 88.99 | 58.69 | 857/963 | 17.46 ± 47.48 |
| no_confidence | 2 | available_new | 20.35 | 59.92 | 82.24 | 91.59 | 65.42 | 882/963 | 15.21 ± 39.86 |
| no_ransac | 1 | available_new | 15.89 | 54.52 | 80.69 | 89.41 | 60.98 | 861/963 | 16.79 ± 45.48 |
| no_ransac | 2 | available_new | 21.18 | 62.41 | 83.59 | 93.04 | 67.08 | 896/963 | 17.80 ± 51.50 |
| no_loo | 1 | available_new | 15.16 | 52.23 | 80.06 | 88.99 | 58.69 | 857/963 | 17.52 ± 47.52 |
| no_loo | 2 | available_new | 20.66 | 60.33 | 82.87 | 92.21 | 65.43 | 888/963 | 17.40 ± 49.40 |
| no_final_support | 1 | available_new | 15.78 | 53.79 | 79.65 | 89.30 | 60.23 | 860/963 | 17.20 ± 44.74 |
| no_final_support | 2 | available_new | 21.81 | 61.68 | 83.28 | 92.42 | 66.74 | 890/963 | 17.36 ± 50.24 |
| confidence_only | 1 | available_new | 15.37 | 55.24 | 81.83 | 93.77 | 58.91 | 903/963 | 17.18 ± 42.91 |
| confidence_only | 2 | available_new | 22.95 | 63.14 | 87.23 | 97.82 | 64.54 | 942/963 | 16.94 ± 46.36 |

## GREEN

| 조건 | R | 상태 | PCK5 % | PCK10 % | PCK20 % | Coverage % | 조건부 Precision10 % | 관측/참조 점 | 오차 평균 ± SD px |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 0 | available_cached | 11.45 | 47.72 | 77.41 | 98.52 | 48.44 | 929/943 | 22.37 ± 39.78 |
| control | 1 | available_cached | 40.72 | 68.50 | 80.91 | 95.23 | 71.94 | 898/943 | 17.86 ± 36.78 |
| control | 2 | available_cached | 42.74 | 70.20 | 83.03 | 97.35 | 72.11 | 918/943 | 17.93 ± 37.19 |
| full | 1 | available_cached | 39.34 | 69.14 | 82.71 | 98.41 | 70.26 | 928/943 | 18.72 ± 37.32 |
| full | 2 | available_cached | 39.34 | 69.67 | 82.40 | 99.15 | 70.27 | 935/943 | 18.71 ± 37.09 |
| no_flip | 1 | available_cached | 35.31 | 67.97 | 82.40 | 98.30 | 69.15 | 927/943 | 18.70 ± 36.73 |
| no_flip | 2 | available_cached | 38.71 | 67.76 | 82.71 | 99.05 | 68.42 | 934/943 | 18.59 ± 36.36 |
| no_confidence | 1 | available_new | 38.92 | 68.82 | 82.93 | 98.41 | 69.94 | 928/943 | 18.59 ± 36.91 |
| no_confidence | 2 | available_new | 38.49 | 68.50 | 82.08 | 99.36 | 68.94 | 937/943 | 18.77 ± 36.72 |
| no_ransac | 1 | available_new | 39.87 | 69.03 | 82.82 | 98.41 | 70.15 | 928/943 | 18.59 ± 36.94 |
| no_ransac | 2 | available_new | 39.24 | 68.72 | 81.87 | 99.36 | 69.16 | 937/943 | 19.24 ± 37.76 |
| no_loo | 1 | available_new | 39.77 | 69.67 | 83.03 | 98.41 | 70.80 | 928/943 | 18.39 ± 36.90 |
| no_loo | 2 | available_new | 40.83 | 70.10 | 83.24 | 99.58 | 70.39 | 939/943 | 18.68 ± 37.04 |
| no_final_support | 1 | available_new | 37.54 | 69.78 | 83.14 | 98.52 | 70.83 | 929/943 | 18.42 ± 36.79 |
| no_final_support | 2 | available_new | 38.81 | 68.93 | 81.23 | 99.05 | 69.59 | 934/943 | 19.14 ± 37.08 |
| confidence_only | 1 | available_new | 32.13 | 66.28 | 82.18 | 98.30 | 67.42 | 927/943 | 18.89 ± 36.62 |
| confidence_only | 2 | available_new | 38.60 | 68.82 | 82.40 | 98.94 | 69.56 | 933/943 | 18.53 ± 36.35 |

## WOOD 수동 클릭 330점 보조 패널

| 조건 | R | PCK10 % | Coverage % | 조건부 Precision10 % |
|---|---|---|---|---|
| R0 | 0 | 40.91 | 80.61 | 50.75 |
| control | 1 | 62.73 | 77.88 | 80.54 |
| control | 2 | 59.39 | 78.18 | 75.97 |
| full | 1 | 64.55 | 85.45 | 75.53 |
| full | 2 | 70.91 | 90.00 | 78.79 |
| no_flip | 1 | 69.09 | 96.97 | 71.25 |
| no_flip | 2 | 73.64 | 99.70 | 73.86 |
| no_confidence | 1 | 64.24 | 87.27 | 73.61 |
| no_confidence | 2 | 69.39 | 90.61 | 76.59 |
| no_ransac | 1 | 67.88 | 87.88 | 77.24 |
| no_ransac | 2 | 73.03 | 91.52 | 79.80 |
| no_loo | 1 | 64.55 | 86.97 | 74.22 |
| no_loo | 2 | 70.00 | 90.30 | 77.52 |
| no_final_support | 1 | 64.55 | 88.79 | 72.70 |
| no_final_support | 2 | 71.52 | 90.91 | 78.67 |
| confidence_only | 1 | 66.97 | 95.76 | 69.94 |
| confidence_only | 2 | 74.55 | 99.70 | 74.77 |

## GREEN 수동 클릭 600점 보조 패널

| 조건 | R | PCK10 % | Coverage % | 조건부 Precision10 % |
|---|---|---|---|---|
| R0 | 0 | 59.83 | 98.83 | 60.54 |
| control | 1 | 75.33 | 94.83 | 79.44 |
| control | 2 | 77.83 | 97.33 | 79.97 |
| full | 1 | 77.50 | 98.33 | 78.81 |
| full | 2 | 78.50 | 99.00 | 79.29 |
| no_flip | 1 | 77.83 | 98.00 | 79.42 |
| no_flip | 2 | 77.83 | 99.17 | 78.49 |
| no_confidence | 1 | 76.83 | 98.17 | 78.27 |
| no_confidence | 2 | 77.83 | 99.50 | 78.22 |
| no_ransac | 1 | 77.00 | 98.17 | 78.44 |
| no_ransac | 2 | 77.33 | 99.17 | 77.98 |
| no_loo | 1 | 78.17 | 98.33 | 79.49 |
| no_loo | 2 | 79.00 | 99.50 | 79.40 |
| no_final_support | 1 | 77.67 | 98.33 | 78.98 |
| no_final_support | 2 | 78.17 | 99.00 | 78.96 |
| confidence_only | 1 | 76.67 | 98.00 | 78.23 |
| confidence_only | 2 | 78.83 | 99.00 | 79.63 |

## full − 제거 조건의 PCK10 차이

| 종류 | R | 제거 조건 | 차이 %p | 95% CI %p | 판정 |
|---|---|---|---|---|---|
| PLASTIC | 1 | no_confidence | 1.32 | [-1.32, 3.97] | includes_zero |
| PLASTIC | 1 | no_ransac | 0.81 | [-1.61, 3.15] | includes_zero |
| PLASTIC | 1 | no_loo | 0.00 | [0.00, 0.00] | includes_zero |
| PLASTIC | 1 | no_flip | 4.26 | [1.22, 7.32] | full_higher |
| PLASTIC | 1 | no_final_support | -0.10 | [-2.52, 2.23] | includes_zero |
| PLASTIC | 1 | confidence_only | 3.35 | [0.41, 6.25] | full_higher |
| PLASTIC | 2 | no_confidence | -0.71 | [-2.81, 1.32] | includes_zero |
| PLASTIC | 2 | no_ransac | -1.12 | [-2.98, 0.72] | includes_zero |
| PLASTIC | 2 | no_loo | 0.51 | [-0.50, 1.52] | includes_zero |
| PLASTIC | 2 | no_flip | 4.37 | [1.33, 7.44] | full_higher |
| PLASTIC | 2 | no_final_support | 1.02 | [-1.03, 3.09] | includes_zero |
| PLASTIC | 2 | confidence_only | 2.54 | [-0.21, 5.33] | includes_zero |
| WOOD | 1 | no_confidence | 0.83 | [-0.32, 2.10] | includes_zero |
| WOOD | 1 | no_ransac | -1.45 | [-3.42, 0.31] | includes_zero |
| WOOD | 1 | no_loo | 0.83 | [-0.41, 2.09] | includes_zero |
| WOOD | 1 | no_flip | -2.70 | [-6.36, 0.73] | includes_zero |
| WOOD | 1 | no_final_support | -0.73 | [-2.47, 0.94] | includes_zero |
| WOOD | 1 | confidence_only | -2.18 | [-5.66, 1.13] | includes_zero |
| WOOD | 2 | no_confidence | 2.39 | [0.93, 3.81] | full_higher |
| WOOD | 2 | no_ransac | -0.10 | [-1.55, 1.24] | includes_zero |
| WOOD | 2 | no_loo | 1.97 | [0.81, 3.20] | full_higher |
| WOOD | 2 | no_flip | 0.00 | [-3.74, 3.58] | includes_zero |
| WOOD | 2 | no_final_support | 0.62 | [-1.04, 2.18] | includes_zero |
| WOOD | 2 | confidence_only | -0.83 | [-4.17, 2.19] | includes_zero |
| GREEN | 1 | no_confidence | 0.32 | [-1.59, 2.22] | includes_zero |
| GREEN | 1 | no_ransac | 0.11 | [-1.48, 1.60] | includes_zero |
| GREEN | 1 | no_loo | -0.53 | [-1.38, 0.21] | includes_zero |
| GREEN | 1 | no_flip | 1.17 | [-1.60, 3.89] | includes_zero |
| GREEN | 1 | no_final_support | -0.64 | [-2.34, 0.96] | includes_zero |
| GREEN | 1 | confidence_only | 2.86 | [0.53, 5.36] | full_higher |
| GREEN | 2 | no_confidence | 1.17 | [-0.32, 2.54] | includes_zero |
| GREEN | 2 | no_ransac | 0.95 | [-0.21, 2.13] | includes_zero |
| GREEN | 2 | no_loo | -0.42 | [-1.80, 0.74] | includes_zero |
| GREEN | 2 | no_flip | 1.91 | [0.11, 3.80] | full_higher |
| GREEN | 2 | no_final_support | 0.74 | [-0.43, 1.92] | includes_zero |
| GREEN | 2 | confidence_only | 0.85 | [-0.75, 2.55] | includes_zero |

양수는 full의 지표가 더 높음을 뜻한다. CI가0을 포함하면 includes_zero다. 동일 종류의 모델·라운드·지표·수동 패널에 같은 프레임 가중치를 공유한다. WOOD125는 seed20261026으로 새10,000개 multinomial 가중치를 생성했다. 모집단이 변하지 않은 PLASTIC/GREEN의 기존 MAIN CI를 그대로 재사용해 난수소비순서 변화에 따른 불필요한 구간 변화도 피했다. 분모0 추출이 하나라도 있으면 CI 전체를 NA로 표시하며 추출을 버리거나 다시 뽑지 않았다. 학습 실패 모델은 NA이며0점 모델로 바꾸지 않았다.
WOOD125는4개 녹화 ID가 모두 알려져 있어 녹화4개를 복원추출하고 해당 녹화의 모든 프레임에 동일 가중치를 부여하는 보조 녹화 CI도 별도 제시했다. 5개 미만의 소수 군집이므로 프레임 CI를 주결과로 유지한다. PLASTIC 메타데이터 상충 및 GREEN 녹화 ID 누락으로 해당 녹화 CI는 NA다.
