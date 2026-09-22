# 종류별 Clean 실사 + 합성 replay 결과

플라스틱10장과 목재9장을 섞지 않고 보정기2개를 각각 합성전용PRIOR1에서 시작했다. seed1/300step, 실사8+합성8, BN통계고정. 실사 감독은 해당 종류만이며 합성 replay 순서는 혼합 실험과 동일하게 유지했다.

평가목록도 혼합학습과 동일: 플라스틱184장·목재116장. 학습이미지 중복0, 같은 촬영 세션은 사용자 요청으로 허용했다. 새환경 일반화 주장이 아니다.

모델당300step이므로 종류별 모델은 해당 종류의 실사 노출이 혼합모델보다 많다. 따라서 재료분리만의 순수 인과효과로 해석하지 않는다.

## plastic

| 난도 | 모델 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---|---:|---:|---:|---:|---:|
| ALL | R0 | 59.53 | 77.53 | 7.63 | 43.13 | 322 |
| ALL | N2_DIM_ONLY | 66.09 | 80.11 | 6.36 | 41.80 | 285 |
| ALL | N3_DIM_SYM | 65.81 | 80.11 | 6.34 | 41.73 | 285 |
| ALL | PRIOR1 | 64.27 | 79.27 | 6.22 | 43.48 | 297 |
| ALL | MIXED19_REPLAY | 67.69 | 80.32 | 5.70 | 44.61 | 282 |
| ALL | TYPE_REPLAY | 67.55 | 80.11 | 5.35 | 46.24 | 285 |
| ALL | TYPE_REPLAY_CAP8 | 66.92 | 80.25 | 5.46 | 43.10 | 283 |
| CLEAN | R0 | 80.81 | 96.80 | 4.43 | 13.65 | 15 |
| CLEAN | N2_DIM_ONLY | 89.13 | 99.36 | 3.81 | 10.16 | 3 |
| CLEAN | N3_DIM_SYM | 89.34 | 99.36 | 3.79 | 10.06 | 3 |
| CLEAN | PRIOR1 | 85.50 | 98.51 | 4.13 | 12.79 | 7 |
| CLEAN | MIXED19_REPLAY | 90.62 | 98.51 | 3.29 | 9.30 | 7 |
| CLEAN | TYPE_REPLAY | 91.26 | 99.57 | 3.28 | 9.56 | 2 |
| CLEAN | TYPE_REPLAY_CAP8 | 89.55 | 99.79 | 3.20 | 10.08 | 1 |
| MODERATE_OCCLUSION | R0 | 62.43 | 79.59 | 7.83 | 94.38 | 69 |
| MODERATE_OCCLUSION | N2_DIM_ONLY | 68.05 | 80.77 | 6.44 | 95.41 | 65 |
| MODERATE_OCCLUSION | N3_DIM_SYM | 67.75 | 80.77 | 6.37 | 95.37 | 65 |
| MODERATE_OCCLUSION | PRIOR1 | 67.46 | 80.18 | 6.78 | 96.12 | 67 |
| MODERATE_OCCLUSION | MIXED19_REPLAY | 69.53 | 80.18 | 5.93 | 95.29 | 67 |
| MODERATE_OCCLUSION | TYPE_REPLAY | 69.82 | 78.70 | 5.72 | 95.53 | 72 |
| MODERATE_OCCLUSION | TYPE_REPLAY_CAP8 | 70.12 | 79.88 | 5.87 | 95.53 | 68 |
| SEVERE_OCCLUSION | R0 | 42.01 | 61.98 | 11.08 | 52.33 | 238 |
| SEVERE_OCCLUSION | N2_DIM_ONLY | 47.76 | 65.34 | 9.63 | 51.69 | 217 |
| SEVERE_OCCLUSION | N3_DIM_SYM | 47.12 | 65.34 | 9.61 | 52.03 | 217 |
| SEVERE_OCCLUSION | PRIOR1 | 46.65 | 64.38 | 9.65 | 53.27 | 223 |
| SEVERE_OCCLUSION | MIXED19_REPLAY | 49.52 | 66.77 | 8.99 | 54.11 | 208 |
| SEVERE_OCCLUSION | TYPE_REPLAY | 48.56 | 66.29 | 9.27 | 53.54 | 211 |
| SEVERE_OCCLUSION | TYPE_REPLAY_CAP8 | 48.24 | 65.81 | 9.26 | 52.64 | 214 |

## wood

| 난도 | 모델 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---|---:|---:|---:|---:|---:|
| ALL | R0 | 67.40 | 83.04 | 6.30 | 43.67 | 155 |
| ALL | N2_DIM_ONLY | 71.44 | 85.12 | 5.38 | 42.14 | 136 |
| ALL | N3_DIM_SYM | 71.55 | 85.34 | 5.37 | 42.38 | 134 |
| ALL | PRIOR1 | 73.41 | 85.56 | 4.89 | 40.44 | 132 |
| ALL | MIXED19_REPLAY | 67.83 | 83.04 | 5.46 | 56.25 | 155 |
| ALL | TYPE_REPLAY | 68.93 | 83.59 | 5.21 | 57.30 | 150 |
| ALL | TYPE_REPLAY_CAP8 | 69.37 | 83.92 | 5.22 | 47.65 | 147 |
| CLEAN | R0 | 67.81 | 86.06 | 6.26 | 29.76 | 81 |
| CLEAN | N2_DIM_ONLY | 73.84 | 88.30 | 5.42 | 26.11 | 68 |
| CLEAN | N3_DIM_SYM | 74.18 | 88.47 | 5.45 | 25.34 | 67 |
| CLEAN | PRIOR1 | 76.59 | 88.64 | 4.84 | 22.92 | 66 |
| CLEAN | MIXED19_REPLAY | 70.05 | 85.54 | 5.48 | 30.15 | 84 |
| CLEAN | TYPE_REPLAY | 71.43 | 86.40 | 5.16 | 26.88 | 79 |
| CLEAN | TYPE_REPLAY_CAP8 | 71.94 | 86.92 | 5.20 | 25.39 | 76 |
| MODERATE_OCCLUSION | R0 | 66.67 | 77.78 | 6.30 | 113.98 | 74 |
| MODERATE_OCCLUSION | N2_DIM_ONLY | 67.27 | 79.58 | 5.27 | 112.43 | 68 |
| MODERATE_OCCLUSION | N3_DIM_SYM | 66.97 | 79.88 | 5.29 | 112.33 | 67 |
| MODERATE_OCCLUSION | PRIOR1 | 67.87 | 80.18 | 5.03 | 106.63 | 66 |
| MODERATE_OCCLUSION | MIXED19_REPLAY | 63.96 | 78.68 | 5.41 | 108.05 | 71 |
| MODERATE_OCCLUSION | TYPE_REPLAY | 64.56 | 78.68 | 5.56 | 115.45 | 71 |
| MODERATE_OCCLUSION | TYPE_REPLAY_CAP8 | 64.86 | 78.68 | 5.69 | 118.18 | 71 |
| SEVERE_OCCLUSION | R0 | — | — | — | — | 0 |
| SEVERE_OCCLUSION | N2_DIM_ONLY | — | — | — | — | 0 |
| SEVERE_OCCLUSION | N3_DIM_SYM | — | — | — | — | 0 |
| SEVERE_OCCLUSION | PRIOR1 | — | — | — | — | 0 |
| SEVERE_OCCLUSION | MIXED19_REPLAY | — | — | — | — | 0 |
| SEVERE_OCCLUSION | TYPE_REPLAY | — | — | — | — | 0 |
| SEVERE_OCCLUSION | TYPE_REPLAY_CAP8 | — | — | — | — | 0 |

cap8은 사전등록 보조출력이다. 모델을 평가 결과로 자동 선택·교체하지 않았다. 기존 모델/GT/319장 수치 보존. 이번 작업은 수동지도 보정기 적응이며 self-training이 아니다.
