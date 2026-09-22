# 초기R0 자기 가림 → 보이는 점 보정 → 숨은 점 PnP 대체

사용자가 지정한 처리 순서로 적용했다. 초기 R0 PnP/가림 마스크는 모든 모델에서 동일하다. 보정기의 숨은 점 출력은 버린다. 비가림 보정점으로만 두 번째 PnP를 풀어 숨은 점만 채운다. 추가 학습0, 기존 예측 재사용.

±2도 경계는 미확정이므로 R0 유지. 재추정은 화면내부·confidence≥0.5 비가림6점 이상에서만 수행. 실패 시 비가림 보정은 유지하고 가림점은 R0 유지. 이 fallback은 보정기 전체 출력으로 되돌리는 것이 아니다.

DIRECT=기존 전체코너 보정 출력. VISIBLE_ONLY=숨은점/경계점은 R0, 확실한 비가림점만 보정. PIPELINE=VISIBLE_ONLY 후 숨은점 재투영.

## ALL300

| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---:|---:|---:|---:|---:|
| R0_DIRECT | 62.59 | 79.68 | 6.96 | 43.49 | 477 |
| R0_VISIBLE_ONLY | 62.59 | 79.68 | 6.96 | 43.49 | 477 |
| R0_PIPELINE | 64.42 | 79.97 | 6.62 | 42.98 | 470 |
| N2_DIM_ONLY_DIRECT | 68.17 | 82.06 | 5.85 | 42.19 | 421 |
| N2_DIM_ONLY_VISIBLE_ONLY | 67.53 | 80.95 | 5.90 | 42.87 | 447 |
| N2_DIM_ONLY_PIPELINE | 69.19 | 81.25 | 5.64 | 42.94 | 440 |
| N3_DIM_SYM_DIRECT | 68.04 | 82.15 | 5.91 | 41.87 | 419 |
| N3_DIM_SYM_VISIBLE_ONLY | 67.41 | 81.00 | 5.98 | 42.46 | 446 |
| N3_DIM_SYM_PIPELINE | 69.19 | 81.34 | 5.65 | 42.91 | 438 |
| TYPE_REPLAY_DIRECT | 68.09 | 81.47 | 5.28 | 47.75 | 435 |
| TYPE_REPLAY_VISIBLE_ONLY | 68.51 | 80.74 | 5.25 | 44.00 | 452 |
| TYPE_REPLAY_PIPELINE | 70.52 | 80.91 | 4.99 | 44.66 | 448 |

## plastic

| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---:|---:|---:|---:|---:|
| R0_DIRECT | 59.53 | 77.53 | 7.63 | 43.13 | 322 |
| R0_VISIBLE_ONLY | 59.53 | 77.53 | 7.63 | 43.13 | 322 |
| R0_PIPELINE | 61.48 | 77.88 | 7.08 | 43.13 | 317 |
| N2_DIM_ONLY_DIRECT | 66.09 | 80.11 | 6.36 | 41.80 | 285 |
| N2_DIM_ONLY_VISIBLE_ONLY | 64.90 | 78.44 | 6.53 | 42.45 | 309 |
| N2_DIM_ONLY_PIPELINE | 66.71 | 78.79 | 5.95 | 42.74 | 304 |
| N3_DIM_SYM_DIRECT | 65.81 | 80.11 | 6.34 | 41.73 | 285 |
| N3_DIM_SYM_VISIBLE_ONLY | 64.69 | 78.44 | 6.48 | 42.04 | 309 |
| N3_DIM_SYM_PIPELINE | 66.71 | 78.79 | 6.00 | 42.61 | 304 |
| TYPE_REPLAY_DIRECT | 67.55 | 80.11 | 5.35 | 46.24 | 285 |
| TYPE_REPLAY_VISIBLE_ONLY | 66.71 | 78.65 | 5.47 | 43.03 | 306 |
| TYPE_REPLAY_PIPELINE | 69.16 | 78.86 | 5.12 | 43.03 | 303 |

## wood

| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---:|---:|---:|---:|---:|
| R0_DIRECT | 67.40 | 83.04 | 6.30 | 43.67 | 155 |
| R0_VISIBLE_ONLY | 67.40 | 83.04 | 6.30 | 43.67 | 155 |
| R0_PIPELINE | 69.04 | 83.26 | 6.19 | 40.22 | 153 |
| N2_DIM_ONLY_DIRECT | 71.44 | 85.12 | 5.38 | 42.14 | 136 |
| N2_DIM_ONLY_VISIBLE_ONLY | 71.66 | 84.90 | 5.39 | 43.65 | 138 |
| N2_DIM_ONLY_PIPELINE | 73.09 | 85.12 | 5.22 | 42.14 | 136 |
| N3_DIM_SYM_DIRECT | 71.55 | 85.34 | 5.37 | 42.38 | 134 |
| N3_DIM_SYM_VISIBLE_ONLY | 71.66 | 85.01 | 5.41 | 43.86 | 137 |
| N3_DIM_SYM_PIPELINE | 73.09 | 85.34 | 5.23 | 42.38 | 134 |
| TYPE_REPLAY_DIRECT | 68.93 | 83.59 | 5.21 | 57.30 | 150 |
| TYPE_REPLAY_VISIBLE_ONLY | 71.33 | 84.03 | 5.01 | 48.33 | 146 |
| TYPE_REPLAY_PIPELINE | 72.65 | 84.14 | 4.75 | 50.78 | 145 |

## CLEAN

| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---:|---:|---:|---:|---:|
| R0_DIRECT | 73.62 | 90.86 | 5.39 | 18.42 | 96 |
| R0_VISIBLE_ONLY | 73.62 | 90.86 | 5.39 | 18.42 | 96 |
| R0_PIPELINE | 76.76 | 91.52 | 5.23 | 17.84 | 89 |
| N2_DIM_ONLY_DIRECT | 80.67 | 93.24 | 4.57 | 14.26 | 71 |
| N2_DIM_ONLY_VISIBLE_ONLY | 79.81 | 92.38 | 4.67 | 15.81 | 80 |
| N2_DIM_ONLY_PIPELINE | 82.86 | 93.05 | 4.41 | 13.71 | 73 |
| N3_DIM_SYM_DIRECT | 80.95 | 93.33 | 4.61 | 14.18 | 70 |
| N3_DIM_SYM_VISIBLE_ONLY | 79.90 | 92.38 | 4.68 | 16.33 | 80 |
| N3_DIM_SYM_PIPELINE | 83.05 | 93.14 | 4.48 | 13.59 | 72 |
| TYPE_REPLAY_DIRECT | 80.29 | 92.29 | 4.10 | 17.55 | 81 |
| TYPE_REPLAY_VISIBLE_ONLY | 80.67 | 91.90 | 4.11 | 16.82 | 85 |
| TYPE_REPLAY_PIPELINE | 83.90 | 92.38 | 3.71 | 15.32 | 80 |

## MODERATE_OCCLUSION

| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---:|---:|---:|---:|---:|
| R0_DIRECT | 64.53 | 78.69 | 7.08 | 99.88 | 143 |
| R0_VISIBLE_ONLY | 64.53 | 78.69 | 7.08 | 99.88 | 143 |
| R0_PIPELINE | 65.28 | 78.69 | 6.92 | 99.88 | 143 |
| N2_DIM_ONLY_DIRECT | 67.66 | 80.18 | 5.82 | 99.72 | 133 |
| N2_DIM_ONLY_VISIBLE_ONLY | 67.81 | 79.58 | 5.92 | 99.72 | 137 |
| N2_DIM_ONLY_PIPELINE | 68.26 | 79.58 | 5.60 | 99.72 | 137 |
| N3_DIM_SYM_DIRECT | 67.36 | 80.33 | 5.79 | 99.72 | 132 |
| N3_DIM_SYM_VISIBLE_ONLY | 67.81 | 79.73 | 5.94 | 99.72 | 136 |
| N3_DIM_SYM_PIPELINE | 68.41 | 79.73 | 5.64 | 99.72 | 136 |
| TYPE_REPLAY_DIRECT | 67.21 | 78.69 | 5.69 | 97.03 | 143 |
| TYPE_REPLAY_VISIBLE_ONLY | 67.96 | 79.14 | 5.50 | 97.03 | 140 |
| TYPE_REPLAY_PIPELINE | 68.85 | 79.14 | 5.38 | 97.03 | 140 |

## SEVERE_OCCLUSION

| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |
|---|---:|---:|---:|---:|---:|
| R0_DIRECT | 42.01 | 61.98 | 11.08 | 52.33 | 238 |
| R0_VISIBLE_ONLY | 42.01 | 61.98 | 11.08 | 52.33 | 238 |
| R0_PIPELINE | 42.81 | 61.98 | 10.79 | 51.44 | 238 |
| N2_DIM_ONLY_DIRECT | 47.76 | 65.34 | 9.63 | 51.69 | 217 |
| N2_DIM_ONLY_VISIBLE_ONLY | 46.65 | 63.26 | 9.91 | 52.50 | 230 |
| N2_DIM_ONLY_PIPELINE | 47.28 | 63.26 | 9.75 | 51.26 | 230 |
| N3_DIM_SYM_DIRECT | 47.12 | 65.34 | 9.61 | 52.03 | 217 |
| N3_DIM_SYM_VISIBLE_ONLY | 46.01 | 63.26 | 9.95 | 53.05 | 230 |
| N3_DIM_SYM_PIPELINE | 46.81 | 63.26 | 9.78 | 52.01 | 230 |
| TYPE_REPLAY_DIRECT | 48.56 | 66.29 | 9.27 | 53.54 | 211 |
| TYPE_REPLAY_VISIBLE_ONLY | 48.72 | 63.74 | 9.38 | 53.13 | 227 |
| TYPE_REPLAY_PIPELINE | 49.84 | 63.58 | 8.86 | 52.78 | 228 |

## 적용 수

- R0: {'reasons': {'fewer_than_6_reliable_visible_points': 62, 'hidden_only_reprojected': 229, 'no_confident_hidden_corner': 9}, 'applied': 229, 'hidden_counts': {1: 202, 2: 89, 0: 9}}
- N2_DIM_ONLY: {'reasons': {'fewer_than_6_reliable_visible_points': 62, 'hidden_only_reprojected': 227, 'no_confident_hidden_corner': 9, 'hidden_set_changed': 2}, 'applied': 227, 'hidden_counts': {1: 202, 2: 89, 0: 9}}
- N3_DIM_SYM: {'reasons': {'fewer_than_6_reliable_visible_points': 62, 'hidden_only_reprojected': 226, 'no_confident_hidden_corner': 9, 'hidden_set_changed': 3}, 'applied': 226, 'hidden_counts': {1: 202, 2: 89, 0: 9}}
- TYPE_REPLAY: {'reasons': {'hidden_only_reprojected': 219, 'fewer_than_6_reliable_visible_points': 58, 'hidden_set_changed': 14, 'no_confident_hidden_corner': 9}, 'applied': 219, 'hidden_counts': {1: 202, 2: 89, 0: 9}}

실제 팔레트의 구멍·외부가림을 추정하는 모델은 아니며 알려진 치수의 직육면체 자기 가림 근사다. K는 기존제공값이다. 결과는 기존 reference와의2D일치도이지 독립 물리 pose 정확도가 아니다. 매칭 실패는 기존 penalty 유지. 같은세션 적응·재사용 평가이며 모델 자동교체 없음.

로컬 단계별 이미지: outputs/pallet_visible_refine_hidden_pnp_v1/GALLERY.html
