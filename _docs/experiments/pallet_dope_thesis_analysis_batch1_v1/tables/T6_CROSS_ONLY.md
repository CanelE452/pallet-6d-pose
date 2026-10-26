[확인] 두 기존 SCORES의 전체72개 모델×종류 행 중 실제 교차 종류36행만 표시한다. 공통18행과 동일종류18행은 T6_CROSS_TYPE.csv에 문맥으로 보존했다. 새 추론0회.

| 소스 실험 | 학습 종류 | 평가 종류 | 모델 | PCK5 % | PCK10 % | PCK20 % | coverage % |
|---|---|---|---|---|---|---|---|
| LOO_flip | GREEN | PLASTIC | GREEN_R1 | 18.98 | 37.97 | 57.06 | 82.94 |
| LOO_flip | GREEN | WOOD | GREEN_R1 | 23.12 | 48.84 | 73.12 | 82.08 |
| LOO_flip | GREEN | PLASTIC | GREEN_R2 | 20.30 | 42.13 | 60.61 | 91.07 |
| LOO_flip | GREEN | WOOD | GREEN_R2 | 19.65 | 49.71 | 72.25 | 82.37 |
| LOO_flip | PLASTIC | WOOD | PLASTIC_R1 | 22.83 | 51.16 | 73.99 | 83.53 |
| LOO_flip | PLASTIC | GREEN | PLASTIC_R1 | 42.95 | 69.03 | 83.35 | 98.20 |
| LOO_flip | PLASTIC | WOOD | PLASTIC_R2 | 19.08 | 47.98 | 71.97 | 82.95 |
| LOO_flip | PLASTIC | GREEN | PLASTIC_R2 | 40.93 | 67.76 | 84.41 | 99.15 |
| LOO_flip | WOOD | PLASTIC | WOOD_R1 | 16.24 | 35.74 | 52.18 | 76.55 |
| LOO_flip | WOOD | GREEN | WOOD_R1 | 30.33 | 64.48 | 83.14 | 98.30 |
| LOO_flip | WOOD | PLASTIC | WOOD_R2 | 20.41 | 39.49 | 58.68 | 86.80 |
| LOO_flip | WOOD | GREEN | WOOD_R2 | 36.80 | 66.70 | 82.93 | 98.52 |
| LOO_only | GREEN | PLASTIC | GREEN_LOO_flip_R1 | 18.98 | 37.97 | 57.06 | 82.94 |
| LOO_only | GREEN | WOOD | GREEN_LOO_flip_R1 | 23.12 | 48.84 | 73.12 | 82.08 |
| LOO_only | GREEN | PLASTIC | GREEN_LOO_flip_R2 | 20.30 | 42.13 | 60.61 | 91.07 |
| LOO_only | GREEN | WOOD | GREEN_LOO_flip_R2 | 19.65 | 49.71 | 72.25 | 82.37 |
| LOO_only | PLASTIC | WOOD | PLASTIC_LOO_flip_R1 | 22.83 | 51.16 | 73.99 | 83.53 |
| LOO_only | PLASTIC | GREEN | PLASTIC_LOO_flip_R1 | 42.95 | 69.03 | 83.35 | 98.20 |
| LOO_only | PLASTIC | WOOD | PLASTIC_LOO_flip_R2 | 19.08 | 47.98 | 71.97 | 82.95 |
| LOO_only | PLASTIC | GREEN | PLASTIC_LOO_flip_R2 | 40.93 | 67.76 | 84.41 | 99.15 |
| LOO_only | WOOD | PLASTIC | WOOD_LOO_flip_R1 | 16.24 | 35.74 | 52.18 | 76.55 |
| LOO_only | WOOD | GREEN | WOOD_LOO_flip_R1 | 30.33 | 64.48 | 83.14 | 98.30 |
| LOO_only | WOOD | PLASTIC | WOOD_LOO_flip_R2 | 20.41 | 39.49 | 58.68 | 86.80 |
| LOO_only | WOOD | GREEN | WOOD_LOO_flip_R2 | 36.80 | 66.70 | 82.93 | 98.52 |
| LOO_only | GREEN | PLASTIC | GREEN_LOO_only_R1 | 17.97 | 38.58 | 56.35 | 84.97 |
| LOO_only | GREEN | WOOD | GREEN_LOO_only_R1 | 23.12 | 50.00 | 71.97 | 80.06 |
| LOO_only | GREEN | PLASTIC | GREEN_LOO_only_R2 | 19.09 | 40.91 | 59.90 | 92.08 |
| LOO_only | GREEN | WOOD | GREEN_LOO_only_R2 | 19.94 | 47.98 | 71.97 | 82.37 |
| LOO_only | PLASTIC | WOOD | PLASTIC_LOO_only_R1 | 21.10 | 53.18 | 72.54 | 82.37 |
| LOO_only | PLASTIC | GREEN | PLASTIC_LOO_only_R1 | 35.95 | 63.63 | 80.38 | 97.99 |
| LOO_only | PLASTIC | WOOD | PLASTIC_LOO_only_R2 | 19.36 | 51.16 | 73.12 | 83.82 |
| LOO_only | PLASTIC | GREEN | PLASTIC_LOO_only_R2 | 36.80 | 64.26 | 80.91 | 98.83 |
| LOO_only | WOOD | PLASTIC | WOOD_LOO_only_R1 | 16.45 | 36.35 | 53.30 | 81.73 |
| LOO_only | WOOD | GREEN | WOOD_LOO_only_R1 | 26.72 | 64.16 | 82.40 | 97.03 |
| LOO_only | WOOD | PLASTIC | WOOD_LOO_only_R2 | 20.51 | 40.20 | 58.58 | 87.21 |
| LOO_only | WOOD | GREEN | WOOD_LOO_only_R2 | 33.62 | 65.75 | 83.78 | 98.09 |
