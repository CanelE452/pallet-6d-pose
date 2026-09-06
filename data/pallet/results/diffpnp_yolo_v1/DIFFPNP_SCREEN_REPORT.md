# DiffPnP 학습 스크린 — lambda_dp 만 다르다

10 epochs · batch 32 · seed 42 · 162분

| 지표 | 대조 lambda=0 | 처치 lambda=1 | 상대변화 |
|---|---:|---:|---:|
| rotation_median_deg | 2.6139 | 3.1928 | +22.15% |
| translation_median_cm | 6.9523 | 8.5838 | +23.47% |
| iou3d_median | 0.6018 | 0.5573 | -7.40% |
| add_sym_auc | 0.4230 | 0.3830 | -9.44% |

판정 **REJECT** · 실질 NOT_IMPROVED · 개선 0/4

| 보조 | 대조 | 처치 |
|---|---:|---:|
| n | 319 | 319 |
| axis_accuracy | 0.7398119122257053 | 0.7147335423197492 |
| yaw_median_deg | 1.1999748340815017 | 1.4479925365089628 |

> PAPER_EVAL 319 는 반복 사용된 development set. 어떤 결과도 held-out/final/SOTA 로 부르지 않는다.
