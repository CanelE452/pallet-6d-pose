# self-training + DiffPnP (R5_PROPOSED 데이터셋, lambda_dp 만 다르다)

10 epochs · 900 optimizer update · lambda=0.1432 · 9분

| 지표 | 대조 λ=0 | 처치 λ>0 | 상대변화 |
|---|---:|---:|---:|
| rotation_median_deg | 2.5345 | 2.6755 | +5.56% |
| translation_median_cm | 8.8265 | 8.0458 | -8.84% |
| iou3d_median | 0.5868 | 0.5955 | +1.48% |
| add_sym_auc | 0.4001 | 0.4104 | +2.58% |

판정 **REJECT** · 실질 NOT_IMPROVED · 개선 3/4

> PAPER_EVAL 319 는 반복 사용된 development set. 어떤 결과도 held-out/final/SOTA 로 부르지 않는다.
