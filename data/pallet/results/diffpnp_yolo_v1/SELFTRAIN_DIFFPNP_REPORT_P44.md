# self-training + DiffPnP (R5_PROPOSED_P44 데이터셋, lambda_dp 만 다르다)

10 epochs · 900 optimizer update · lambda=0.1432 · 8분

| 지표 | 대조 λ=0 | 처치 λ>0 | 상대변화 |
|---|---:|---:|---:|
| rotation_median_deg | 2.7107 | 2.6842 | -0.98% |
| translation_median_cm | 8.3098 | 8.0983 | -2.54% |
| iou3d_median | 0.5913 | 0.5839 | -1.25% |
| add_sym_auc | 0.4161 | 0.4028 | -3.20% |

판정 **REJECT** · 실질 NOT_IMPROVED · 개선 2/4

> PAPER_EVAL 319 는 반복 사용된 development set. 어떤 결과도 held-out/final/SOTA 로 부르지 않는다.
