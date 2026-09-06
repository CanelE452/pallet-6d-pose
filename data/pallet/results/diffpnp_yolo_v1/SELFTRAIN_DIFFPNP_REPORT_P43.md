# self-training + DiffPnP (R5_PROPOSED_P43 데이터셋, lambda_dp 만 다르다)

10 epochs · 900 optimizer update · lambda=0.1432 · 8분

| 지표 | 대조 λ=0 | 처치 λ>0 | 상대변화 |
|---|---:|---:|---:|
| rotation_median_deg | 2.5999 | 2.4227 | -6.82% |
| translation_median_cm | 7.6472 | 7.8778 | +3.02% |
| iou3d_median | 0.5963 | 0.5898 | -1.08% |
| add_sym_auc | 0.4186 | 0.4160 | -0.62% |

판정 **REJECT** · 실질 NOT_IMPROVED · 개선 1/4

> PAPER_EVAL 319 는 반복 사용된 development set. 어떤 결과도 held-out/final/SOTA 로 부르지 않는다.
