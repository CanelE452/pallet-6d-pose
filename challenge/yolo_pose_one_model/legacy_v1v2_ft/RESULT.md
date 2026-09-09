LEGACY_V1V2_P0_10K FT on G38 — 사전등록 gate 판정

metric                        G38    LV1V2FT   변화
────────────────────────────────────────────────────────────────────────
correct_box_recall         0.8500     0.9857   +0.1357 pp
corner_median (px)        11.5483    13.6407   -18.1% 개선
corner_p90 (px)           62.6473    58.8122   +6.1% 개선
night_p90 (px)            78.1645    93.5196   -19.6% 개선
night_margin               0.0362     0.7127   +0.6764
detection_recall           0.9857     1.0000   
night top1 (frames)            15         26   +11
night any  (frames)            23         27   +4

hits 3/3 필요 — {'all_cbox_+2pp': True, 'all_median_-8%': False, 'all_p90_-10%': False, 'night_top1_+3f': True, 'night_p90_-15%': False, 'night_margin_+0.10': True}
guards {'all_cbox': False, 'night_any_cbox': False, 'all_p90': False}  tripped=[]

VERDICT = GAIN

★ CAUSAL ABLATION 아님 — FT 는 추가 step 을 동반한다(교락). BEST-PERFORMANCE CANDIDATE 비교.
★ n(real)=140  n(night)=28  seed 1개.
