LEGACY_V1V2_P0_10K FT on G38 (seed 43) — 사전등록 gate 판정

metric                        G38  LV1V2FT43   변화
────────────────────────────────────────────────────────────────────────
correct_box_recall         0.8500     0.9929   +14.29 pp
corner_median (px)        11.5483    13.3579   15.7% 악화
corner_p90 (px)           62.6473    58.7606   6.2% 개선
night_p90 (px)            78.1645   102.0869   30.6% 악화
night_margin               0.0362     0.7050   +0.6688
detection_recall           0.9857     1.0000   
night top1 (frames)            15         27   +12
night any  (frames)            23         27   +4

hits 3/3 필요 — {'all_cbox_+2pp': True, 'all_median_-8%': False, 'all_p90_-10%': False, 'night_top1_+3f': True, 'night_p90_-15%': False, 'night_margin_+0.10': True}
guards {'all_cbox': False, 'night_any_cbox': False, 'all_p90': False}  tripped=[]

VERDICT = GAIN

★ CAUSAL ABLATION 아님 — FT 는 추가 step 을 동반한다(교락). BEST-PERFORMANCE CANDIDATE 비교.
★ n(real)=140  n(night)=28  seed 1개.
