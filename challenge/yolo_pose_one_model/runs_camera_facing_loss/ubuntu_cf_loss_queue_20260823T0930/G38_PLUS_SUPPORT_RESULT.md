# G38 + SUPPORT — target-free BEST-PERFORMANCE CANDIDATE

```
model             cbox   median      p90  gross20
--------------------------------------------------
A42              0.422    53.46    87.91    0.917
C43              0.797    14.28    91.98    0.401
G38              0.852    12.03    66.66    0.314
COMBO            0.836    11.18    75.56    0.307
OLD              0.969     9.68    40.99    0.222
FT               0.984     6.47    25.40    0.135
```

## NIGHT
```
model        any    top1      med      p90  cand/fr  wrong%   margin
--------------------------------------------------------------
G38        0.821   0.536    16.38    78.16     7.93   0.857   +0.036
COMBO      0.714   0.500    13.44   105.65     4.54   0.857   +0.115
OLD        0.964   0.929     9.84    22.76     2.25   0.286   +0.879
FT         0.964   0.964     6.31    21.37     1.32   0.071   +0.962
```

## GAP CLOSURE (G38 → OLD 구간에서 COMBO 위치)
```
R_cbox             -13.3%
R_med              +36.2%
R_p90              -34.7%
R_night_cbox        -9.1%
R_night_p90        -49.6%
R_margin            +9.4%
```

hits 0/6 → {'all_cbox_+2pp': False, 'all_median_-8%': False, 'all_p90_-10%': False, 'night_top1_+3f': False, 'night_p90_-15%': False, 'night_margin_+0.10': False}
guards {'all_cbox': True, 'night_any_cbox': False, 'all_p90': True}

**TARGET_FREE_COMBO_SIGNAL = HARM**
**CURRENT_TARGET_FREE_BEST = G38**

RENDER_TRACK = CLOSED   RENDER_RESUME = FALSE
epochs 60/60 · batches/ep 1248 · checkpoint last.pt

★ CAUSAL ABLATION 아님 — exposure/steps +5.1% 동반. OLD/FT 는 REFERENCE/UPPER
  DIAGNOSTIC 이며 같은 training data 조건이 아니다. NIGHT n=28, seed 1.