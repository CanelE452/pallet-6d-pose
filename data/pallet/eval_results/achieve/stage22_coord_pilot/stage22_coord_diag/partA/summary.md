# STAGE22 PART A — "윗면 보임" 앙각 실측 (real 98, 학습·재추론 X)

real enriched N = 98. elevation = GT pose 시야각(edge-on~0, top-down~90).
band_px = |mean_y(kp4,5)-mean_y(kp0,1)| (top-front↔top-rear 화면 수직간격, 수평 팔레트 근사).

## elevation bin × 측정
```
   elev    n  band_med  band_p90  front_med  rear_med  rear_spk      elev_span
--------------------------------------------------------------------------------
     <3   34       1.0       5.7      11.91     19.23     0.765     [-7.7,2.9]
    3-8   58      17.2      24.1      11.05     18.45     0.724      [3.3,7.8]
   8-15    1      37.5      37.5      13.36     53.08       1.0      [8.3,8.3]
  15-25    0      None      None       None      None      None              -
    25+    5     301.8    309.28      14.09      9.77       0.4    [29.4,35.3]
```

## 상관 (Spearman)
- elev↔band = 0.895 (앙각↑ → 윗면 밴드↑ 기대)
- band↔rear_err = 0.124 (밴드 두꺼울수록 rear 정확?)
- elev↔rear_err = 0.075

## 기운 프레임 caveat: tilt>15deg = 1/98 (band 수평근사 부정확 프레임)

## 판정: "윗면 보임 인상" vs 실측
- real 저앙각(<8도) N=92, band_med=12.8px, rear_med=19.0px
- real 고앙각(≥8도) N=6, band_med=292.1px, rear_med=13.9px