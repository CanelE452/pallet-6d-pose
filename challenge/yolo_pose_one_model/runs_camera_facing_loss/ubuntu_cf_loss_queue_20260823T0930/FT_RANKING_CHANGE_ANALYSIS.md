# FT RANKING CHANGE ANALYSIS

NIGHT 28 · DAY control 28 (seed42) · IoU>=0.5 · conf=0.001 · pad=100

## margin = best_correct_conf − best_wrong_conf
```
model     n      p10      p25   median      p75    pos%
--------------------------------------------------
A42       2  -0.3582  -0.3016  -0.2073  -0.1129    0.0%
C42      20  -0.0625  -0.0175   0.3509   0.6093   65.0%
C43      27  -0.3215  -0.0237   0.4312   0.7278   66.7%
E42      22  -0.1083  -0.0401   0.0169   0.2391   59.1%
FT       27   0.9036   0.9499   0.9616   0.9756  100.0%
```

## DAY control margin
```
model     n   median    pos%
------------------------------
A42      17   0.7780   94.1%
C42      25   0.7506   88.0%
C43      26   0.7669   84.6%
E42      19   0.5488   84.2%
FT       28   0.9741  100.0%
```

## C43 → FT transition (night): {'T0': 10, 'T1': 17, 'T3': 1}
  T0 = C43 wrong → FT correct (핵심), T1 둘 다 correct, T2 둘 다 wrong, T3 역행

## conf 변화 (같은 프레임 paired, C43 → FT)
```
correct_candidate_conf     n 26  C43 0.4822 → FT 0.9604  Δmed +0.4666  FT우세 100%
wrong_candidate_conf       n  2  C43 0.0205 → FT 0.0071  Δmed -0.0135  FT우세 0%
top1_conf                  n 28  C43 0.5044 → FT 0.9604  Δmed +0.4273  FT우세 96%
```

## candidate separation (frame-level P(correct conf > wrong conf))
```
model     NIGHT      DAY
--------------------------
A42       0.000    0.917
C42       0.562    0.769
C43       0.625    0.733
E42       0.526    0.750
FT        1.000   n/a
```

**ROUTING_CASE = R2   NEXT_ONE_ACTION = PSEUDO_POSITIVE_SCREEN**

self-training 위험: NIGHT correct-but-not-top1 9/28

★ FT 는 real positive + negative + synthetic 동시 사용 — 인과 확정 금지. observational only.