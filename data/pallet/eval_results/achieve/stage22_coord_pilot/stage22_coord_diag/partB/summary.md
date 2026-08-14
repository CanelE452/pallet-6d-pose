# STAGE22 PART B — crop-and-refine 2단 추론 (B2, 학습 X)

records=248  skip_pass1(det<6)=97  skip_pass2(no-bbox)=0
pass1=aspect-only(패딩X). refine=pred bbox+margin20% crop→aspect400→재추론→offset역변환.
GT=projected_cuboid[:8]. real=same-idx per-corner(corner01 검증). syn=mixed_v8 convention→order-free(Hungarian) full-8 만 신뢰.

## REAL same-frame pairs (delta<0 = refine 개선)
```
      metric    n  pass1_med  refine_med   delta  improve   worse
------------------------------------------------------------------
       front    8       7.45       69.95  +63.51        2       6
        rear   14      11.83       94.59  +78.53        1      12
  full8_hung    0  (no pairs)
      reproj    3       2.19        1.68   -0.51        1       1
```

## SYN (in-domain control; front/rear convention-confounded, full8_hung/reproj 신뢰)
```
      metric    n  pass1_med  refine_med   delta  improve   worse
------------------------------------------------------------------
       front   64     117.51      117.23   +5.84       23      38
        rear   56     102.29       83.97   +9.67       21      33
  full8_hung    6       7.14       60.71  +54.05        0       6
      reproj    0  (no pairs)
```

## REAL rear good/gross rate
```
rear good(<10px)  pass1=0.175  refine=0.071
rear gross(>20px) pass1=0.325  refine=0.929
```

## REAL elevation-bin (rear paired)
```
  elev    n  rear_p1  rear_ref   delta  imp  wrs
    <3   36  (no rear pairs)
   3-8   13    12.24     98.96   +78.6    1   12
    8+    1      6.3       5.9    -0.4    0    0
```

## 해석 가이드 (task 지정)
- REAL rear delta=+78.5px (악화) → 해상도 기각 단정 금지. 스케일 분포 이탈 confound 가능(near-large 전례) — CONFOUNDED 표기.

★ 페어 N 명시: real rear N=14, syn full8 N=6.