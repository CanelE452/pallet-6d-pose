# PAPER_S2 Phase 5 — REAL eval (final transfer judgement)

Q: DiffPnP3D (Stage B λ0.005) rear/low-angle improvement TRANSFER to real?

## Protocol / parity
```
eval-parity (each model = its own training preprocess; downstream metric IDENTICAL):
  paper_s2_stageA/B : squash 640x480->400x400 aniso, belief(50)->orig x(W/50,H/50)
  paper_s1 / base   : aspect-preserving no-pad (PAD=0, official A_nopad)
metric: order-free Hungarian corner + split_metrics front/back + solve_pose
  (order-free W/D, dims=(1.1, 1.3, 0.11) ~= measured 1.1x1.3x0.12) + honest full-8 reproj.
good%<10px gross%>20px per-corner. det%=n_det>=6. elevation=GT pose_transform.
```

## Sets
```
filterval  N=123 (outside44+night43+manual36) = LOW-ANGLE (elev -7.7~10.4;
                  46 frames <3deg, 67 in 3-8deg) -> PRIMARY rear/low-angle test.
handannot17 N=17 = HIGH-ANGLE (25+deg 13/17). memory: NOT real-representative;
                  qualitative only, NO rear claim.
final-test sessions SEALED (split-lock).
```

## handannot17
elev dist(deg): <3:0, 3-8:0, 8-15:3, 15+:14

### overall (all models)
```
model                n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
---------------------------------------------------------------------------------------
paper_s2_stageB     17  24.0    5.2    8.2     6.2    11.9  35.0     10.3   70.4    7.4
paper_s1            17  24.0    6.0    9.8     7.1    14.7  24.0     12.6   73.1    7.7
paper_base_v2       17  24.0    5.8    9.3     7.6    15.6  24.0     28.0   76.9    3.8
paper_s2_stageA     17  18.0    7.0    6.0     5.9    10.8  29.0      7.6   90.5    0.0
```

### V=8 (full-view) vs V<8 (truncated)
```
model                n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
---------------------------------------------------------------------------------------
paper_s2_stageB V8   4  75.0    7.2    7.9     7.2    11.6 100.0      8.8   71.4    0.0
paper_s2_stageB V<8  13   8.0    2.9   37.9     5.2    70.8  15.0     24.9   66.7   33.3
paper_s1 V8          4  75.0    7.0   11.1     7.6    17.1  75.0     15.9   70.0   10.0
paper_s1 V<8        13   8.0    4.4    8.2     6.4    10.9   8.0      9.3   83.3    0.0
paper_base_v2 V8     4  75.0    6.6    9.2     7.7    17.4  75.0     19.9   80.0    5.0
paper_base_v2 V<8   13   8.0    4.3    9.3     7.5    13.9   8.0     87.9   66.7    0.0
paper_s2_stageA V8   4  75.0    7.0    6.0     5.9    10.8  75.0      6.0   90.5    0.0
paper_s2_stageA V<8  13   0.0   None   None    None    None  15.0     20.5   None   None
```

### PAIRED: paper_s2_stageB vs paper_s1 (Delta = challenger - baseline)
```
### handannot17 overall
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct               24.0           24.0          +0
front_med              6.0            5.2  -0.8 +good
rear_med               9.8            8.2  -1.6 +good
corner_med             7.1            6.2  -0.9 +good
honest8_med           12.6           10.3  -2.3 +good
good_pct              73.1           70.4   -2.7 -bad
gross_pct              7.7            7.4  -0.3 +good
pnp_pct               24.0           35.0   +11 +good

### handannot17 V=8
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct               75.0           75.0          +0
front_med              7.0            7.2   +0.2 -bad
rear_med              11.1            7.9  -3.2 +good
corner_med             7.6            7.2  -0.4 +good
honest8_med           15.9            8.8  -7.1 +good
good_pct              70.0           71.4  +1.4 +good
gross_pct             10.0            0.0   -10 +good
pnp_pct               75.0          100.0   +25 +good
```

## filterval
elev dist(deg): <3:46, 3-8:67, 8-15:10, 15+:0

### overall (all models)
```
model                n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
---------------------------------------------------------------------------------------
paper_s2_stageB    123  76.0    8.8   19.0    13.5    38.7  76.0     17.9   41.9   33.7
paper_s1           123  70.0    9.2   20.5    12.5    38.4  72.0     17.7   40.8   34.8
paper_base_v2      123  68.0   16.4   34.7    27.5    54.3  71.0     31.7   28.7   48.1
paper_s2_stageA    123  59.0    8.6   20.3    12.7    37.2  60.0     20.7   42.3   34.8
```

### V=8 (full-view) vs V<8 (truncated)
```
model                n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
---------------------------------------------------------------------------------------
paper_s2_stageB V8 106  84.0    8.7   18.8    13.5    35.1  84.0     17.4   42.2   32.6
paper_s2_stageB V<8  17  29.0   63.4   40.9    24.8   148.0  29.0     55.4   36.8   52.6
paper_s1 V8        106  79.0    9.3   20.2    12.7    37.3  81.0     17.6   40.5   34.8
paper_s1 V<8        17  12.0    5.8   40.7     8.2    73.1  18.0     51.6   58.3   33.3
paper_base_v2 V8   106  77.0   15.3   34.2    27.2    53.9  79.0     28.4   28.6   47.7
paper_base_v2 V<8   17  12.0   54.5   64.6    60.0   115.6  18.0     70.2   31.2   62.5
paper_s2_stageA V8 106  66.0    8.5   20.2    12.6    36.5  68.0     20.2   42.9   33.9
paper_s2_stageA V<8  17  12.0   73.4   28.1    28.1   146.4  12.0     55.2   25.0   62.5
```

### PAIRED: paper_s2_stageB vs paper_s1 (Delta = challenger - baseline)
```
### filterval overall
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct               70.0           76.0    +6 +good
front_med              9.2            8.8  -0.4 +good
rear_med              20.5           19.0  -1.5 +good
corner_med            12.5           13.5     +1 -bad
honest8_med           17.7           17.9   +0.2 -bad
good_pct              40.8           41.9  +1.1 +good
gross_pct             34.8           33.7  -1.1 +good
pnp_pct               72.0           76.0    +4 +good

### filterval V=8
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct               79.0           84.0    +5 +good
front_med              9.3            8.7  -0.6 +good
rear_med              20.2           18.8  -1.4 +good
corner_med            12.7           13.5   +0.8 -bad
honest8_med           17.6           17.4  -0.2 +good
good_pct              40.5           42.2  +1.7 +good
gross_pct             34.8           32.6  -2.2 +good
pnp_pct               81.0           84.0    +3 +good
```

### elevation bins — PAIRED (challenger vs baseline)
```
### elev <3 deg (n=46)
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct               85.0           98.0   +13 +good
front_med              9.3            8.7  -0.6 +good
rear_med              16.7           15.0  -1.7 +good
corner_med            12.6           10.6    -2 +good
honest8_med           16.3           15.8  -0.5 +good
good_pct              41.4           46.8  +5.4 +good
gross_pct             30.0           27.2  -2.8 +good
pnp_pct               85.0           98.0   +13 +good

### elev 3-8 deg (n=67)
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct               70.0           72.0    +2 +good
front_med              8.6            9.0   +0.4 -bad
rear_med              27.5           23.7  -3.8 +good
corner_med            12.1           17.6   +5.5 -bad
honest8_med           22.3           26.0   +3.7 -bad
good_pct              40.3           37.4   -2.9 -bad
gross_pct             38.8           39.9   +1.1 -bad
pnp_pct               73.0           72.0     -1 -bad

### elev 8-15 deg (n=10)
metric            paper_s1paper_s2_stageB      d(B-A)
-----------------------------------------------------
det_pct                0.0           10.0   +10 +good
front_med              -              5.0            
rear_med               -             40.9            
corner_med             -             13.3            
honest8_med          104.8           54.0 -50.8 +good
good_pct               -             33.3            
gross_pct              -             33.3            
pnp_pct               10.0           10.0          +0

```

### elevation bins — rear_med (all models)
```
elev       paper_s2_stageB          paper_s1     paper_base_v2   paper_s2_stageA
<3                    15.0              16.7              32.1              14.3
3-8                   23.7              27.5              36.1              24.2
8-15                  40.9              None              None              None
15+                   None              None              None              None
```

## VERDICT — transfer to real (filterval PRIMARY)
```
  rear_med   : paper_s1=20.5 -> paper_s2_stageB=19.0  d=-1.5
  honest8    : paper_s1=17.7 -> paper_s2_stageB=17.9  d=0.2
  corner_med : paper_s1=12.5 -> paper_s2_stageB=13.5  d=1.0
  front_med  : paper_s1=9.2 -> paper_s2_stageB=8.8  d=-0.4
  det%       : paper_s1=70.0 -> paper_s2_stageB=76.0  d=6.0
  gross%     : paper_s1=34.8 -> paper_s2_stageB=33.7  d=-1.1

  low-angle rear elev <3 (n=46): 16.7 -> 15.0  d=-1.7
  low-angle rear elev 3-8 (n=67): 27.5 -> 23.7  d=-3.8
```

★ Small-sample + convention caveats: filterval N=123 (domain-mixed, low-angle),
  handannot17 N=17 high-angle (not real-representative). ckpt = synthetic-val best
  (NOT real-selected). Judgement only. rear improvement lower=better; a negative or
  ~zero d(rear) on low-angle bins = synth-only (STAGE16 precedent), not transfer.
