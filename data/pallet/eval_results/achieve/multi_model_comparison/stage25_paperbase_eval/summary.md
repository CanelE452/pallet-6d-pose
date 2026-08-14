# STAGE25 — paper_base_v2 vs B2 vs challenge0123 (same sets, same metrics)

eval: order-free Hungarian corner + solve_pose(order-free W/D) + honest
full-8 reproj(GT projected_cuboid) + reflect-pad100 + per-frame K.
front/rear = matched-GT-idx bucket. good%/gross% = per-corner over all matched.

★ handannot17 N=17 극소·고앙각 편향(정성). filter-val = 주 신호(대표).
★ paper_base_v2 는 palletobj scan(v1/v2/v3/addon) 미학습 → cad/noapril 진짜 unseen.

## handannot17
V_geom: V2:1, V3:1, V4:3, V5:3, V6:2, V7:3, V8:4
elev bins(deg): -90~5:0, 5~10:0, 10~15:3, 15~25:1, 25~90:13

### overall
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2    17  47.0    9.1   17.0    13.2    23.7  47.0     18.9   45.5   21.8
B2               17  41.0   13.3   13.3    11.8    20.7  59.0     20.0   44.4   15.6
challenge0123    17  35.0   11.7   25.5    14.1    38.6  47.0     24.8   26.8   41.5

### handannot17 / domain=cad
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2    11  27.0   11.6   17.1    13.9    23.0  27.0     29.6   27.8   27.8
B2               11  27.0   14.1   10.2    11.8    20.7  45.0     13.2   35.0   10.0
challenge0123    11  27.0   13.0   18.9    14.8    26.2  36.0     22.9   15.8   47.4

### handannot17 / domain=noapril
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2     6  83.0    8.8   15.5    10.1    24.3  83.0     12.0   54.1   18.9
B2                6  67.0   11.2   13.5    11.3    24.6  83.0     25.1   52.0   20.0
challenge0123     6  50.0    8.5   29.2    11.1    39.9  67.0     25.8   36.4   36.4

### handannot17 / truncation split (V=8 full-view vs V<8 truncated)
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2 V8   4 100.0    7.9   14.5     9.2    19.7 100.0      9.0   63.3   13.3
paper_base_v2 V<8  13  31.0   13.3   25.0    18.7    35.6  31.0     28.6   24.0   32.0
B2 V8             4  75.0    9.0   10.2     9.3    20.3  75.0     23.8   57.1    9.5
B2 V<8           13  31.0   13.7   16.1    14.2    26.2  54.0     16.3   33.3   20.8
challenge0123 V8   4  75.0    8.5   29.2    11.1    39.9  75.0     17.7   39.1   34.8
challenge0123 V<8  13  23.0   12.4   21.8    14.6    37.3  38.0     29.1   11.1   50.0

## filterval
V_geom: V4:4, V5:1, V6:11, V7:1, V8:106
elev bins(deg): -90~5:61, 5~10:60, 10~15:2, 15~25:0, 25~90:0

### overall
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2   123  79.0   16.2   33.5    22.6    50.9  79.0     25.1   21.8   51.6
B2              123  75.0   12.1   16.8    13.5    26.1  75.0     15.6   29.1   29.1
challenge0123   123  72.0   12.4   18.7    14.8    32.2  73.0     18.4   27.0   31.6

### filterval / domain=manual
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2    36  72.0   16.6   22.4    20.1    41.6  72.0     21.9   19.2   49.5
B2               36  67.0   14.0   13.3    13.5    28.7  67.0     15.7   18.3   26.7
challenge0123    36  81.0   13.1   16.1    14.2    32.0  81.0     17.0   24.6   24.1

### filterval / domain=night
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2    43  74.0   14.1   35.5    23.4    56.7  74.0     32.5   18.4   54.0
B2               43  67.0    9.6   16.6    13.0    21.8  67.0     13.7   33.2   23.6
challenge0123    43  58.0   12.4   22.3    18.5    32.2  58.0     22.5   23.1   44.1

### filterval / domain=outside
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2    44  89.0   17.1   39.3    30.9    81.4  89.0     42.0   26.3   51.0
B2               44  89.0   10.7   19.0    14.2    30.7  89.0     17.6   32.9   34.5
challenge0123    44  80.0   10.9   20.7    14.6    36.9  82.0     20.7   31.6   29.4

### filterval / truncation split (V=8 full-view vs V<8 truncated)
model             n  det%  front   rear  corner  worst2  pnp%  honest8  good% gross%
------------------------------------------------------------------------------------
paper_base_v2 V8 106  86.0   15.7   32.4    21.8    48.4  86.0     23.7   22.1   50.5
paper_base_v2 V<8  17  35.0   61.0   69.6    37.6   131.4  35.0     64.3   17.8   68.9
B2 V8           106  83.0   12.2   16.6    13.3    25.8  83.0     15.1   29.4   28.0
B2 V<8           17  24.0    8.2   52.7    24.5    77.2  24.0     53.0   21.4   57.1
challenge0123 V8 106  81.0   12.3   18.3    14.8    31.2  82.0     18.1   27.2   30.2
challenge0123 V<8  17  18.0   28.0   84.1    40.6   136.8  18.0     59.2   18.2   72.7

## paper_base_v2 vs B2 GAP (filterval overall)
  corner_med: paper=22.6 B2=13.5 chal=14.8  Δ(paper-B2)=9.1
  rear_med  : paper=33.5 B2=16.8 chal=18.7  Δ(paper-B2)=16.7
  honest8   : paper=25.1 B2=15.6 chal=18.4  Δ(paper-B2)=9.5
  det%      : paper=79.0 B2=75.0 chal=72.0
  good%     : paper=21.8 B2=29.1 chal=27.0

