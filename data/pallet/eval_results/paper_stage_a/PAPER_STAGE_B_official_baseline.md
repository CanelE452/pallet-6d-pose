# PAPER_STAGE_B — paper_base_v2 공식 논문 baseline (no-pad 확정)

weights: weights/paper_base/paper_base_v2/final_net_epoch_0060.pth (procedural 19,308, palletobj-free, scratch 60ep, 무패딩 학습)
source: data/pallet/eval_results/paper_stage_a/eval.json (재실행 없음, PAPER_STAGE_A 측정 재사용)
★ official quantitative protocol = A_nopad(aspect) | qualitative coverage demo = C_pad100

## filterval (N=123, ★주 신호)
```
preprocess   det%  front   rear  corner  honest8  good%  gross%   PnP%
----------------------------------------------------------------------
A_nopad        68   16.4   34.7    27.5     31.7   28.7    48.1     71
B_pad50        74   16.9   30.9    22.2     24.2   26.0    47.8     75
C_pad100       79   16.2   33.5    22.6     25.1   21.8    51.6     79
D_pad150       71   18.6   34.8    23.4     32.6   17.4    51.9     72
  V=8 (nopad)   77   15.3   34.2    27.2     28.4   28.6    47.7     79
  V<8 (nopad)   12   54.5   64.6    60.0     70.2   31.2    62.5     18
```

## handannot17 (N=17, 고앙각 편향, 정성)
```
preprocess   det%  front   rear  corner  honest8  good%  gross%   PnP%
----------------------------------------------------------------------
A_nopad        24    5.8    9.3     7.6     28.0   76.9     3.8     24
B_pad50        24    6.2    9.6     7.6     16.9   63.0    14.8     29
C_pad100       47    9.1   17.0    13.2     18.9   45.5    21.8     47
D_pad150       53   12.2   19.4    13.8     20.6   31.7    30.2     53
  V=8 (nopad)   75    6.6    9.2     7.7     19.9   80.0     5.0     75
  V<8 (nopad)    8    4.3    9.3     7.5     87.9   66.7     0.0      8
```

## 판정 (filter-val 주 신호)
- official = **A_nopad**: good% 28.7 / gross% 48.1 / corner 27.5 / rear 34.7 / honest8 31.7 / det 68 / PnP 71
- demo = C_pad100: good% 21.8 / gross% 51.6 / corner 22.6 / det 79  (검출↑ 순도↓)
- no-pad 선택 근거: train/infer parity(무패딩 학습) + 순도(good 29>22, gross 48<52). pad100은 검출/coverage demo 별도.
- ⚠ median 축(corner/rear/honest)은 B_pad50이 최강이나 train/infer parity 밖 → robustness ablation 행으로만. 소표본(outside44/night43/manual36, handannot17).
- rear>front 전 pad·전 set 일관(rear 병목). paper_base_v2 = 정직한 논문 baseline.