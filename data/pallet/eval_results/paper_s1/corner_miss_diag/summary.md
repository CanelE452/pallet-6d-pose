# Per-corner detection diagnostic

- frames (full-9 manual GT): 191
- inference: official no-pad, aspect-preserving (short=400)
- threshold: 0.3
- belief index i == GT manual_kps index i (camera-facing v4)


## Model S1 — per-index miss-rate

```
idx role         n    miss  miss%  meanpeak medLocErr  inside/outside
--------------------------------------------------------------------------
0   near-top-L  191    46   24.1  0.628     7.6    39/7
1   near-top-R  191    91   47.6  0.490     9.1    63/28
2   near-bot-R  191    94   49.2  0.472     8.7    66/28
3   near-bot-L  191    63   33.0  0.569     9.2    50/13
4   far-top-L   191    34   17.8  0.597    26.1    32/2
5   far-top-R   191    53   27.7  0.523    36.6    46/7
6   far-bot-R   191    50   26.2  0.550    35.7    44/6
7   far-bot-L   191    33   17.3  0.611    23.7    31/2
8   centroid    191    17    8.9  0.715    24.5    17/0
```

## Model base — per-index miss-rate

```
idx role         n    miss  miss%  meanpeak medLocErr  inside/outside
--------------------------------------------------------------------------
0   near-top-L  191    53   27.7  0.601     8.4    46/7
1   near-top-R  191    78   40.8  0.469    11.6    52/26
2   near-bot-R  191    81   42.4  0.457    12.4    56/25
3   near-bot-L  191    69   36.1  0.537     8.8    56/13
4   far-top-L   191    47   24.6  0.517    25.4    45/2
5   far-top-R   191    63   33.0  0.506    44.2    57/6
6   far-bot-R   191    55   28.8  0.523    39.8    50/5
7   far-bot-L   191    45   23.6  0.519    29.1    43/2
8   centroid    191    17    8.9  0.664    22.6    17/0
```