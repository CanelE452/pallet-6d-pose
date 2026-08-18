# fourway — ★정본 평가셋(split==eval 56장) Method 표

```
domain         method           weight              N  succ%   det%   ADD m    yaw°
--------------------------------------------------------------------------
outside_ft     Synthetic only   R0             35/60    45.0   58.3   0.358    7.47
outside_ft     Naive ST         h9_outside     43/60    43.3   71.7   0.361    9.06
outside_ft     Reproj+flip ST   rf_outside     47/60    40.0   78.3   0.429   10.12
outside_ft     Ours (loo+flip)  h8_outside     51/60    45.0   85.0   0.334    9.55
--------------------------------------------------------------------------
outside_ft_p07 Synthetic only   R0             15/26    46.2   57.7   0.292    4.91
outside_ft_p07 Naive ST         h9_outside     16/26    46.2   61.5   0.290    6.24
outside_ft_p07 Reproj+flip ST   rf_outside     16/26    42.3   61.5   0.319    6.99
outside_ft_p07 Ours (loo+flip)  h8_outside     19/26    61.5   73.1   0.262    4.40
--------------------------------------------------------------------------
outside_ft_p09 Synthetic only   R0             20/34    44.1   58.8   0.941   16.11
outside_ft_p09 Naive ST         h9_outside     27/34    41.2   79.4   0.447   11.10
outside_ft_p09 Reproj+flip ST   rf_outside     31/34    38.2   91.2   0.581   12.98
outside_ft_p09 Ours (loo+flip)  h8_outside     32/34    32.4   94.1   0.351   13.49
--------------------------------------------------------------------------
outside_fv     Synthetic only   R0             18/22    50.0   81.8   0.228    4.41
outside_fv     Naive ST         h9_outside     21/22    59.1   95.5   0.291    6.15
outside_fv     Reproj+flip ST   rf_outside     21/22    40.9   95.5   0.562    5.94
outside_fv     Ours (loo+flip)  h8_outside     20/22    59.1   90.9   0.158    3.61
--------------------------------------------------------------------------
night_ft       Synthetic only   R0             24/39    38.5   61.5   0.281    5.93
night_ft       Naive ST         h9_night       25/39    35.9   64.1   0.393    3.02
night_ft       Reproj+flip ST   rf_night       29/39    38.5   74.4   0.430    2.56
night_ft       Ours (loo+flip)  h8_night       26/39    46.2   66.7   0.297    2.76
--------------------------------------------------------------------------
night_ft_n08   Synthetic only   R0             12/17    47.1   70.6   0.160    3.30
night_ft_n08   Naive ST         h9_night       13/17    58.8   76.5   0.359    1.84
night_ft_n08   Reproj+flip ST   rf_night       12/17    47.1   70.6   0.204    2.17
night_ft_n08   Ours (loo+flip)  h8_night       12/17    64.7   70.6   0.208    2.24
--------------------------------------------------------------------------
night_ft_n09   Synthetic only   R0             12/22    31.8   54.5   0.417    9.52
night_ft_n09   Naive ST         h9_night       12/22    18.2   54.5   0.470   16.80
night_ft_n09   Reproj+flip ST   rf_night       17/22    31.8   77.3   0.448    3.92
night_ft_n09   Ours (loo+flip)  h8_night       14/22    31.8   63.6   0.350    8.84
--------------------------------------------------------------------------
noapril        Synthetic only   R0             11/12    91.7   91.7   0.056    0.89
noapril        Naive ST         h9_noapril     11/12    91.7   91.7   0.061    0.95
noapril        Reproj+flip ST   rf_noapril     11/12    91.7   91.7   0.083    1.19
noapril        Ours (loo+flip)  h8_noapril     11/12    91.7   91.7   0.053    0.73
--------------------------------------------------------------------------
cad            Synthetic only   R0              2/22     9.1    9.1   0.051    1.45
cad            Naive ST         NA                 NA
cad            Reproj+flip ST   rf_combined     0/22     0.0    0.0     nan     nan
cad            Ours (loo+flip)  h8_combined     2/22     9.1    9.1   0.050    1.26
--------------------------------------------------------------------------
```

n_gt (split==eval, GT reproj<=5 통과): {'outside_ft': 60, 'outside_ft_p07': 26, 'outside_ft_p09': 34, 'outside_fv': 22, 'night_ft': 39, 'night_ft_n08': 17, 'night_ft_n09': 22, 'noapril': 12, 'cad': 22}
night 은 정본 eval 프레임 0장이라 이 표에 없다.
cad 는 도메인 전용 self-train 모델이 없어 combined 가중치로 대체(weight 열 참조).
ADD 는 표준 ADD(대칭 fold 없음). transductive 한계는 C3 스펙 문서 §0 참조.
