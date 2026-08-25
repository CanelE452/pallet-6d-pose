# NIGHT FAILURE DECOMPOSITION

membership n=28 sha16=492e736ba74f0338  sessions {'eval_night08': 12, 'eval_night09': 16}
correct-box rule = IoU >= 0.5 (기존 계약 그대로), conf=0.001, pad=100

## CASCADE (NIGHT)
```
model     n    N0   N1A   N1B   N2   N3   N4
--------------------------------------------
A42      28     0    26     2    0    0    0
C42      28     0     8     7    0   11    2
C43      28     0     1     9    0    8   10
E42      28     0     6     9    0    9    4
FT       28     0     1     0    0    2   25
```

## BOX ROUTING (NIGHT)  ★가장 중요
```
model    any-det   any-cbox  top1-cbox  best IoU med
----------------------------------------------------
A42        1.000      0.071      0.000         0.116
C42        1.000      0.714      0.464         0.633
C43        1.000      0.964      0.643         0.772
E42        1.000      0.786      0.464         0.695
FT         1.000      0.964      0.964         0.865
```

## DAY CONTROL (동일 cascade)
```
model     n    N0   N1A   N1B   N3   N4  top1-cbox   kp med
--------------------------------------------------------------
A42      28     0    11     1   16    0      0.571    65.70
C42      28     0     3     3   10   12      0.786    16.54
C43      28     0     2     4   10   12      0.786    17.95
E42      28     0     9     3    6   10      0.571    14.64
FT       28     0     0     0    2   26      1.000     6.61
```

## RANKING (night, 정답 후보의 순위)
```
{'A42': {'not_present': 26, 'rank3+': 1, 'rank2': 1}, 'C42': {'not_present': 8, 'rank2': 5, 'rank1': 13, 'rank3+': 2}, 'C43': {'rank3+': 2, 'rank2': 7, 'rank1': 18, 'not_present': 1}, 'E42': {'rank1': 13, 'not_present': 6, 'rank3+': 4, 'rank2': 5}, 'FT': {'rank1': 27, 'not_present': 1}}
```

**NIGHT_FAILURE_TYPE = MIXED_NIGHT_FAILURE**  (dominant N3 39%)

- KEYPOINT_FAILURE_NOT_REACHED = False
- NIGHT_FAILURE_SEED_ROBUST = False
- DATA_SCALE_HELPED_NIGHT = True
- FT_REFERENCE = AVAILABLE   FT_SOLVES_NIGHT = True

association only — 인과 주장 아님. 새 학습·새 loss 없음.