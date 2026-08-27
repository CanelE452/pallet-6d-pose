# REAL_DEV FAILURE ATTRIBUTE

model: yolo26n_paper_generic_v1 (60 epoch, target-free BROAD 40K)

> ★ **인과 분리 주장 금지.** target 세션과 night 세션은 **같은 물체**를
> 쓴다. geometry 와 appearance 가 세션 수준에서 얽혀 있어, 아래 교차표는
> 얽힘을 드러내는 것이지 푸는 것이 아니다.

### domain x failure_type

```
bucket              n   NO_BOX   KP_BAD  POSE_BAD    GOOD
outside            22     0.27     0.14      0.00    0.59
noapril            12     0.00     0.00      0.00    1.00
cad                22     0.00     0.00      0.00    1.00
pallet07           27     0.00     0.26      0.00    0.74
pallet09           36     0.50     0.17      0.08    0.25
night08            17     0.59     0.18      0.00    0.24
night09            25     0.44     0.40      0.00    0.16
```

### luma x failure_type

```
bucket              n   NO_BOX   KP_BAD  POSE_BAD    GOOD
dark<60            35     0.43     0.34      0.00    0.23
dim60-100          69     0.28     0.16      0.01    0.55
mid100-140         57     0.19     0.11      0.04    0.67
```

### object size x failure_type

```
bucket              n   NO_BOX   KP_BAD  POSE_BAD    GOOD
small<0.20          2     0.50     0.00      0.00    0.50
mid0.20-0.40       56     0.59     0.07      0.00    0.34
>=0.40            103     0.11     0.24      0.03    0.62
```

### elevation x failure_type

```
bucket              n   NO_BOX   KP_BAD  POSE_BAD    GOOD
<3                 35     0.43     0.11      0.06    0.40
3-8                61     0.48     0.16      0.02    0.34
8-15               34     0.03     0.44      0.00    0.53
15+                31     0.00     0.00      0.00    1.00
```

### truncation x failure_type

```
bucket              n   NO_BOX   KP_BAD  POSE_BAD    GOOD
full              125     0.32     0.11      0.02    0.54
truncated          36     0.14     0.42      0.00    0.44
```

### per-keypoint (CHALLENGE_105)

```
kp  group              err med   err p90  conf med  missing
0   near top              11.6     272.6     0.969     0.37
1   near top              13.7     237.8     0.993     0.37
2   near bottom           13.6     271.0     0.992     0.37
3   near bottom           11.1     336.8     0.975     0.37
4   far top               22.7     283.1     0.992     0.37
5   far top               26.4     197.2     0.990     0.37
6   far bottom            30.1     196.2     0.993     0.37
7   far bottom            25.7     305.5     0.994     0.37
```

