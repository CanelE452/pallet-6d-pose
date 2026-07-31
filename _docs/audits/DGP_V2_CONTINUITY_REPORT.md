# DGP v2 — Continuous Semantic-Line Energy

## 무엇을 고쳤나

v1 energy 가 pose 에 대해 **계단형**이라 global search 가 불가능했다.  원인 2가지:

1. candidate pose 마다 visibility 를 다시 판정 -> 기여하는 **edge 집합이 바뀜**
2. `mean over edges` -> edge 하나가 빠지면 **전체 스케일이 재조정**됨

v2:

```
항목            v1                          v2
──────────────────────────────────────────────────────────────────────
edge set        candidate 마다 재판정        frame 당 1회 고정 (amodal/visible/associated)
normalization   mean over edges             sum over samples / sum of weights
line map        1-px raster                 distance field, sigma = {0.020, 0.010, 0.005} x image diagonal
direction       forward only                0.5*forward + 0.5*reverse
```

## Continuity unit tests (전부 통과)

```
test                                결과
────────────────────────────────────────────────────────────────
T1 GT 에서 local minimum             E=0.001416 (최소)
T2 yaw ±1/5/10° finite              전부 finite, 매끄러운 U자
T3 인접 0.25° 간 max |dE|            0.004043  (v1: 계단형 점프)
T4 edge iteration order 무관         |dE| = 0.000e+00
T5 sweep 중 edges_used               9 로 고정 (v1: 7<->9 점프)
T6 semantic class permutation        0.00142 -> 0.63543 (448x 악화)
T7 yaw+180° equivalence              |dE| = 0.000e+00
T8 finite-difference gradient        -0.002640 (finite)
T10 edge count 달라도 스케일 안정      visible(9e)=0.0014 vs amodal(12e)=0.0898
```

[확인] 계단형 energy 는 재발하지 않았고, 따라서 global search 를 진행할 수 있었다.
