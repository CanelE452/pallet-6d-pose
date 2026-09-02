# V4 geometry repair — 학습 전 proxy test

복원된 좌표가 teacher 원본보다 GT 에 가까운지 **학습 전에** 채점한다.
teacher 는 R0, 모집단은 PAPER_EVAL 319 (V4 development set).

같은 keypoint 에 대한 paired 비교다.  NME 는 GT projected cuboid diagonal 로
정규화했고, gross 는 20 px, catastrophic 은 40 px 다.

```text
group                          n   raw NME   rep NME          Δ  raw gross  rep gross
------------------------------------------------------------------------------------
ALL                            1   0.02069   0.02599   +0.00530      0.000      1.000
nighttime                      0  (표본 없음)
daytime                        0  (표본 없음)
night_occlusion_medium         0  (표본 없음)
teacher_conf_0.50_0.80         0  (표본 없음)
teacher_conf_0.80_0.95         1   0.02069   0.02599   +0.00530      0.000      1.000
```

## 복원 상태 분포

```text
AMBIGUOUS_VIEW             15
HYPOTHESIS_DISAGREE        5
NO_VALID_HYPOTHESIS        1
OUT_OF_IMAGE               2
REPAIRED                   2
```

## Proxy gate

```text
FAIL  P1_night_median_improves: 야간 복원 표본이 0 개 — 비교 불가
FAIL  P2_night_gross_not_worse: 야간 복원 표본이 0 개 — 비교 불가
FAIL  P3_power: 복원 성공 keypoint  ALL n=1  Night n=0  LOW_POWER
```

**FAIL** — `GEOMETRY_REPAIR_MECHANISM_FAIL`

P3 는 최소 N 임계를 새로 만든 것이 아니라, 표본이 비교를 지탱하지 못할 때 LOW_POWER 로 표시하기 위한 것이다.

