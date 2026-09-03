# Self-training corner regression — 원인 분해

진단 전용이다.  이 문서를 근거로 threshold·tau·pseudo pool·model 을 바꾸지 않는다.

corner 정의: original-image Euclidean pixel error; predictions are un-padded by INFERENCE_PAD before comparison.

## §3 검출 교차표 (R0 대 R5)

```text
domain         N  BOTH_DET  R0_ONLY  R5_ONLY  BOTH_MISS
────────────────────────────────────────────────────────
daytime       70        69        1        0          0
nighttime     50        42        0        6          2
```

## §4 BOTH_DETECTED 만 — 같은 프레임·같은 supervised keypoint

raw 표와 **별개**다.  여기서는 모집단이 두 모델에 대해 동일하다.

```text
domain       frames  R0 med[px]  R5 med[px]  Δframe med                   CI95       p
────────────────────────────────────────────────────────────────────────────
daytime          69      10.462      11.576      +0.342       [-0.412, +0.814]   0.365
nighttime        42       7.686       8.943      +1.079       [-0.101, +2.633]   0.094
```

`Δframe med` 는 프레임별 median 오차의 차이이고 CI 는 프레임 재표집이다.

### 악화가 고르게 퍼졌나, 몇 장에 몰렸나

```text
domain       worse  better     Δp50     Δp90      Δmax     상위5 몫
──────────────────────────────────────────────────────────────
daytime         37      32    +0.34    +4.64    +202.8     87.5%
nighttime       27      15    +1.08    +8.46    +135.6     76.9%
```

`상위5 몫` = 악화량 합계에서 가장 나빠진 5 장이 차지하는 비율.

## §5 R5 가 새로 회수한 프레임 (R5_ONLY) 과 raw 차이의 분해

통계는 raw 표와 같다 — supervised keypoint 를 통째로 모은 median 이다.

```text
domain        회수 frame    회수 kp med[px]   R0 raw   R5 raw      R5(회수제외)
──────────────────────────────────────────────────────────────────────
daytime              0                —   10.556   11.576        11.576
nighttime            6           22.234    7.686   10.072         8.943
```

### raw 차이가 어디서 왔나 (px, 합이 정확히 total 이 된다)

```text
domain         total     base 모집단  localisation  selection
──────────────────────────────────────────────────────────
daytime       +1.020       -0.094        +1.114     +0.000
nighttime     +2.386       +0.000        +1.257     +1.130
```

`base 모집단` = R0 가 BOTH 에서 낸 값 − R0 가 자기 전체에서 낸 값 (R0_ONLY 프레임의 영향).
`localisation` = 같은 프레임에서 R5 가 R0 보다 나빠진 몫.
`selection` = R5 가 새로 회수한 어려운 프레임이 끌어올린 몫.

## §6 scale-normalised keypoint error (NME)

분모는 GT 에서만 온다.  raw px 는 지우지 않는다 — 둘 다 보고한다.

### 분모 = cuboid_diagonal_px

```text
domain/model/scope                         n_kp    NME med    NME p90
────────────────────────────────────────────────────────────────────
daytime/R0/all_detected                     609     0.0261     0.1650
daytime/R0/both_detected                    600     0.0260     0.1497
daytime/R5_PROPOSED/all_detected            600     0.0284     0.1784
daytime/R5_PROPOSED/both_detected           600     0.0284     0.1784
nighttime/R0/all_detected                   378     0.0262     0.0812
nighttime/R0/both_detected                  378     0.0262     0.0812
nighttime/R5_PROPOSED/all_detected          428     0.0301     0.1215
nighttime/R5_PROPOSED/both_detected         378     0.0286     0.1240
```

### 분모 = gt_bbox_diagonal_px

```text
domain/model/scope                         n_kp    NME med    NME p90
────────────────────────────────────────────────────────────────────
daytime/R0/all_detected                     609     0.0257     0.1625
daytime/R0/both_detected                    600     0.0255     0.1493
daytime/R5_PROPOSED/all_detected            600     0.0279     0.1771
daytime/R5_PROPOSED/both_detected           600     0.0279     0.1771
nighttime/R0/all_detected                   378     0.0242     0.0806
nighttime/R0/both_detected                  378     0.0242     0.0806
nighttime/R5_PROPOSED/all_detected          428     0.0284     0.1196
nighttime/R5_PROPOSED/both_detected         378     0.0265     0.1217
```

### 분모 자체의 산포 — px 가 비교 가능한 단위인가

```text
domain          n   med[px]   p90[px]
────────────────────────────────────
daytime        70     420.6     540.2
nighttime      50     326.8     489.3
```

