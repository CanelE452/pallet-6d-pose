# Appendix tables

## A7 — elevation and broad lighting subgroups

```text
Condition            N  R0 corner↓  R5 corner↓        Δ
────────────────────────────────────────────────────────
Low                122       5.166       4.626   -0.539
Mid                138       4.035       3.787   -0.248
High                57       4.591       4.521   -0.069
Lighting_day       168       4.042       3.710   -0.332
Lighting_night     106       4.735       4.613   -0.123
```

## A1 — pseudo-label counts and exposure contract

```text
Arm              filter              unique PL  pseudo/unique  pseudo exp  synth exp
────────────────────────────────────────────────────────────────────────────────────
A2_NAIVE_MATCHED_S1 A2_NAIVE_MATCHED_S1        258           5.58       14400      14400
A2_NAIVE_MATCHED_S2 A2_NAIVE_MATCHED_S2        259           5.56       14400      14400
A2_NAIVE_MATCHED_S3 A2_NAIVE_MATCHED_S3        259           5.56       14400      14400
```

모든 arm 이 같은 900 optimizer update 를 쓴다.
MAIN 은 EXPOSURE-MATCHED 이고, unique PL 개수를 맞추는 실험은 A2 다.

## A3 — repeatability across pseudo-sampling replicates

Ultralytics 의 `seed` 는 dataloader 에 도달하지 않는다.  seed 42/43/44 로
학습한 가중치는 **비트 동일**했다 (max|Δw| = 0, 텐서 비교로 확인) — 따라서
seed override 는 독립 반복이 아니다.  여기서 쓰는 replicate 는 우리가
통제하는 **pseudo 샘플링**을 바꾼 것이고, 노출 총량은 그대로다.

```text
Method                   metric         rep1     rep2     rep3     mean      std
──────────────────────────────────────────────────────────────────────────────────
Naive self-training      corner↓       4.335    4.422    4.367    4.375    0.036
Naive self-training      det↑          0.981    0.991    0.984    0.985    0.004
Naive self-training      AUROC↑       0.9913   0.9894   0.9861   0.9889   0.0021
Naive self-training      FPR95↓       0.0558   0.0651   0.0948   0.0719   0.0167
Proposed                 corner↓       4.180    4.177    4.149    4.169    0.014
Proposed                 det↑          0.984    0.987    0.987    0.986    0.001
Proposed                 AUROC↑       0.9953   0.9933   0.9929   0.9938   0.0011
Proposed                 FPR95↓       0.0283   0.0364   0.0409   0.0352   0.0052
```

세 replicate 에서 Proposed 와 Naive 의 구간이 겹치는지: corner: 구간 분리 · AUROC: 구간 분리 · FPR95: 구간 분리

구간이 겹치면 그 지표에서는 효과가 산포 안이라는 뜻이므로 주장하지 않는다.

## A2 — unique-quantity-matched control

성능 향상이 **선별 품질** 때문인지 **pseudo-label 개수** 때문인지 가른다.
Naive pool 에서 Proposed 와 같은 unique 개수(259)를 무작위로 뽑아,
나머지는 전부 같게 두고 3 회 반복했다.  MAIN 의 EXPOSURE-MATCHED 와는
다른 실험이다 — 여기서 맞추는 것은 노출량이 아니라 unique 개수다.

```text
Arm                          metric         rep1     rep2     rep3     mean      std
──────────────────────────────────────────────────────────────────────────────────────
Naive, quantity-matched      corner↓       4.435    4.332    4.174    4.314    0.108
Naive, quantity-matched      AUROC↑       0.9841   0.9893   0.9924   0.9886   0.0034
Naive, quantity-matched      FPR95↓       0.0997   0.0747   0.0398   0.0714   0.0246
Proposed                     corner↓       4.180    4.177    4.149    4.169    0.014
Proposed                     AUROC↑       0.9953   0.9933   0.9929   0.9938   0.0011
Proposed                     FPR95↓       0.0283   0.0364   0.0409   0.0352   0.0052
```

판정: corner: 구간 겹침 · AUROC: 구간 분리 · FPR95: 구간 겹침

구간이 겹치는 지표에서는 **개수를 맞춘 무작위 선별로도 비슷한 값에
도달한다**는 뜻이다.  그 지표에 대해서는 geometry filter 의 기여를
주장하지 않는다.  개수 효과와 선별 품질 효과를 합쳐 말하지 않는다.

## External keypoint baselines

```text
SingleShotPose   NOT_EVALUATED   repository audit 미실시
PVNet            NOT_EVALUATED   repository audit 미실시
```

억지 wrapper 로 숫자를 만들지 않는다. 감사 결과가 나오면 여기 채운다.
