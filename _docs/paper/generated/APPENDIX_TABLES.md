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
R0_CONT          None                        0           None           0      28800
R1_NAIVE         F0_NAIVE                  924           1.56       14400      14400
R2_CONF          F1_CONF                   272           5.29       14400      14400
R3_CONF_REPROJ   F2_CONF_REPROJ            251           5.74       14400      14400
R4_CONF_REMOVE   F3_CONF_REMOVE            267           5.39       14400      14400
R5_PROPOSED      F4_PROPOSED               259           5.56       14400      14400
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

## External keypoint baselines

```text
SingleShotPose   NOT_EVALUATED   repository audit 미실시
PVNet            NOT_EVALUATED   repository audit 미실시
```

억지 wrapper 로 숫자를 만들지 않는다. 감사 결과가 나오면 여기 채운다.
