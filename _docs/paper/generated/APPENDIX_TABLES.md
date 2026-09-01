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

## External keypoint baselines

```text
SingleShotPose   NOT_EVALUATED   repository audit 미실시
PVNet            NOT_EVALUATED   repository audit 미실시
```

억지 wrapper 로 숫자를 만들지 않는다. 감사 결과가 나오면 여기 채운다.
