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

## A12 — self-training strength sensitivity

Proposed(F4) pseudo-label manifest 를 그대로 쓰고 pseudo:synthetic 비율만
바꿨다.  총 optimizer update · LR · init · augmentation · seed 는 모두 같다.

묻는 것은 "Proposed 의 효과가 특정 mixing ratio 에만 의존하는가" 다.
**hyperparameter search 가 아니다** — MAIN row 는 결과와 무관하게 0.50 이다.

```text
Pseudo fraction   corner↓    det↑   AUROC↑   FPR95↓    Day↓*   Night↓*
──────────────────────────────────────────────────────────────────────
0.25                4.266   0.984   0.9930   0.0390   10.612     9.641
0.50 (MAIN)         4.180   0.984   0.9953   0.0283   11.592    10.072
0.75                4.265   0.984   0.9916   0.0602   11.824     9.789
```

corner spread across ratios: 0.086 px (min 4.180 / max 4.266)

* Day/Night 는 all-annotated 진단값이다.

0.25 나 0.75 가 더 좋아도 MAIN row 를 교체하지 않는다.

## Supervised upper bound (Real-FT)

**controlled comparison 이 아니다.**  이 checkpoint 들은 real GT 로 직접
학습했고, base 도 R0 가 아니라 다른 synthetic run 이다.  M1 의 controlled
row 로 읽으면 안 된다 — 도달 가능한 상한을 가늠하는 용도다.

leakage 감사: PAPER_EVAL 319 와의 중복이 **이미지 SHA 0 건, 파일명 stem
0 건** 이다.  따라서 `LEAKED_SUPERVISED_UPPER_BOUND` 가 아니라 그냥
`SUPERVISED UPPER BOUND` 로 표기한다.

```text
Checkpoint                          corner↓    det↑  AP50-95↑   AUROC↑   FPR95↓
────────────────────────────────────────────────────────────────────────────────
ft_a (real157+neg259+synth12k)        4.553   0.994    0.8399   0.9991   0.0041
ft_b (patience0 ep40)                 4.119   0.994    0.8293   0.9992   0.0000
legacy v1v2 FT                        5.547   0.984    0.7172   0.9896   0.0513
Proposed (label-free)                 4.180   0.984    0.7585   0.9953   0.0283
```

Proposed 는 real label 을 한 장도 쓰지 않았다.  같은 표에 두는 이유는
상한과의 거리를 보이기 위해서지 같은 조건의 비교라서가 아니다.

## External keypoint baselines

```text
SingleShotPose   NOT_EVALUATED   repository audit 미실시
PVNet            NOT_EVALUATED   repository audit 미실시
```

억지 wrapper 로 숫자를 만들지 않는다. 감사 결과가 나오면 여기 채운다.
