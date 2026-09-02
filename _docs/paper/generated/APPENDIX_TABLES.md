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
B_CONF_RANDOM_S1 B_CONF_RANDOM_S1          259           5.56       14400      14400
B_CONF_RANDOM_S2 B_CONF_RANDOM_S2          259           5.56       14400      14400
B_CONF_RANDOM_S3 B_CONF_RANDOM_S3          259           5.56       14400      14400
B_CONF_TOPN      B_CONF_TOPN               259           5.56       14400      14400
B_CONF_DECILE    B_CONF_DECILE             259           5.56       14400      14400
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

## A2b — self-training baseline comparison (aggregate)

제안 방법의 효과가 단순 self-training 이나 reprojection filtering 만으로
얻어지는지 확인한다.  전체 population 집계다 (도메인별은 M2).

```text
Method              corner↓    det↑   @5px↑  @10px↑  @20px↑     AP↑   AUROC↑   FPR95↓   R med↓    yaw↓
────────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic only        4.420   0.975   0.573   0.814   0.887  0.7688   0.9921   0.0417        —       —
Naive ST              4.335   0.981   0.564   0.807   0.893  0.7622   0.9913   0.0558        —       —
Reproj-only ST        4.274   0.987   0.568   0.799   0.874  0.7643   0.9920   0.0487        —       —
Ours                  4.180   0.984   0.595   0.812   0.907  0.7585   0.9953   0.0283        —       —
```

반드시 답해야 하는 질문:

```text
1. synthetic only 보다 self-training 이 좋은가   4.420 -> 4.335   예
2. naive 보다 filtering 이 좋은가                4.335 -> 4.180   예
3. reproj-only 보다 제안 filter 가 좋은가        4.274 -> 4.180   예
```

1·3 은 replicate 산포와 함께 읽어야 한다 — A3 · 2x2 표 참조.

## A7 — full robustness metric battery (Proposed)

M5 는 지면 때문에 지표를 줄여 싣는다.  여기서는 같은 subgroup 을 전체 열로 본다.
subgroup 은 서로 중복될 수 있어 합계가 전체 N 이 되지 않는다.

```text
Subgroup             N     src  corner↓     p90↓   @5px↑  @10px↑  @20px↑  gross↓    det↑   AUROC↑   FPR95↓  R med↓   yaw↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Plastic            194  strict    3.985    22.87   0.638   0.822   0.897   0.103   0.979   0.9948   0.0324       —      —
Wood               125  strict    4.362    18.28   0.560   0.803   0.916   0.084   0.992   0.9962   0.0231       —      —
Daytime             70    diag   11.592    56.57   0.169   0.433   0.673   0.327   0.986   0.9940   0.0294       —      —
Nighttime           50  strict    5.271    17.89   0.480   0.737   0.909   0.091   0.960   0.9920   0.0588       —      —
Lighting_day       168  strict    3.710    64.97   0.662   0.840   0.889   0.111   0.982   0.9954   0.0294       —      —
Lighting_night     106  strict    4.613    17.70   0.538   0.788   0.923   0.077   0.981   0.9939   0.0152       —      —
Clean              184  strict    4.031    16.32   0.623   0.830   0.920   0.080   0.989   0.9968   0.0216       —      —
Occlusion          135  strict    4.626    25.82   0.526   0.767   0.876   0.124   0.978   0.9933   0.0294       —      —
Truncation          51  strict    6.122   115.78   0.407   0.661   0.836   0.164   0.922   0.9922   0.0402       —      —
Far                 59  strict    3.285    14.31   0.744   0.876   0.927   0.073   1.000   0.9959   0.0257       —      —
Low                122  strict    4.626    16.91   0.551   0.801   0.949   0.051   0.967   0.9920   0.0361       —      —
Mid                138  strict    3.787    21.27   0.640   0.819   0.897   0.103   0.993   0.9973   0.0152       —      —
High                57  strict    4.521    20.33   0.544   0.803   0.897   0.103   1.000   0.9975   0.0231       —      —
```

`src` 가 diag 인 행은 all-annotated 진단값이다 — strict 행과 절대값을
직접 비교하지 않는다.  pose 열은 BLOCKED.

## A9 — same-data backbone control

"왜 YOLO26 인가" 에 답한다.  두 모델은 같은 55,980 synthetic frame 으로
60 epoch 학습했고 real 감독은 0 이다.  다른 것은 백본뿐이다.

```text
Model             Train frames  Epochs    det↑  corner med↓  corner p90↓   @5px↑  @10px↑  @20px↑
──────────────────────────────────────────────────────────────────────────────────────────────
DOPE                    55,980      60   0.737       10.083        27.85   0.148   0.494   0.853
YOLO26n-Pose            55,980      60   0.975        4.420        27.08   0.573   0.814   0.887
```

`det 8/8` 와 `det>=6` 은 넣지 않았다.  DOPE 의 코너 검출은 belief
threshold, YOLO 는 keypoint confidence 라 같은 양이 아니다 — 한 열에
놓으면 비교처럼 보이지만 비교가 아니다.  대신 GT 와의 거리로만 정의되는
corner 와 Proj@N 을 쓴다.

## A1 — pseudo-label pass / retention by domain

필터가 unlabeled pool 에서 pseudo-label 을 얼마나 남기는지 본다.
**모델 정확도 표가 아니다** — 통과 수가 적다는 사실만으로 품질을 주장하지 않는다.

```text
Filter                                        Day pass  Day ret  Night pass  Night ret   Total
──────────────────────────────────────────────────────────────────────────────────────────────
No filter                                          472    0.944         454      0.908     926
Confidence                                         123    0.246         149      0.298     272
Confidence + Reprojection                          116    0.232         135      0.270     251
Confidence + Keypoint-removal consistency          120    0.240         147      0.294     267
Confidence + Horizontal-flip consistency           123    0.246         140      0.280     263
Proposed                                           120    0.240         139      0.278     259
```

funnel (unlabeled 1000 장):

```text
total                              1000
detected                           954
candidate_min_valid_corners        926
confidence                         272
confidence_reprojection            251
confidence_keypoint_removal        267
confidence_flip                    263
proposed                           259
```

retention = 통과 수 / 그 도메인 pool 500.

## A6 — evaluation dataset composition

논문이 쓴 평가셋이 무엇인지 재현 가능하게 남긴다.  수치는 manifest 에서 읽는다.

```text
Population                      N   근거
──────────────────────────────────────────────────────────────────────────────
PAPER_EVAL_ALL_POS            319   plastic + wood, SHA-dedup union(DEV_EVAL, NEW_EVAL)
PAPER_EVAL_PLASTIC_POS        194   DEV role
PAPER_EVAL_WOOD_POS           125   CROSS_SHAPE_DEV role
DEV_NEG2689                  2689   negative, 2,688 unique image
```

```text
Condition             N
────────────────────────
Daytime              70
Nighttime            50
Clean               184
Occlusion           135
Truncation           51
Far                  59
Low                 122
Mid                 138
High                 57
```

조건은 서로 중복될 수 있고 합계가 전체 N 이 되지 않는다.
held_out_final 은 false 다 — PAPER_EVAL 은 DEV role 이다.

adaptation pool 은 평가셋과 분리돼 있다.

```text
adapt session ∩ eval session   0
adapt image SHA ∩ eval SHA     0
U_MAIN                         1000  (Daytime 500 + Nighttime 500)
```

## A8 — cross-domain transfer matrix

M2 는 "그 도메인 데이터로 적응하면 그 도메인이 좋아지는가" 를 묻는다.
여기서는 **다른 도메인으로 적응해도 좋아지는가** 를 묻는다.

행 = 적응에 쓴 unlabeled 도메인, 열 = 평가 도메인.  값은 detection rate(↑)와
괄호 안 corner(↓, all-annotated 진단).  모두 Proposed 필터를 쓴 pseudo-label 이다.

```text
Adaptation        unique PL           Test Daytime         Test Nighttime
──────────────────────────────────────────────────────────────────────────
None                      0          1.000 (10.93)           0.840 (7.69)
Daytime                 120          0.929 (11.14)           0.940 (8.76)
Nighttime               139          0.971 (11.21)           0.960 (9.08)
Day + Night             259          0.986 (11.59)          0.960 (10.07)
```

### 해석 (규칙은 결과 보기 전에 고정됐다)

```text
대각선만 개선     target-specific adaptation — 도메인마다 데이터가 필요
비대각선도 개선   cross-domain transfer — 한 도메인이 다른 도메인도 돕는다
Day+Night 최선    도메인을 나눌 필요가 없다
Day+Night 열위    도메인 혼합이 해롭다 (negative transfer)
```

행마다 unlabeled pool 크기가 다르다 (Day 120 · Night 139 · 합 259).
그 차이가 결과를 설명할 수 있으므로 unique PL 을 같이 싣는다.

## A2c — geometry incremental control (confidence pool)

A2 는 Naive pool 에서 뽑았다.  이건 **confidence 를 이미 통과한** pool(272)에서
Proposed 와 같은 unique 수(259)를 뽑는다.  즉 geometry 가 추가로 걷어낸 13 장의
고유 기여만 분리한다.  A2 와 다른 실험이므로 섞지 않는다.

```text
Selection from confidence pool    unique   corner↓    AUROC↑    FPR95↓
────────────────────────────────────────────────────────────────────────
random matched (n=3)                 259     4.244    0.9941    0.0345
confidence-ranking top-N             259     4.315    0.9944    0.0342
confidence-decile matched            259     4.201    0.9947    0.0286
Proposed (geometry) (n=3)            259     4.169    0.9938    0.0352
Confidence only, all 272             272     4.242    0.9923    0.0469
```

### 판정

```text
corner   random 4.2436  Proposed 4.1685   구간 겹침
AUROC    random 0.9941  Proposed 0.9938   구간 겹침
FPR95    random 0.0345  Proposed 0.0352   구간 겹침
```

**confidence 를 통과한 pool 안에서는 무작위로 같은 수를 뽑아도 geometry
선별과 구분되지 않는다.**  A2 에서 유일하게 분리됐던 AUROC 도 여기서는
겹친다 — 그 이득은 geometry 가 아니라 confidence 단계에서 온 것이다.

geometry 필터의 **추가** 기여는 이 데이터로 입증되지 않는다.  M4 가 보여준
선별 능력(통과분 gross 0.072 대 기각분 0.299)은 실재하지만, 그것이
downstream 성능 이득으로 전이된다는 증거는 없다.

## External keypoint baselines

```text
SingleShotPose   NOT_EVALUATED   repository audit 미실시
PVNet            NOT_EVALUATED   repository audit 미실시
```

억지 wrapper 로 숫자를 만들지 않는다. 감사 결과가 나오면 여기 채운다.
