# MULTI_TEACHER_CORNER_DISTILL_V1 — 최종 보고

모집단 DEV_EVAL 319 (PAPER_EVAL positive). 역할은 DEV 다 — 반복 사용됐고 held-out 이 아니다.
독립 확증 모집단이 저장소에 없어(`INDEPENDENT_CONFIRMATION_AVAILABLE = NO`)
아래 어떤 수치도 held-out / confirmed / final / state-of-the-art 가 아니다.
상태는 **DEVELOPMENT_METHOD_SIGNAL** 이다.

## 판정

```
MULTI_TEACHER_HEADROOM            STRONG
LOCAL_CORNER_HEADROOM             STRONG
CLASSICAL_LOCAL_SELECTOR_SIGNAL   ORACLE_ONLY_HEADROOM
REAL_LOCAL_SPECIALIST             STOP
DISTILL_TARGET_QUALITY            FAIL
BEST_STUDENT                      NOT_RUN
TARGET_BIAS_SIGNAL                DOMAIN_SEPARABLE_BUT_NOT_ERROR_LINKED
TARGET_ADAPTER                    NOT_RUN
FINAL_CASE                        CASE_B ORACLE_COMPLEMENTARITY_ONLY
```

## [TEACHERS]

```
teacher                   축                             검출   median      p90  gross20
─────────────────────────────────────────────────────────────────────────────────────
T0_R0_YOLO26N_G38LEGACY   REFERENCE                  1.000     6.36    43.89    0.157
T1_YOLOV8N_G38            ARCHITECTURE               0.994     8.10   108.69    0.219
T2_YOLO11N_G38            ARCHITECTURE               1.000     9.25   109.82    0.257
T3_DOPE_HEATMAP           REPRESENTATION             0.870    13.54    73.38    0.283
T4_YOLO26N_G38ONLY60      TRAINING_SOURCE_COVERAGE   0.994     6.71    68.85    0.180
T5_YOLO26N_BROAD40K       TRAINING_SOURCE_COVERAGE   1.000     6.62   133.84    0.206
T6_YOLO26N_G38ONLY30      ARCHITECTURE_CONTROL       1.000     7.19   135.89    0.231
```

## [HEADROOM]  visible 코너

```
arm                              median      p90  gross20  gross40
──────────────────────────────────────────────────────────────────
best single = R0                   6.36    43.89    0.157    0.102
F1 좌표 성분별 median                   6.31    72.66    0.173    0.129
F2 geometric medoid                6.36    69.91    0.173    0.129
ORACLE per-keypoint (배포불가)         3.44    13.34    0.063    0.040

F3 불확실성 가중 = BLOCKED_NOT_COMPARABLE (SIGMA_STATUS = DIAGNOSTIC_ONLY)
R0 gross20 구제율 (어느 교사라도 <=10px) = 0.303
```

## [CORNER EVIDENCE]  visible 코너, 반경 12px

```
arm                              median      p90  gross20
─────────────────────────────────────────────────────────
R0                                 6.36    43.89    0.157
oracle 후보                          2.00    33.88    0.117
prediction-only 선택기                7.07    45.03    0.168

GT 5px 이내 후보 존재율   0.807
  같은 개수 균등난수 대비 lift   3px +0.069  5px +0.015  10px +0.004
```

## [LOCAL SPECIALIST]  visible 코너

```
arm                              median      p90  gross20    IoU3D   ADDauc
───────────────────────────────────────────────────────────────────────────
R0                                 6.36    43.89    0.157    0.603    0.428
C0 SYN_LOCAL                       5.83    42.17    0.154    0.603    0.437
C1 SYN_PLUS_REAL_SOFT              5.66    43.59    0.156    0.601    0.435
```

## [DISTILL TARGET]  usable 부분집합

```
arm                              median      p90  gross20
─────────────────────────────────────────────────────────
R0                                 5.80    20.14    0.102
F1                                 5.97    21.20    0.109
F2                                 6.14    20.59    0.104
F2S                                6.18    20.82    0.104

coverage  394/2499 = 0.1577
```

## [합의 게이트의 전이]

```
모집단                                        통과율
──────────────────────────────────────────────
SOURCE_DEV                              0.1975
DEV_EVAL_real_visible                   0.1713
TARGET_UNLABELED                        0.0152
```

## [DOMAIN]

```
level0       feature dim    64   domain AUROC 1.0000
level1       feature dim   128   domain AUROC 0.9999
level2       feature dim   256   domain AUROC 0.9999

DEV_EVAL n=319   spearman(score, kp median) -0.0238
gross-frame 분리 AUC  0.48200319233838784
```

## [ADAPTER]

```
NOT_RUN
```
