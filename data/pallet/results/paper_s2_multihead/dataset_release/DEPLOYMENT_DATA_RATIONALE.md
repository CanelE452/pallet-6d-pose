# DEPLOYMENT DATA RATIONALE

## [BASE]

`BROAD_40K MH_TRAIN` 33,758 장이 core 다. 이유는 크기가 아니라 증거다 —
E3/E4 를 포함해 지금까지 **효과가 확인된 학습은 전부 이 pool 위에서 났다**.
다른 어떤 pool 도 positive 학습 증거가 없다.

## [EXCLUDED FROM MAIN]

```
Y15_30      2,500   ABLATION_ONLY   canonical 재분류 시 겨냥 cell 이 near-baseline
                                    (9.62px, x1.14) — 개입이 아니라 control
Y30_PLUS    2,500   ABLATION_ONLY   C1/C1_RESCUE 에서 두 seed 방향 충돌, CI 0 포함
FRONTAL         0   PRESERVE_UNUSED 렌더 0장. FRONTAL_DATA_DECISION.md 가
                                    TARGETED_ENRICHMENT_NOT_ESTABLISHED 로 종결
NEGATIVE   10,000   CALIBRATION     dense negative training 은 REJECTED
                                    (seed2 pose safety 대실패). 최종 rejection 은
                                    score_4kp 이고 pose network 를 건드리지 않는다
```

CORNER_LA 두 세트는 **5,000 장이 있다는 이유로는 들어가지 않는다.** 효과가
미확립이고, 넣으면 그 효과를 분리할 수 없게 된다.

## [LINE HARD SUPPLEMENT]

사용 여부 = **candidate 로만**. main 승격 아님.

```
ratio         = 0.05  (CONSERVATIVE)
근거          = protective clauses only (clause 1 vacuous: BROAD covers every line-hard cell with 0 frames, so no ratio can be shown to be required by coverage) + conservatism default
```

후보 3개의 실제 값:

```
                ratio   broad mode 보존(min)   EDGE 반복(broad 1회당)
CONSERVATIVE    0.05    0.950                  0.18x
BALANCED        0.12    0.880                  0.46x
AGGRESSIVE      0.20    0.800                  0.84x   <- 0.85 미달로 탈락
```

**어떤 cell 이 얼마에서 얼마로 늘어나는가** — 정직하게 쓰면 이렇다.

```
cell (V_vis<=3)                      BROAD    EDGE      노출 5% 적용 후
──────────────────────────────────────────────────────────────────────
('<=3','truncated','0.40-0.60')          0    2,841     0  ->  0.0142
('<=3','truncated','0.60-0.85')          0    2,759     0  ->  0.0138
('<=3','truncated','>=0.85')             0    2,202     0  ->  0.0110
('<=3','truncated','0.25-0.40')          0    1,614     0  ->  0.0081
('<=3','truncated','<0.25')              0      583     0  ->  0.0029
('<=3','full','0.25-0.40')               0        1     0  ->  0.0000
합                                       0   10,000     0  ->  0.0500
```

즉 "N 에서 N 으로 늘었다" 가 아니라 **0 에서 생겼다**. BROAD 는 G1 게이트가
`V_vis >= 4` 를 요구해 이 영역을 정의상 배제한다. 그래서 coverage 논거만으로는
5% 와 20% 중 무엇이 필요한지 **가릴 수 없다**. 5% 는 보호 조항(broad mode 보존
0.950, EDGE 반복 0.18x)과 보수적 기본값으로 고른 값이지, coverage 가 요구한
값이 아니다. 이 구분을 뭉개지 않는다.

broad 는 그대로 95% 를 유지하고, EDGE 10,000 장은 broad 1회 통과당 0.18배만
재사용되므로 **같은 프레임을 반복 노출해 exposure 만 늘리는 상태가 아니다**.

## [DIVERSITY]

```
axis               BROAD unique   혼합 후 unique   BROAD 최대비중   혼합 후
pallet_type                   4                4           0.255     0.254
resolution                    4                4           0.483     0.494
background_asset              2                2           0.510     0.507
noise_tier                    4                4           0.605     0.603
```

## [FINAL]

```
Corner training pool = BROAD_40K MH_TRAIN 33,758   (effective exposure 1.00)
Line   training pool = BROAD_40K MH_TRAIN 33,758   (effective exposure 0.95)
                     + EDGE_HARD_TRUNC_TRAIN 10,000  (effective exposure 0.05)
```

corner stream 에 EDGE 는 **절대 들어가지 않는다** — G1 이 False 라 4개 미만
코너로 pose 를 가르치게 된다. 이건 성능 판단이 아니라 계약 위반이다.
