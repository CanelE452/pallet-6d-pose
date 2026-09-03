# Table M2 — Target-domain adaptation under daytime and nighttime conditions

Daytime N=70, Nighttime N=50 (plastic only — morphology 를 lighting 효과와 섞지 않는다).

## primary — detection and ranking

2026-09-02 의 visibility 확정 전에는 MAIN Daytime 70 장의 supervision mask 가
비어 strict keypoint 오차를 낼 수 없었다. 지금은 319 장 전부 strict 를 내지만,
arm 을 가르는 축은 여전히 detection 과 ranking 이다. 그래서 detection 과
ranking 을 primary 로 둔다.

```text
Method                              Day det↑  Night det↑    Mean↑   Worst↑   AUROC↑   FPR95↓
────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only                         1.000       0.840    0.920    0.840   0.9921   0.0417
Source-only continuation               1.000       0.800    0.900    0.800   0.9872   0.0573
Naive self-training                    0.971       0.960    0.966    0.960   0.9913   0.0558
Confidence-based self-training         0.971       0.980    0.976    0.971   0.9923   0.0469
Reprojection-based self-training       0.971       0.960    0.966    0.960   0.9920   0.0487
Proposed                               0.986       0.960    0.973    0.960   0.9953   0.0283
```

detection 은 ↑ 가 좋으므로 `Worst` 는 두 조건 중 **낮은** 쪽이다.
AUROC / FPR95 는 전체 population 대 negative 2,689 로 계산한 frame-level 값이다.

## secondary — keypoint localisation

`strict` 는 evaluator 의 supervision mask 를 쓴 값이고, `diagnostic` 은
visibility 와 무관하게 좌표가 있는 점을 전부 센다. **diagnostic 은
visible/occluded 주장이 아니다** — 두 열은 서로 다른 모집단이라 직접 비교하지 않는다.

```text
Method                              Day strict↓  Night strict↓  Day diag↓  Night diag↓  ALL strict↓
──────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only                           10.556          7.686     10.928        7.686        6.616
Source-only continuation                 10.555          7.964     10.588        7.964        6.911
Naive self-training                      11.852          8.309     12.023        8.440        7.120
Confidence-based self-training           12.380          9.465     12.447        9.601        7.037
Reprojection-based self-training         11.461          8.642     11.541        8.687        7.044
Proposed                                 11.576         10.072     11.592       10.072        7.210
```

Daytime strict keypoint 수 = 609 / 630 annotated (사용 가능, 참조 모델 R0).  n_keypoints 는 검출된 프레임에서만 모이므로 모델마다 다르다 — 데이터셋 속성이 아니다.

POSE_METRIC_BLOCKED: primary 로 쓸 6D pose metric 이 아직 없다.

모든 ST arm 은 EXPOSURE-MATCHED 다 — 같은 init, 같은 optimizer update 수,
같은 pseudo/synthetic 노출 수. 다른 것은 pseudo-label selection rule 뿐이다.
