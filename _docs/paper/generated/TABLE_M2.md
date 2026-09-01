# Table M2 — Target-domain adaptation under daytime and nighttime conditions

Daytime N=70, Nighttime N=50 (plastic only — morphology 를 lighting 효과와 섞지 않는다).

## primary — detection and ranking

MAIN Daytime 의 70 프레임은 전부 legacy 세션이고 keypoint visibility 가
unknown 이라 **supervision mask 가 비어 있다**. strict keypoint 오차를
그 조건에서 낼 수 없으므로, 두 조건 모두에서 계산 가능한 detection 과
ranking 을 primary 로 둔다.

```text
Method                              Day det↑  Night det↑    Mean↑   Worst↑   AUROC↑   FPR95↓
────────────────────────────────────────────────────────────────────────────────────────────
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
visibility 가 unknown 인 legacy 점까지 포함한다. **diagnostic 은
visible/occluded 주장이 아니다** — 두 열은 서로 다른 모집단이라 직접 비교하지 않는다.

```text
Method                              Day strict↓  Night strict↓  Day diag↓  Night diag↓  ALL strict↓
──────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only                                —          5.478     10.928        7.686        4.420
Source-only continuation                      —          5.525     10.588        7.964        4.352
Naive self-training                           —          5.715     12.023        8.440        4.335
Confidence-based self-training                —          5.341     12.447        9.601        4.242
Reprojection-based self-training              —          5.403     11.541        8.687        4.274
Proposed                                      —          5.271     11.592       10.072        4.180
```

Daytime strict keypoint 수 = 0 (UNAVAILABLE_METADATA).

POSE_METRIC_BLOCKED: primary 로 쓸 6D pose metric 이 아직 없다.

모든 ST arm 은 EXPOSURE-MATCHED 다 — 같은 init, 같은 optimizer update 수,
같은 pseudo/synthetic 노출 수. 다른 것은 pseudo-label selection rule 뿐이다.
