# 재학습 없이 얻을 수 있는 값 — selective prediction

작성 2026-09-06 · HEAD `2e5ec0e` · 새 추론 0 회
모집단 `PAPER_EVAL_ALL_POS` 319 · role = **DEV** · 아래 수치는 전부 **POST-HOC DIAGNOSTIC**

질문은 "모든 프레임을 더 정확하게" 가 아니라 **"틀릴 것 같은 결과를 거부할 수 있는가"** 다.
GT 는 곡선 평가에만 쓰고 runtime gate 입력으로 쓰지 않는다.

---

## 결론

```
DEPLOYABLE_REJECTION_SIGNAL_FOUND = WEAK
  최고 = kp_conf_min, AUROC 0.744.
  coverage 0.70 에서 gross 30.7% -> 21.1%. 실재하지만 변혁적이지 않다.
GEOMETRIC_SELF_CONSISTENCY_INFORMATIVE = NO  (AUROC 0.44~0.65)
```

실패(= identity 최대 코너 오차 > 25 px) 기저율 30.7%.

| 신호 | 추론 시점 사용 가능 | AUROC |
|---|---|---|
| `-box_conf` | 가능 | 0.697 |
| `bbox 종횡비` | 가능 | 0.569 |
| `bbox 대각` | 가능 | 0.561 |
| `candidate_count` | 가능 | 0.528 |
| `centroid_delta_px` | **불가 — GT 사용** | 0.862 |

[확인] `centroid_delta_px` 는 `scripts/paper/diagnose_axis_failures.py:127` 에서
`‖mean(pred[:8]) − mean(gt[:8])‖` 로 계산된다. **GT 중심을 쓰므로 배포 신호가 아니다.**
0.862 는 오라클이며 selective gate 근거로 쓸 수 없다.

---

## box_conf selective-risk curve

거부는 `box_conf` 낮은 순. coverage 를 숨기지 않고 함께 적는다.

```
coverage  conf 임계   accepted max px p50    p90    gross%   거부분 중 실패%
   1.00     0.0029              16.37      88.02    30.7%          —
   0.95     0.2602              15.78      75.95    29.0%       62.5%
   0.90     0.5202              14.98      68.28    26.5%       68.8%
   0.80     0.8405              14.88      51.03    25.1%       53.1%
   0.70     0.9050              14.88      51.73    23.8%       46.9%
   0.60     0.9337              14.92      42.24    23.0%       42.2%
   0.50     0.9499              14.37      41.72    20.0%       41.5%
```

[확인] **프레임의 절반을 버려도 gross 비율은 30.7% → 20.0% 로 10.7 %p 밖에 안 준다.**
median 은 16.37 → 14.37 px 로 12% 개선. 이건 배포로 쓸 만한 맞교환이 아니다.
거부한 쪽의 실패율이 41~69% 라는 것은 신호가 완전히 무의미하진 않다는 뜻이지만
(기저율 30.7% 보다는 높다), 정확도를 의미 있게 끌어올리기에는 분리력이 부족하다.

기존 판정과 일치한다 [확인] — `correct-box-bad-keypoint-is-the-bottleneck` 메모리의
"ranking·threshold·solver 셋 다 무력", `yolo-conf-threshold-is-not-the-lever`.

---

---

## 추가 계산 — 배포 가능 신호를 실제로 만들어 재봤다

앞 절의 "열린 항목" 을 그대로 두지 않고 계산했다.
`scripts/research/accuracy_root_cause_v1/selective_signals.py` · 새 추론 0 회.
입력은 `multiteacher_corner_distill_v1/predictions/T0_R0_YOLO26N_G38LEGACY.json`
— R0 의 319 프레임 예측 keypoint 9개 + keypoint confidence 9개가 그대로 들어 있다. [확인]

만든 신호는 전부 **예측만으로 계산** 된다(GT 없음):

```
kp_conf_min / mean          모델이 내는 keypoint 신뢰도
centroid 자기일관성          ‖pred kp8 − mean(pred kp0..7)‖ / bbox대각
공간대각 교점 산포            0-6, 1-7, 2-4, 3-5 네 대각의 쌍별 교점이 pred kp8 에서 벗어난 거리(중앙값)
면대각 교점 산포              앞면 0123 · 뒷면 4567 의 대각 교점이 각 면 중심에서 벗어난 거리
연결선 길이 변동계수          0-4, 1-5, 2-6, 3-7 네 연결선 길이의 std/mean
```

결과 (N=319, 실패 기저율 30.7%):

```
-kp_conf_min            AUROC 0.744    ← 최고
-kp_conf_corner_min     AUROC 0.744
-kp_conf_mean           AUROC 0.740
-box_conf               AUROC 0.697
공간대각 교점 산포          AUROC 0.649
연결선 길이 변동계수         AUROC 0.606
면대각 교점 산포           AUROC 0.585
centroid 자기일관성        AUROC 0.444    ← 0.5 미만. 역방향이다
```

[확인] **기하 자기일관성은 실패를 가리키지 못한다.** 네 신호 모두 0.44~0.65 로
모델이 스스로 내는 keypoint confidence(0.744)보다 나쁘고, centroid 자기일관성은
0.5 아래라 정보가 없다. 즉 R0 가 코너를 틀릴 때 **틀린 코너들끼리는 여전히 기하학적으로
일관된 육면체를 이룬다** — 구조가 무너지는 게 아니라 육면체 전체가 잘못된 자리에 놓인다.
이건 `corner-residual-systematic-not-noise` 메모리와 같은 방향이다.

`kp_conf_min` 으로 거부한 selective-risk curve:

```
coverage   accepted gross%   거부분 실패%
   1.00           30.7%           —
   0.90           28.2%        53.1%
   0.80           25.1%        53.1%
   0.70           21.1%        53.1%
   0.60           17.8%        50.0%
   0.50           14.4%        47.2%
```

box_conf 보다는 낫다(coverage 0.5 에서 gross 20.0% → **14.4%**). 그러나 여전히
**프레임의 30% 를 버려야 실패율이 30.7% → 21.1%** 이고, 절반을 버려야 반토막이 난다.
지게차가 프레임의 30~50% 에서 "모르겠다" 를 내는 것이 허용되는지는 배포 요구사항 문제이며,
이 감사는 그 요구사항을 모른다. 정확도 지표만 놓고 보면 **변혁적이지 않다.**

산출물 `data/pallet/results/accuracy_root_cause_v1/R0_SELECTIVE_SIGNALS.csv` (319행).

---

## 이 감사가 시험하지 못한 것

- ~~keypoint confidence~~ — 위 추가 계산에서 시험했다. AUROC 0.744.
- **교사 불일치**(multi-teacher disagreement). `multiteacher_corner_distill_v1/predictions/`
  에 9 arm × 319 프레임이 같은 `population_frame_order_sha256` 로 있으므로 계산 가능하지만,
  기존 판정이 이미 나와 있다 — 합의 필터는 **R0 가 이미 맞히는 자리를 고른다**
  (`MULTITEACHER_FINAL_REPORT.md`, memory `multiteacher-consensus-gives-confidence-not-labels`).
  거부 축(AUC 0.79)으로서의 재평가는 열려 있으나 이번 감사 범위 밖으로 둔다.
- ~~기하 자기일관성~~ — 위 추가 계산에서 시험했다. 네 종류 전부 정보 없음(0.44~0.65).
- **여러 신호의 결합**. 단일 신호만 봤다. `kp_conf_min` + `box_conf` + 기하 신호를
  학습된 결합기로 묶으면 0.744 를 넘을 수 있는지는 시험하지 않았다. [추정][미검증]
  다만 개별 AUROC 가 0.60~0.74 범위라 큰 도약은 기대하기 어렵다.
