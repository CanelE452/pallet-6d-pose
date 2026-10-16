# Utility-aware conditional self-training (utility_selftrain_v1)

지시문 원문 `_docs/experiments/utility_selftrain_v1/INSTRUCTION_AS_RECEIVED.txt`.
착수 전 감사 `_docs/experiments/utility_selftrain_v1/PREFLIGHT_AUDIT.md`.

## 1. 제안

**가설** — pseudo-label 을 "GT 에 얼마나 가까운가"(correctness)로 고르는 대신
"그것으로 학습했을 때 학생이 실제로 좋아지는가"(training utility)로 고르면,
GT-free 로 관측 가능한 조건 중 utility-positive 인 레짐이 존재한다.

**방법** — 무라벨 adaptation pool 을 GT-free 조건(신뢰도 · 기하 일관성 · 투영크기 ·
예측 앙각)으로 subset 을 나누고, 각 subset 으로 **동일 조건 micro fine-tuning** 한 뒤
disjoint labeled session 에서 학생 성능을 잰다. utility-positive 조건이 나오면
해석 가능한 GT-free 규칙(depth<=3 결정트리 / 단조 규칙목록 / 로지스틱)으로 회수하고
freeze 후 STUDENT_EVAL 에서 확인한다.

**판정 지표** — 학생의 held-out 성능. corner median/p90 (primary),
translation·depth median/p90, detection. 기준 대조는 U0 RANDOM_MATCHED 와
U1 CURRENT_F4 둘 다. gate 는 `METHOD_LOCK_UTILITY_ST.json` 에 결과 보기 전 확정.

**예상 실패 모드**
- arm 표본 붕괴: 등N 규칙이 U7(32장)에 묶여 모든 arm 이 32장이 된다.
  노출 슬롯 14,400 고정이라 장당 450회 반복 = 암기 레짐.
- 표집 잡음이 효과보다 크다: 동일 arm draw 간 translation median 14.3%,
  corner p90 17.5~55.6% 산포가 이미 측정돼 있다.
- 선행 반복: B_* 대조군이 "confidence 통과 pool 안에서 geometry 선별 ≡ 무작위" 를,
  V5 가 "자르지 않고 신뢰도 가중" 실패를 이미 보였다.

**중단 기준** — 지시문 §21 그대로. 추가로: utility-positive subset 이
독립 draw 3개에서 방향 일관되지 않으면 `NO_USEFUL_PSEUDOLABEL_REGIME_FOUND`.

**판정 등급 상한** — `DEVELOPMENT_ONLY`.
`EXPERIMENT_STOP_LOCK.json` 재개 조건 1(미사용 평가 모집단)이 저장소에 없고,
`PAPER_CLAIM_LOCK.json` 이 PAPER_EVAL 을 `held_out_final: false` 로 못박았다.
논문 final claim 을 만들지 않는다.

## 2. 결과

### 2.1 착수 전 감사 (학습 0회)

`_docs/experiments/utility_selftrain_v1/PREFLIGHT_AUDIT.md` 참조. 요지:
F4 는 reprojection 을 쓰지 않는다(conf+remove+flip, 6 arm 전수 재현) ·
pool↔eval 누수 0(sha256 전수) · pool 은 1초 간격 유효 프레임 648장이라 확대 무의미 ·
표집 draw 잡음이 지시문 예시 gate 5% 보다 크다 · 트랙에 종료 선언이 있어 판정 상한은
DEVELOPMENT_ONLY.

### 2.2 §4 분할 (SPLIT_CONTRACT.json)

```
POLICY_DEV    91  (3 sess)  plastic 66 / wood 25   day 44 night 22 unknown 25
POLICY_VAL    96  (4 sess)  plastic 40 / wood 56   day 40 night 56
STUDENT_EVAL 132  (6 sess)  plastic 88 / wood 44   day 84 night 28 unknown 20
```
프레임·세션 교집합 전부 0. STUDENT_EVAL 은 기존 `metric_split_lock.md` §1.6 의
frozen final_test 4 세션을 그대로 받아 §12 까지 열지 않는다.

### 2.3 §2 baseline — 지시문이 지목한 네 셀은 실재하나 얇다

POLICY_DEV(66 plastic 기록) 기준 F4: 통과 57 중 오답 8, 기각 9 중 정답 5.
`high_confidence_but_wrong` 11 · `low_confidence_but_geometry_consistent_and_accurate` 4 ·
`correct_but_rejected` 5 · `wrong_but_accepted` 8.
POLICY_VAL 에서는 필터가 거의 작동하지 않는다 (F1/F3/F5/F4 coverage 가 전부 0.875 로 동일).

### 2.4 §5 purity map (POLICY_DEV n=91) — 신뢰도는 신호가 아니고, 기하와 근접이 신호다

| feature | 방향 | Q1 정확도 | Q4 정확도 |
|---|---|---|---|
| `max_geom_f4` | 단조 | 1.000 (gross 0.000) | 0.208 (gross 0.792) |
| `kp_conf_min` | 단조 | 0.227 (gross 0.773) | 0.750 |
| `bbox_diag_frac` | 역단조 | 0.818 | 0.292 (gross 0.708) |
| `pred_depth_m` | 역단조 | 0.273 (가장 가까움) | 0.792 |
| `box_conf` | **없음** | — | 0.625~0.750 로 평평 |
| `flip_box_iou` | 비단조 | 0.682 | 0.542 |

`box_conf` 는 91장 중 79장이 0.90 이상이라 구간이 사실상 하나다 — 지시문 §15 의
"confidence 하나로 최종 선택을 만들지 않는다" 전제가 데이터로 확인된다.
투영이 **클수록** 나쁘다(근접 왜곡). memory `stage16-v8lt-failure-distribution` 과 정합.

⚠ POLICY_DEV 의 예측 앙각 사분위는 12.2 / 19.0 / 31.0 도인데 **pool 은 중앙값 1.63도**다.
labeled 평가 프레임과 무라벨 pool 의 시점 분포가 크게 다르다 — arm 을 pool 앙각으로
정의하고 labeled 에서 재는 구조 자체에 도메인 격차가 있다.

### 2.5 §6 threshold sweep (POLICY_DEV) — 현재 임계는 순도 기준으로 매우 느슨하다

```
tau   accepted coverage purity          conf  accepted coverage purity
0.01     14     0.154    1.000          0.70     87     0.956    0.690
0.02     47     0.516    0.915          0.80     83     0.912    0.687
0.03     67     0.736    0.791          0.85     79     0.868    0.709
0.05     79     0.868    0.709          0.95     66     0.725    0.727
```
기하 임계는 순도를 크게 움직이고 신뢰도 임계는 거의 못 움직인다.
(§6 규정대로 이 표에서 최고 순도 임계를 정책으로 채택하지 않는다.)

### 2.6 ★ §7 arm 구성 — 핵심 셀이 pool 에 없다

사전등록한 분위수 규칙으로 pool candidate 926 을 나눈 결과:

```
U0_RANDOM_MATCHED            926
U1_CURRENT_F4                259
U3_LARGE_SCALE               192
U5_HIGH_ELEV                 149
U4_LOW_ELEV                   81
U6_LOWCONF_STRONGGEOM         76
U2_SMALL_SCALE                43
U7_HIGHCONF_BORDERLINEGEOM     7   <- 구성 불가
```

원인은 표집 운이 아니라 구조다. pool 에서 `box_conf` 와 `max_geom_f4` 의
**Spearman 상관이 −0.714** 다. 4x4 분위수 결합분포에서 비대각 셀은

```
confQ4 x geomQ4 (고신뢰·약기하) = 5장  (독립 기대의 0.09배)
confQ1 x geomQ1 (저신뢰·강기하) = 4장  (독립 기대의 0.07배)
```

즉 지시문 §15 가 "이번 실험의 핵심" 이라고 한 confidence x geometry 상호작용은
**이 pool 에 비대각 질량이 거의 없다.** U7 은 학습 없이 census 로 구성 불가 판정.


### 2.7 ★ §7 Stage C 판정 — NO_USEFUL_PSEUDOLABEL_REGIME_FOUND

matched N=43, arm 7개 x pseudo-draw 3개 = 21 run (22초/epoch, 10 epoch, 900 update).
평가는 POLICY_VAL 96프레임의 풀링된 supervised keypoint 오차.

```
subset                    cornerMed  draw폭%    p90    det    dU0%   dF4%
R0 (self-training 안 함)      5.183      --   17.64  0.990      --     --
U0_RANDOM_MATCHED            5.444     4.8   18.86  0.976      --     --
U1_CURRENT_F4                5.380     3.1   20.12  0.979      --     --
U2_SMALL_SCALE               5.133     2.4   18.52  0.969    +5.7   +4.6
U3_LARGE_SCALE               5.119     2.9   18.25  0.990    +6.0   +4.8
U4_LOW_ELEV                  5.460     3.5   18.51  0.979    -0.3   -1.5
U5_HIGH_ELEV                 5.262     7.2   19.70  0.983    +3.4   +2.2
U6_LOWCONF_STRONGGEOM        5.393     1.3   20.67  0.986    +0.9   -0.2
U7_HIGHCONF_BORDERLINEGEOM      구성 불가 (pool 926 중 7장)
```

**arm 간 평균 폭 6.7% 가 arm 내부 draw 폭(중앙값 3.1%, 최대 7.2%)과 같은 크기다.**
사전등록 gate 15% 를 넘은 arm 은 없다. 그리고 **학습하지 않은 R0 가 p90·검출에서 전 arm 보다
낫고 corner median 도 최상위권**이다 — 가장 나은 U3 가 R0 대비 +1.2% 로 draw 폭 안이다.
세션별 방향도 갈린다 (U2 는 eval_outside −12.0%, U4 는 −22.6%).

§21 STOP RULE 발동 → **SELFTRAIN_UTILITY_TRACK_CLOSED**.
§10 Stage D(정책 설계)·§12 최종 비교 미진입, **STUDENT_EVAL 132프레임 미개봉**.
전체 보고는 `_docs/experiments/utility_selftrain_v1/FINAL_REPORT.md`.

말할 수 있는 것 / 없는 것은 그 보고의 WHAT THIS SUPPORTS / DOES NOT SUPPORT 절 참조.
요지: 순도는 기하로 조절되지만(tau 0.05→0.01 에서 0.709→1.000) 학생으로 전이되지 않는다.
"원리적 불가"도 "self-training 이 해롭다"도 주장하지 않는다 — draw 폭 안에서 겹친다.
