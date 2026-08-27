# RERANKING — top-1 confidence ranking 이 놓치는 몫

**PHASE 1 = `HEADROOM_SUFFICIENT_PROCEED`  ·  FINAL VERDICT = `RERANKING_SUPPORTED`**

질문 하나: *low-conf candidate list 안에 존재하는 correct pallet candidate 를
top-1 confidence ranking 이 놓치는 24 장을, GT 없이 reranking 해서 **실제 pose
까지** 회수할 headroom 이 있는가.*

한 줄 답: **box 는 회수되지만 pose 는 거의 회수되지 않는다.** 진짜 병목은
ranking 이 아니라 **correct box 인데 keypoint 가 나쁜 59 장(36.6%)** 이다.

새 학습 0 · 새 architecture 0 · 재추론 0 (`_cc_raw_dump.json` = epoch60
checkpoint 의 conf=0.001 전수 후보 캐시 재사용).

---

## PHASE 1 — TOP-K ORACLE HEADROOM

★ oracle 은 deployment result 가 아니다. reranking 가치의 **상한**이다.
선택 규칙은 사전등록: top-K 중 **R 오차 최소** candidate.
`5cm5(any)` 는 top-K 중 아무거나 5cm5 성공 = 5cm5 의 **진짜 상한**.

```
  K    avail  corr_rec    R med    R p90    t med    t p90  5cm5(oR)   5cm5(any)
────────────────────────────────────────────────────────────────────────────────
  1    1.000      0.708     4.33    57.03   0.1267   8.2848      0.304       0.304
  2    1.000      0.752     3.60    22.18   0.1361   9.9961      0.298       0.304
  3    1.000      0.789     3.24    21.90   0.1577   9.4759      0.304       0.311
  5    1.000      0.826     3.11    19.09   0.1899   8.8535      0.311       0.317
```

```
Top1 -> Top5   correct recall  +11.80pp   (STOP 기준 5pp)
               5cm5 (any)      +1.24pp   (STOP 기준 3pp)
```

STOP RULE 은 **AND** 라 발동하지 않았다 → `HEADROOM_SUFFICIENT_PROCEED`.

그러나 두 수의 **비대칭이 이 실험의 답**이다. K 를 1→5 로 늘리면
correct box 는 **+11.8pp** 오르는데 5cm5 상한은 **+1.24pp**(161 장 중 2 장)
밖에 안 오른다. 후보 목록 안에 "맞는 상자" 는 더 있지만 "쓸 만한 pose" 는 없다.

---

## PHASE 2 — CORRECT BOX BUT BAD POSE ★ 이 분류가 핵심

`usable pose` 정의(사전등록): **IoU≥0.5 AND R≤10° AND t≤0.10m**.

```
분류                             n     비율    population
──────────────────────────────────────────────────────────────
TOP1_ALREADY_GOOD              65   40.4%   OPEN 40 / CH 25
A_GOOD_CANDIDATE_MISRANKED      9    5.6%   OPEN  2 / CH  7
B_CORRECT_BOX_BAD_KP           59   36.6%   OPEN 11 / CH 48
C_NO_CORRECT_CANDIDATE         28   17.4%   OPEN  3 / CH 25
```

- **A = reranking 이 가져갈 수 있는 전부** — correct box + usable pose 가
  후보에 있는데 top-1 이 아닌 프레임. **9 장(5.6%)** 뿐이다.
- **B 가 압도적이다 — 59 장(36.6%).** correct box 는 있는데 **어떤 후보의
  pose 도 쓸 수 없다.** reranking 으로는 원리적으로 손댈 수 없다.
  이건 **keypoint 품질** 문제다.
- B 는 CHALLENGE 에 48 / OPEN 에 11 로 몰려 있다.

브리프의 출발점이던 "24 장" 은 **box 기준**이었다. pose 까지 따지면 상금은
**9 장**이다.

---

## PHASE 3 — DEPLOYMENT-AVAILABLE RERANK FEATURES

learned reranker 금지. feature 를 **하나씩** 쟀고, 결과를 보고 조합하지 않았다.
전부 GT 미사용.

```
feature              AUC(box)  MRR(box)   AUC(use)  MRR(use)
─────────────────────────────────────────────────────────────
box_area                0.9800   0.9664     0.9088   0.9146
box_diag                0.9901   0.9889     0.9043   0.9034
kp_conf_mean            0.6767   0.7729     0.8215   0.8378
box_conf                0.7248   0.8556     0.8182   0.8548
kp_conf_min             0.6058   0.7433     0.8147   0.8209
neg_pnp_reproj          0.3454   0.6386     0.6438   0.6798
cuboid_plausible        0.2873   0.8556     0.1926   0.8548
depth_valid             0.2752   0.8556     0.1818   0.8548
```

세 가지가 나왔다.

1. **`box_area` / `box_diag` 가 `box_conf` 를 크게 앞선다** (AUC 0.98/0.99 vs
   0.72). 모델의 confidence 는 *어느 상자가 팔레트인지* 를 잘 못 가른다 —
   D1 confidence calibration 문제와 같은 방향이다.
2. **`neg_pnp_reproj` 는 무작위보다 나쁘다**(AUC 0.345). 직전 조건부 Hough
   실험에서 reproj 의 gate AUC 가 0.597 이었던 것과 **독립적인 두 번째 확증**이다.
   reproj 는 이 모델에서 신뢰할 신호가 아니다.
3. **`depth_valid` / `cuboid_plausible` 은 강하게 역상관**(AUC 0.18~0.29).
   기하 sanity check 를 걸면 **오히려 정답을 떨어뜨린다.** 필터로 쓰지 말 것.
   (두 feature 의 MRR 이 `box_conf` 와 정확히 같은 것은 이진값이라 동점이
   원래 conf 순서로 깨지기 때문이다 — MRR 은 이 둘에 대해 무의미하다.)

---

## PHASE 4 — SESSION LOSO

held-out session 을 빼고 나머지 6 개에서 MRR(usable) 최고 feature 를 고른 뒤
held-out 에서만 평가. 7 회.

```
held out    feature       nat rec   rr rec   nat 5cm5   rr 5cm5
────────────────────────────────────────────────────────────────
cad         box_area        0.864    1.000      0.818     0.818
night08     box_area        0.353    0.588      0.176     0.176
night09     box_area        0.280    0.400      0.040     0.040
noapril     box_area        1.000    1.000      0.917     0.917
outside     box_diag        0.727    0.864      0.182     0.182
pallet07    box_area        1.000    1.000      0.407     0.444
pallet09    box_area        0.750    0.861      0.028     0.028
```

7/7 fold 에서 `box_area`(또는 `box_diag`)가 선택됐다 — feature 선택이 session
에 흔들리지 않는다.

### native confidence top-1 vs reranked top-1 (CV 예측, n=161)

```
지표             native    reranked     변화
──────────────────────────────────────────────
recall           0.708      0.814     +10.56pp
5cm5             0.304      0.311     +0.62pp
R median (deg)    4.33       4.33      +0.00
t median (m)     0.1267     0.1114
```

population 별:
```
population                  n   native recall   rerank recall   native 5cm5   rerank 5cm5
──────────────────────────────────────────────────────────────────────────────────────────
OPEN_56                    56          0.839           0.946         0.589         0.589
CHALLENGE_DEV_105         105          0.638           0.743         0.152         0.162
```

사전등록 게이트(recall gain ≥ +5pp AND 5cm5 악화 ≤ 0 AND R median 악화 ≤ 0)를
모두 통과 → **`RERANKING_SUPPORTED`**.

---

## ★ 그러나 배포에서 무엇을 사는가 — 반드시 같이 읽을 것

판정은 사전등록 게이트대로 두되, 게이트가 recall 중심이었으므로 pose 로
환산해 그대로 적는다. **rerank 는 후보가 2 개 이상일 때만 작동한다.**

```
    conf   n_cand   multi   nat rec   rr rec   +장   nat use   rr use   +장
──────────────────────────────────────────────────────────────────────────────
   0.001        4     0.83     0.708    0.814    17     0.404    0.422     3
    0.01        2     0.62     0.702    0.795    15     0.404    0.410     1
    0.05        1     0.38     0.689    0.727     6     0.404    0.416     2
     0.1        1     0.25     0.689    0.714     4     0.404    0.416     2
     0.2        1     0.16     0.640    0.658     3     0.391    0.404     2
     0.4        1     0.06     0.584    0.590     1     0.385    0.391     1
```

- 현재 운영점 **conf=0.40 에서는 후보가 2 개 이상인 프레임이 6% 뿐**이라
  rerank 가 사는 것은 **box +1 장, usable pose +1 장**이다.
- rerank 의 큰 이득(+17 장)은 **conf=0.001 에서만** 나오는데, 직전 실험에서
  그 운영점은 pose 를 파괴한다는 것이 확인됐다(회수분 t median 31 배).
  실제로 conf=0.001+rerank 의 usable 은 **0.422**, 현재 운영점(conf=0.40
  native)의 **0.385** 대비 **+3.7pp = 6 장**이다 — 대신 negative FP 발생
  프레임이 **20.1% → 87.3%**(하한)로 간다.

### negative 부작용 (secondary — 편향 표본이라 참고만)

```
    conf    FP 프레임   FP/image
─────────────────────────────────────────
   0.001       0.873      5.510   rerank 무관
    0.05       0.649      1.004   rerank 무관
     0.4       0.201      0.208   rerank 무관
```

**rerank 는 negative FP 를 바꾸지 않는다.** threshold 를 먼저 적용하므로
살아남는 박스 개수가 같고, 어느 것을 보고할지만 달라진다. 이건 rerank 의
장점이다 — 공짜로 얻는다.

> 계산 정정: 첫 판에서 negative 쪽만 "전체에서 가장 큰 박스를 고른 뒤 conf 확인"
> 으로 계산해 positive 와 계약이 어긋났다(FP 0.649→0.375 처럼 보였다).
> threshold 를 먼저 적용하도록 바로잡았고, 올바른 답은 "rerank 무관" 이다.

---

## VERDICT

**`RERANKING_SUPPORTED`** — 사전등록 게이트 기준.

정확히 무엇이 지지됐는지 좁혀 적는다:

```
지지됨      correct box 를 top-1 으로 올리는 능력 (recall +10.56pp)
            negative FP 를 늘리지 않음
            feature 선택이 7/7 session 에서 안정 (box_area)
지지 안 됨   pose 품질 개선 (5cm5 +0.62pp = 161 장 중 1 장,
            R median 변화 +0.00)
            현재 운영점(conf=0.40)에서의 실효 이득 (box +1, pose +1)
```

**RANKING 축은 여기서 닫는다.** 상한 자체가 9 장이고, 그 중 배포 가능한 규칙으로
얻는 건 1~6 장이다. 다음 레버가 아니다.

## 이 실험이 가리키는 진짜 다음 축

**B_CORRECT_BOX_BAD_KP 59 장(36.6%)** — 상자는 맞는데 keypoint 가 나쁘다.
ranking 도 threshold 도 손댈 수 없고, solver 도 아니다(입력 점이 나쁘다).
이건 **keypoint 품질 = 데이터 축**이고, 이미 진행 중인 `BROAD_FAMILY_V2` 와
§9 `LOW_ANGLE_ROLE_DISAMBIGUATION_COVERAGE` 가 정확히 그 축이다.
→ **BROAD_FAMILY_V2 유지. 수정 없음.**

## 적용 범위 (넘어서 주장하지 말 것)

- `REAL_DEV_POS` 161 장(OPEN56 + CHALLENGE105), **DEV** 다. final test 아님.
- oracle 은 진단용 상한이며 **deployment result 가 아니다**.
- `box_area` 는 "이 장면에 팔레트가 하나이고 그게 가장 큰 물체" 라는 가정에
  올라타 있다. 팔레트가 여럿이거나 원거리 팔레트가 목표인 장면에서는
  **검증되지 않았다** `[미검증]`.
- negative FP 수치는 `REAL_NEG_DEV_V1` 편향 표본 기준이라 **하한**이다.

## 산출물

`RERANK_ORACLE.json` · `RERANK_FEATURES.json` · `RERANK_DEPLOY_COUPLING.json` ·
`RERANK_PER_FRAME.csv` · `plots/{TOPK_ORACLE,FEATURE_RANKING,RERANK_COUPLING}.png` ·
스크립트 `rr_oracle.py` · `rr_features.py` · `rr_deploy_check.py` · `rr_plots.py`
