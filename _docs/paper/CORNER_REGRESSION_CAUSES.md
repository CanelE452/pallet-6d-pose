# corner 회귀의 원인 분해 — A/B/C/D

진단 전용이다.  이 문서를 근거로 threshold·tau·pseudo pool·model 을 바꾸지 않았고,
바꾸지 않는다.

수치 출처는 전부 생성물이다 — `_docs/paper/generated/REGRESSION_DIAGNOSIS.md`,
`.../VISUAL_AUDIT.md`, `data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json`.
여기서는 그 수치를 네 가설에 배분한다.

## 확인한 사실 (§1)

corner 는 **원본 이미지 좌표계의 Euclidean px** 다.  예측은 추론 padding 을
`- INFERENCE_PAD` 로 되돌린 뒤 GT `keypoint_annotations[].xy` 와 비교한다
(`challenge/evaluation_v2/paper_real_eval.py:2409`).  리사이즈 배율이 개입하지 않는다.
생성 표의 열 이름에 `[px]` 를 명시했다.

## 설명해야 할 것

```text
subgroup      R0      R5    Δ
ALL         6.616   7.210  +0.594
Daytime    10.556  11.576  +1.020
Nighttime   7.686  10.072  +2.386
```

---

## A) raw px scale effect — 도메인 간 비교에서는 결정적, R0/R5 차이는 설명 못 함

투영된 물체 크기가 도메인마다 다르다.

```text
domain   cuboid diagonal med   p90
daytime              420.6   540.2
nighttime            326.8   489.3
```

daytime 물체가 중앙값으로 **29% 크다**.  그래서 raw px 로 읽으면 daytime 이 더
나빠 보인다 (R0 10.556 대 7.686, +37%).  같은 데이터를 NME 로 보면:

```text
              raw px   NME(cuboid)
R0 daytime    10.556        0.0261
R0 nighttime   7.686        0.0262
```

**day/night 격차가 사라진다 (+37% -> +0.4%).**  즉 "daytime 이 nighttime 보다
나쁘다" 는 읽기는 scale artefact 다.

그러나 R0 대 R5 는 **같은 프레임·같은 물체 크기** 위에서 비교된다.  분모가 같으므로
scale 이 그 차이를 만들 수 없고, 실제로 NME 로도 남는다:

```text
domain      R0 NME   R5 NME   상대변화   (BOTH_DETECTED)
daytime     0.0260   0.0284     +9.2%
nighttime   0.0262   0.0286     +9.2%
```

**판정: R0->R5 회귀의 원인으로는 지지되지 않는다(0%).  도메인 간 해석의 교란으로는
강하게 지지된다.**  두 도메인의 상대 열화가 9.2% 로 정확히 같다는 점은, raw px 에서
보이던 "야간이 훨씬 더 나빠졌다" 도 상당 부분 scale·selection 이었다는 뜻이다.

---

## B) detection-selection effect — nighttime 의 절반, daytime 은 0

두 모델의 raw 수치는 **서로 다른 프레임 집합** 위에서 계산된다.  검출된 프레임에서만
오차가 모이기 때문이다.

```text
domain       N  BOTH_DET  R0_ONLY  R5_ONLY  BOTH_MISS
daytime     70        69        1        0          0
nighttime   50        42        0        6          2
```

nighttime 에서 R5 는 R0 가 놓친 6 장을 새로 잡는다.  그 6 장은 어렵다 — 회수분의
pooled keypoint median 이 **22.2 px** 로, R5 가 나머지에서 내는 8.9 px 의 2.5 배다.

정확히 더해지는 분해(px):

```text
domain       total   base 모집단   localisation   selection
daytime     +1.020      -0.094         +1.114      +0.000
nighttime   +2.386      +0.000         +1.257      +1.130
```

**판정: nighttime 회귀의 47.4% 가 selection 이다.  daytime 은 0%.**
검출이 좋아진 것(det 0.975 -> 0.984)이 corner 지표를 나쁘게 만드는 구조다 —
회수된 어려운 프레임이 분포에 들어오기 때문이다.  이건 모델이 나빠진 게 아니라
**지표가 두 모집단을 섞은 것**이다.

시각 확인(`C_NIGHT_R5_ONLY.jpg`): 그 6 장에서 R0 의 top-1 은 팔레트가 아니라
**트래픽 콘**에 붙는다.  R5 는 팔레트를 찾는다.  R0 는 "검출 실패" 가 아니라
"다른 물체를 골랐다".

---

## C) paired localisation drift — 위치 이동이 아니라 **90도 축 배정 실패**다

### 초판 정정

초판에서 이 절을 "가림이 심한 소수 프레임에서 파국을 새로 만든다" 로 썼다.
**틀렸다.**  근거는 contact sheet 를 훑어본 인상이었고, 측정하니 다른 것이 나왔다.

파국 프레임의 예측 코너를 GT 에 자유 배정(Hungarian)하면 최대 오차가 13~22 px 다.
큐보이드는 **제자리에 있다**.  그리고 세 프레임이 독립적으로 **같은 순열**을 냈다:

```text
gt <- pred  [1, 5, 6, 2, 0, 4, 7, 3]
```

규약(0~3 근면 · {0,1,4,5} 위 · flip pair (0,1)(3,2)(4,5)(7,6))으로 풀면 수직 모서리가
`near-left -> far-left -> far-right -> near-right -> near-left` 로 한 칸 돈다 —
**수직축 90도 회전**이다.

중간에 한 번 더 틀렸다.  median 으로 재배정 오차를 재니 "좌우 뒤집힘" 으로 보였는데,
90 도 순열은 8 코너 중 절반만 맞히므로 오차가 이봉분포가 되고 median 이 작은 쪽
봉우리에 앉는다(예: `[2.8, 4.9, 5.5, 5.9, 5.9, 376.7, 380.8, 386.3, 386.4]` 의 median
은 5.9).  판정은 **최대 오차**로 해야 한다.

### 모집단 통계

```text
model        frames     OK   AXIS_PERMUTED   MISLOCATED      yaw90  yaw270  yaw180  mirror
R0              319    221              15           83          8       7       0       0
R2_CONF         319    203              17           99         11       6       0       0
R5_PROPOSED     319    199              16          104         10       6       0       0
```

**순열은 전부 yaw90 또는 yaw270 이다.  yaw180 과 거울은 0 건이다.**
(`MISLOCATED` 는 최대 오차 기준이라 "코너 하나가 25 px 넘게 어긋남" 을 포함한다 —
박스가 틀렸다는 뜻이 아니다.)

R5 는 3 장을 새로 어긋내고 2 장을 고친다.  **새로 어긋난 3 장은 R2_CONF 에서도 똑같이
어긋난다** — 기하 필터 탓이 아니라 self-training 공통이다.

```text
eval_pallet07:1778652138515809024   identity max 287.7 -> yaw90 max 22.0   centroidΔ 2.5
eval_pallet09:1778653664407620608   identity max 207.8 -> yaw90 max 13.0   centroidΔ 4.0
eval_night09:1779449631842893312    identity max 156.0 -> yaw90 max 17.7   centroidΔ 5.5
```

### 왜 90 도인가 — 투영이 정사각에 가까울 때

근면 폭과 측면 깊이의 min/max 비(1.0 이면 투영이 정사각):

```text
verdict            n   median     p90
OK               199    0.500   0.886
AXIS_PERMUTED     16    0.870   0.960
MISLOCATED       104    0.291   0.840
파국 3 장                0.797 / 0.955 / 0.964
```

물리 팔레트가 110x130 mm(비 0.846)라, 시점에 따라 두 변이 거의 같게 투영되면 90 도
회전이 **시각적 대칭**이 된다.  그때 축을 가를 단서가 사라진다.

### 왜 R5 만 — pool 이 그 시점을 담고 있지 않다

```text
투영 정사각성            n     median   >0.75 비율
EVAL (GT)              319      0.476       21.6%
adaptation pool 전체   954      0.200        3.6%
Confidence 통과        263      0.215        3.4%
Proposed 통과          250      0.211        1.6%
```

평가셋의 **21.6%** 가 정사각에 가까운 시점인데 pool 은 **3.6%** 뿐이고, 기하 필터가
다시 절반으로 줄여 **1.6%** 로 만든다.  self-training 은 pool 이 대표하는 시점
(길쭉한 측면 뷰)으로 모델을 끌고 가고, 평가셋에만 있는 정사각 시점에서 축 배정이
무너진다.  **시점 covariate shift** 다.

이 지표는 min/max 비라 90 도 순열에 **불변**이므로, pool 값을 teacher 예측으로
재도 축 오류에 오염되지 않는다.

기하 필터의 `s_flip` 은 이 모호성을 실제로 감지한다 — 파국 3 장에서 R0 의 s_flip 이
0.4827 / 0.5053 / 0.5168 로 `tau_flip = 0.05` 의 열 배다.  다만 그 판정은 pool 의
pseudo-label 을 거를 때만 쓰이므로, 학생이 평가 시점에 같은 실수를 하는 것을 막지
못한다.  오히려 그런 프레임을 학습에서 더 지운다.

### 축 순열을 빼면 C 가 얼마나 남나

```text
domain     n_all    Δ_all              CI_all   n_clean  Δ_clean            CI_clean  제외
daytime       69   +0.342   [-0.41, +0.81]          65   +0.056   [-0.50, +0.81]     4
nighttime     42   +1.079   [-0.16, +2.63]          40   +1.079   [-0.10, +2.63]     2
```

**daytime 의 paired drift 는 사실상 전부 축 순열이었다** (+0.342 -> +0.056).
nighttime 은 그대로 남는다 (+1.079) — 야간 열화는 축 문제가 아니다.

pooled keypoint median 으로도 같다:

```text
domain      R0      R5       Δ      n_kp   (축 순열 프레임 제외)
daytime    10.152  10.634  +0.482    565
nighttime   7.576   8.607  +1.030    360
```

**판정: daytime 은 축 배정 실패 3 장이 원인이고, 그 밖의 위치 열화는 없다
(+0.056 px).  nighttime 은 축과 무관한 열화가 남지만 CI 가 0 을 문다.**

## D) pseudo-label noise — 실재하고, 필터가 거의 걸러내지 못한다

M4(plastic 194 장, teacher = R0):

```text
Proposed PASS  이고 gross > 20 px   58 장
Proposed REJECT 이고 gross > 20 px   34 장
```

gross 를 가진 92 장 중 필터가 잡는 것은 **34 장(37%)** 이다.  나머지 58 장은 통과해
학습 라벨이 된다.

U_MAIN 실제 pool(각 500 장)에서 필터가 하는 일은 더 작다:

```text
condition   pool   Confidence 통과   Proposed 통과   기하로 제거된 수
daytime      500              123             120                3
nighttime    500              149             139               10
```

**confidence 가 통과시킨 272 장 중 기하 필터가 버리는 것은 13 장(4.8%)이다.**
학습 집합이 사실상 바뀌지 않으므로 R2_CONF(7.037)와 R5_PROPOSED(7.210)가 비슷한 것은
당연하다.  M4 의 separation 도 confidence 단독 7.17 이 Proposed 6.26 보다 크다.

시각 확인(`H_POOL_CONF_ONLY_DAYTIME.jpg`): 기하가 버린 daytime 3 장은 전부
`s_remove > 0.05` 로 걸렸는데, 육안으로는 keypoint 가 팔레트에 맞아 보인다 — 저앙각·
잘림 프레임이다.  버린 것이 명백한 오라벨이라는 증거는 이 3 장에서 보이지 않는다.

**판정: 강하게 지지된다.  다만 "필터가 나쁜 라벨을 통과시킨다" 가 아니라
"필터가 거의 아무것도 하지 않는다" 가 정확하다.**  회귀의 원인은 기하 필터의 오작동이
아니라, **confidence 통과분에 이미 들어 있는 gross 라벨**이다.

---

## 종합

```text
원인                        daytime          nighttime        성격
A raw px scale                   0%                 0%        도메인 간 해석의 교란
B detection selection            0%              47.4%        지표가 모집단을 섞음
C1 축 배정 실패 (90도)         103%              10~15%        정사각 시점, pool 미대표
C2 그 밖의 위치 열화             6%              47~53%        CI 0 포함, 미확립
D pseudo-label noise       배경 조건          배경 조건        필터가 4.8% 만 제거
```

- **nighttime 회귀의 약 절반은 모델 열화가 아니다.**  R5 가 어려운 프레임을 새로
  검출해 분포에 넣은 결과다.
- **daytime 회귀는 90 도 축 배정 실패 3 장이 전부다.**  그 프레임을 빼면 남는 위치
  열화는 +0.056 px 로 사실상 0 이다.  원인은 가림이 아니라 **정사각에 가까운 투영**
  이고, adaptation pool 이 그 시점을 3.6% 밖에 담고 있지 않다(평가셋은 21.6%).
- **A 는 R0/R5 비교의 원인이 아니다.**  다만 day/night 를 raw px 로 나란히 놓은
  기존 서술은 scale artefact 였다 — NME 로는 두 도메인이 같다.
- **D 가 배경 조건이다.**  기하 필터는 학습 집합을 4.8% 만 바꾸므로, 지금 구조에서
  Proposed 와 Confidence 를 가르는 실험은 검정력이 없다.

## 이 진단이 확정하지 못한 것

- nighttime 의 C2(축과 무관한 열화 +1.079 px).  CI 가 0 을 물고 표본이 40 장이다.
  같은 seed 문제로 replicate 도 검정력이 없다
  (`ultralytics-seed-does-not-reach-dataloader`).
- 축 실패의 표본이 작다.  R5 에서 16 장, 새로 생긴 것은 3 장이다.  "pool 이 정사각
  시점을 안 담아서" 는 분포 차이(21.6% 대 3.6%)로 뒷받침되지만, 인과는 그 시점을
  넣어 다시 학습해야 확정된다 — 이 진단에서는 하지 않는다.
- `MISLOCATED` 83~104 장의 내용.  최대 오차 기준이라 "코너 하나가 25 px 초과" 를
  전부 포함한다.  그 안에 또 다른 구조적 실패가 있는지는 보지 않았다.
