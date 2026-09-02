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

## C) paired localisation drift — 점추정은 양수, 통계적으로는 미확립, 소수 프레임에 편중

같은 프레임·같은 supervised keypoint 로만 비교한 결과:

```text
domain      frames  R0 med   R5 med   Δframe med              CI95      p
daytime         69  10.462   11.576       +0.342  [-0.412, +0.814]  0.365
nighttime       42   7.686    8.943       +1.079  [-0.101, +2.633]  0.094
```

두 도메인 모두 **CI 가 0 을 포함한다.**  방향은 악화지만 이 표본으로는 확정되지
않는다.

분포를 보면 왜 그런지가 보인다:

```text
domain     worse  better    Δp50    Δp90     Δmax   상위5 몫
daytime       37      32   +0.34   +4.64   +202.8     87.5%
nighttime     27      15   +1.08   +8.46   +135.6     76.9%
```

daytime 은 악화 37 대 개선 32 로 거의 반반이고, 악화량 합계의 **87.5% 를 상위 5 장이
차지한다.**  즉 고른 열화가 아니라 **소수의 파국적 실패**다.

시각 확인(`A_WORSE_TOP20.jpg`): 대부분의 셀에서 R0(파랑)와 R5(빨강)가 거의 겹친다.
극단 사례는 팔레트 위에 다른 물체(파란 의자·분홍 조형물)가 올라가 가려진 프레임으로,
R0 5.2 px -> R5 190.7 px, R0 6.8 px -> R5 142.4 px 처럼 통째로 어긋난다.

**판정: 원인으로 지지되나 크기는 미확정.  daytime 회귀의 109%, nighttime 의 52.7% 를
점추정으로 차지하지만, 둘 다 CI 가 0 을 물고 소수 프레임에 몰려 있다.**
"self-training 이 위치추정을 전반적으로 망친다" 는 서술은 이 데이터로 지지되지
않는다.  "가림이 심한 소수 프레임에서 파국을 새로 만든다" 가 관측에 맞는다.

---

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
원인                      daytime            nighttime          성격
A raw px scale            0%                 0%                 도메인 간 해석의 교란
B detection selection     0%                 47.4%              지표가 모집단을 섞음
C paired drift            109% (CI 0 포함)   52.7% (CI 0 포함)  소수 파국에 편중
D pseudo-label noise      배경 조건          배경 조건          필터가 4.8% 만 제거
```

- **nighttime 회귀의 약 절반은 모델 열화가 아니다.**  R5 가 어려운 프레임을 새로
  검출해 분포에 넣은 결과다.
- **daytime 회귀는 전부 paired drift 로 귀속되지만 통계적으로 확립되지 않았고,
  악화량의 87.5% 가 5 장에서 나온다.**
- **A 는 R0/R5 비교의 원인이 아니다.**  다만 day/night 를 raw px 로 나란히 놓은
  기존 서술은 scale artefact 였다 — NME 로는 두 도메인이 같다.
- **D 가 배경 조건이다.**  기하 필터는 학습 집합을 4.8% 만 바꾸므로, 지금 구조에서
  Proposed 와 Confidence 를 가르는 실험은 검정력이 없다.

## 이 진단이 확정하지 못한 것

- C 의 크기.  frame-level CI 가 0 을 물고, 두 도메인 모두 표본이 42~69 장이다.
  같은 seed 문제로 replicate 도 검정력이 없다
  (`ultralytics-seed-does-not-reach-dataloader`).
- 파국 프레임의 원인.  `A_WORSE_TOP20.jpg` 의 극단 사례는 전부 적재물 가림이지만,
  n=5 라 가설이지 결론이 아니다.
