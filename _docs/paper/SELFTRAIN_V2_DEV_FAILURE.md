# V2 development — FAILED, 그리고 무엇이 실제로 움직였나

`V2_METHOD_STATUS = FAILED`.  §20 에 따라 threshold·q·LR·pseudo fraction 을 sweep
하지 않았고, 손대지 않은 confirmation set 을 열지 않았다.

수치는 전부 `data/pallet/results/paper_selftrain_v2/V2_DEV_METRICS.json` 과
`_docs/paper/generated/V2_DEV_RESULTS.md` 에서 온다.

## Gate

```text
   PASS  1  Night detection        R0 0.840 -> V2-D 0.960
   PASS  2  Day detection          R0 1.000 -> V2-D 0.986  (허용 -0.02)
   FAIL  3  Common NME < R0        Δframe median +0.00030  (음수여야 한다)
   PASS  4  Common NME < V2-A      V2-A 0.0219 -> V2-D 0.0212
   PASS  5  q>=0.75 축 순열 < V2-A  V2-A 0.157 -> V2-D 0.120
   FAIL  6  전체 축 순열 <= R0      R0 0.047 -> V2-D 0.053
   FAIL  7  detection 만 좋아짐     (gate 3 과 동일)
```

## 전체 표

```text
model           det ALL  det day  det night       NME  axis all  axis q>=.75  kp_conf p05
R0                0.975    1.000      0.840    0.0209     0.047        0.096        0.976
V2A_CONF25        0.994    0.986      1.000    0.0219     0.053        0.157        0.969
V2B_KP_MASK       0.984    0.971      0.960    0.0217     0.050        0.133        0.826
V2C_AMBIG         0.991    1.000      0.960    0.0216     0.044        0.120        0.754
V2D_FULL          0.987    0.986      0.960    0.0212     0.053        0.120        0.762
```

## 무엇이 성공했나

**1. detection 은 의도대로 고쳐졌다.**  야간 검출이 0.840 -> 0.96~1.00 으로 오르고
주간은 거의 유지된다.  V1 에서 "difficult viewpoint 의 keypoint 가 불안정하다는
이유로 detection adaptation 신호까지 버린다" 던 문제는 사라졌다 — frame 을 keypoint
때문에 버리지 않으므로 273 장 전부가 box supervision 에 기여한다.

**2. V2 구성요소는 V1 대비 실제로 작동한다.**  V2-A 를 기준으로 하면 per-keypoint
mask 와 ambiguity mask 가 localisation 손해를 줄이고(Δframe +0.00091 -> +0.00030)
모호 시점의 축 순열을 낮춘다(0.157 -> 0.120).  Q3 와 Q4 에 대해 **방향은 지지된다.**

```text
arm            frames   R0 NME   arm NME    Δframe                CI95
V2A_CONF25        309   0.0208   0.0219   +0.00091  [+0.00023, +0.00160]
V2B_KP_MASK       308   0.0207   0.0217   +0.00030  [-0.00017, +0.00089]
V2C_AMBIG         310   0.0208   0.0216   +0.00093  [+0.00052, +0.00148]
V2D_FULL          309   0.0209   0.0212   +0.00030  [-0.00024, +0.00090]
```

V2-B 와 V2-D 는 CI 가 0 을 포함한다 — **R0 보다 나쁘다고 말할 수 없다.**  V2-A 와
V2-C 는 CI 가 0 을 배제하고 악화 쪽이다.

## 왜 실패했나

**A. localisation 이 R0 를 넘지 못하고 같아졌을 뿐이다.**  gate 3 은 "V2-D < R0" 를
요구하는데 결과는 통계적으로 동률이다.  V1 의 명백한 악화는 사라졌지만 개선은 없다.

**B. 남은 악화는 전부 야간이다.**

```text
arm            domain      n    Δframe                CI95
V2D_FULL       daytime    69  -0.00092  [-0.00290, +0.00060]   개선 (CI 0 포함)
V2D_FULL       nighttime  42  +0.00297  [+0.00052, +0.00564]   악화 (CI 0 배제)
```

주간은 오히려 좋아졌다.  야간만 악화가 남고 CI 가 0 을 배제한다.  이는 V1 진단의
"nighttime C2 — 축과 무관한 열화" 와 같은 자리다
(`_docs/paper/CORNER_REGRESSION_CAUSES.md`).  **V2 는 축 문제를 건드렸지 야간 문제를
건드리지 않았다.**

**C. balanced replay 가 축을 전체적으로 개선하지 못했다.**  V2-C(0.044)는 R0(0.047)
보다 축 순열이 적은데, 여기에 replay 균형을 더한 V2-D 는 0.053 으로 되돌아간다.
모호 subset 에서는 둘 다 0.120 으로 같다.  즉 B2 를 720 슬롯까지 올린 것이 모호
시점을 더 고치지는 못하면서 다른 시점의 노출을 줄였다.

**D. 마스킹의 대가가 측정됐다.**  `KEYPOINT_MASK_CONTRACT.json` 이 예고한 대로
`kobj` 가 masked 코너에 "보이지 않음" 을 학습시킨다.

```text
model           kp_conf median   p05    kp_conf < 0.5 비율
R0                       0.999  0.976              0.001
V2A_CONF25               0.999  0.969              0.010
V2B_KP_MASK              0.998  0.826              0.022
V2C_AMBIG                0.998  0.754              0.028
V2D_FULL                 0.999  0.762              0.031
```

mask 를 켠 arm 에서 p05 가 0.97 -> 0.75~0.83 으로 떨어지고 `kp_conf < 0.5` 비율이
0.1% -> 2~3% 로 오른다.  배포가 `kp_conf >= 0.5` 를 쓰므로 이는 실질적인 비용이다.
중앙값은 그대로이므로 대다수 코너는 영향이 없고, **꼬리에서만 눌린다** — 즉 정확히
mask 대상이 된 어려운 코너들이다.

## 다음에 무엇을 보아야 하나 (실행하지 않았다)

이 절은 기록일 뿐이다.  §20 에 따라 이번 트랙에서는 아무것도 추가로 돌리지 않았다.

- 야간 localisation 은 pseudo-label 선택 문제가 아닐 수 있다.  V2 는 어떤 arm 도
  야간 Δ 를 0 아래로 내리지 못했고, V1 의 nighttime C2 도 미설명으로 남아 있다.
- `kobj` 를 통한 negative visibility supervision 을 피하려면 loss 계약을 바꿔야
  하는데, 그건 §11 이 금지한 범위다.  별도 사전등록이 필요하다.
- V2-C 가 축 순열에서 R0 를 이긴 유일한 arm 이라는 점(0.044 대 0.047)은 기록해 둔다.
  다만 gate 는 V2-D 를 Proposed 로 고정했고, 결과를 보고 Proposed 를 바꾸지 않는다.
