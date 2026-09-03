# V5 — 라벨 품질은 올랐는데 학생은 안 나아졌다 (CASE B)

`V5_METHOD_STATUS = FAILED`

§26 에 따라 같은 PAPER_EVAL 을 보고 V6 를 설계하지 않는다.  threshold·weight sweep 을
하지 않았고, untouched confirmation 을 열지 않았다.

V1~V4 및 FILTER_SEPARABILITY 무결성: 봉인한 105 항목 전부 해시 일치.

## 이번에 검증한 것

"kp confidence 를 더 세게 자르기" 가 아니라 —

> frame context + keypoint confidence + geometry consistency 를 GT 없는 하나의
> reliability score 로 묶고, 점수가 높은 pseudo frame 을 **더 자주 노출한다**

를 딱 한 번.  자르지 않고 **반복 횟수만** 바꿨다.

V3-B 와의 차이는 그것 하나다 — 라벨 집합 해시가 동일하고(273 개, `c9f303e5b0903b4f`),
슬롯 수도 동일하다(Day 327 / Night 393).  노출 분포만 `{2:99, 3:174}` 에서
`{1:14, 2:96, 3:138, 4:25}` 로 바뀌었다.

## 메커니즘은 통과했다 — 점수는 옳은 방향으로 가중한다

학습 **전에** 잰 것 (`V5_MECHANISM_CHECK.md`):

```text
signal          AUC        n=79 프레임, frame gross 51.9%
R_total       0.763        <- 어떤 단일 신호보다 높다
s_reproj      0.745
s_remove      0.694
box_conf      0.653
s_flip        0.583
```

```text
metric              uniform V3-B   weighted V5     변화
frame_gross               0.5190        0.4615   -0.0575
corner_gross              0.2078        0.1823   -0.0255
median_error_px          20.0479       18.2491   -1.7988
p90_error_px             31.4206       28.7036   -2.7170
```

**M1·M2·M3 전부 PASS.**  학생이 기대상 보게 되는 라벨이 실제로 깨끗해진다.

점수가 신호 하나의 복사본도 아니다 — 성분과의 최대 Spearman 이 0.787 이고, 값은
0.03~0.77 에 퍼져 있다.

## 그런데 학생은 나아지지 않았다

```text
metric                    R0      V3-B        V5     V5-R0    V5-V3B
ALL detection          0.975     0.991     0.984    +0.009    -0.006
Day detection          1.000     1.000     0.986    -0.014    -0.014
Night detection        0.840     0.960     0.960    +0.120    +0.000

ALL paired NME        0.0207    0.0219    0.0220   +0.0013   +0.0001
Day paired NME        0.0260    0.0284    0.0289   +0.0029   +0.0005
Night paired NME      0.0262    0.0283    0.0280   +0.0018   -0.0004

Axis all               0.047     0.041     0.044    -0.003    +0.003
Axis q>=.75            0.096     0.084     0.084    -0.012    +0.000
kp_conf p05            0.976     0.982     0.985
```

```text
paired NME delta (V5 - R0)
ALL        n=308  Δ +0.00086  [+0.00018, +0.00140]   resolved
daytime    n= 69  Δ +0.00011  [-0.00155, +0.00228]   NOT_STATISTICALLY_RESOLVED
nighttime  n= 42  Δ +0.00174  [+0.00006, +0.00356]   resolved
```

**ALL NME 가 V3-B 0.0219 에서 V5 0.0220 으로, 사실상 동일하다.**  기대 corner gross 를
0.208 -> 0.182 로 12% 낮췄는데 학생의 localisation 은 0.5% 도 안 움직였다.

## Gate

```text
   PASS  G1  Night detection 유지        V3-B 0.960 -> V5 0.960
   PASS  G2  Day detection 파국 없음      R0 1.000 -> V5 0.986
   FAIL  G3  ALL NME < R0                 Δ +0.00086
   FAIL  G4  Night NME <= R0              Δ +0.00174
   FAIL  G5  V3-B 보다 나음               0.0219 -> 0.0220
   PASS  G6  Night 이 V3-B 보다 나음       0.0283 -> 0.0280
   PASS  G7  축 <= R0                     0.047 -> 0.044
   PASS  G8  q>=0.75 축 <= R0             0.096 -> 0.084
   FAIL  G9  detection-only 아님          (G3/G4 실패)
```

## 판정 — §27 의 CASE B

```text
proxy quality 개선   O   (corner gross -12%, frame gross -11%)
V5 NME 개선          X   (+0.0001 대 V3-B, +0.0013 대 R0)
```

> **pseudo-label purity 를 높여도 student localisation 이 개선되지 않는다.**
> 이 경우 single-frame pseudo-label selection 자체를 더 이상 붙잡지 않는다.

이건 V5 만의 결론이 아니라 **V1~V5 다섯 트랙의 수렴점**이다.

```text
V1  frame-level 필터            localisation 악화
V2  per-keypoint mask           악화는 줄었으나 R0 미달, kp_conf 억압
V3  true-ignore                 kp_conf 완전 회복, localisation 그대로
V4  geometry repair             복원 대상이 1.2% 뿐, 메커니즘 단계에서 종료
V5  reliability weighting       라벨 품질 개선 확인, localisation 그대로
```

**선택 방식을 다섯 번 바꿨고, 그중 둘은 대리 지표에서 실제로 개선을 만들었다.
그런데 학생의 keypoint 위치는 한 번도 R0 를 넘지 못했다.**

## 왜 그럴 수밖에 없었는지 — 이미 측정된 것

R0 자체가 supervised keypoint 의 **17.2% 를 20 px 이상 틀린다** (주간 30.2%).
teacher 가 그 수준인데 teacher 의 예측을 골라 먹여서 학생이 teacher 를 넘기를
기대하는 구조다.  선택을 아무리 정교하게 해도 **teacher 가 모르는 것을 만들어내지는
못한다.**

V4 가 그걸 만들어 보려다(기하 복원) 대상이 1.2% 뿐이라 멈췄고, V5 가 남은 것을
정교하게 고르려다 품질은 올렸지만 학생은 그대로였다.

## 남은 선택지

```text
NEW_DEV_DATA_REQUIRED       새 development set 을 만든 뒤 새 방법을 개발
PAPER_CLAIM_PIVOT_REQUIRED  self-training localisation claim 을 포기하고
                            detection/ranking 이득 + 다섯 트랙의 음성 결과로 서술
TEMPORAL_MULTIFRAME_TRACK   단일 프레임이 아닌 시계열/다중 프레임으로 전환
BETTER_TEACHER              teacher 자체를 바꾼다 (selection 이 아니라 source)
```

네 번째가 이번 결과로 새로 뚜렷해졌다 — 다섯 트랙이 전부 **selection** 을 건드렸고
전부 실패했으므로, 남은 레버는 **teacher** 다.

## 확정된 이득은 유지된다

```text
detection   야간 0.840 -> 0.960, 주간 1.000 유지 (V3-B) / 0.986 (V5)
축 순열     전체 0.047 -> 0.041 (V3-B) / 0.044 (V5)
            q>=0.75  0.096 -> 0.084 (V3-B, V5 동일)
kp_conf     true-ignore 가 V2 의 억압을 회복 (p05 0.976 -> 0.982~0.985)
```

localisation 만 다섯 번 모두 R0 를 넘지 못했다.
