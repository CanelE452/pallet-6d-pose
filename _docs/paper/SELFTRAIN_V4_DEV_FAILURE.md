# V4 geometry repair — 메커니즘 단계에서 실패.  학습하지 않았다

`V4_METHOD_STATUS = FAILED`
`GEOMETRY_REPAIR_MECHANISM_FAIL`
`SINGLE_FRAME_SELFTRAINING_DEV_BUDGET_EXHAUSTED = true`

§16 에 따라 **학습을 시작하지 않고** 종료했다.  threshold 를 고쳐 gate 를 다시 맞추지
않았다.  untouched confirmation 은 열지 않았다.

V1·V2·V3 무결성: 봉인한 104 파일 전부 해시 일치.

## 왜 멈췄나 — 복원할 대상이 거의 없다

V4 의 전제는 "teacher 가 중간 정도 확신하는(0.5 ≤ c < 0.95) 코너가 문제이고, 그것을
신뢰 코너 + 기하로 복원하면 원본보다 정확해진다" 였다.

첫 번째 전제부터 성립하지 않는다.

### adaptation pool (GT 없이, §13)

```text
condition  frames  HIGH  CAND  LOW  repaired  no_anchor  hyp_disagree  ambiguous
daytime       123   956    20    8         0         14             6          0
nighttime     150  1158    28   14         1          5            15          7
```

2,184 코너 중 **HIGH-CONFIDENCE 가 2,114 개(96.8%)** 이고 repair candidate 는
48 개(2.2%)뿐이다.  실제로 복원에 성공한 것은 **1 개**다.

즉 V4-C 와 V4-B 는 학습셋에서 **2,184 개 중 단 1 개**의 keypoint 만 다르다.
gate G5·G6(V4-C < V4-B)은 복원이 아무리 잘 작동해도 검정력이 없다.  이 사실은
결과를 보기 전에 method lock 에 기록해 두었다.

### PAPER_EVAL (proxy, §15)

```text
domain      HIGH  CAND  LOW    합   CAND 비율
daytime      403    12    1   416       2.9%
nighttime    212     4    0   216       1.9%
none        1375     9    0  1384       0.7%
전체        1990    25    1  2016       1.2%
```

평가셋에서도 candidate 는 25 개(1.2%)다.

## 복원이 실제로 어떻게 되나

```text
복원 상태 (candidate 25 개)
AMBIGUOUS_VIEW           15    q >= 0.75 라 복원 금지
HYPOTHESIS_DISAGREE       5    W/D hypothesis 들이 2D 위치에 동의하지 않음
OUT_OF_IMAGE              2    복원 좌표가 화면 밖 (clipping 하지 않고 버림)
NO_VALID_HYPOTHESIS       1    anchor reprojection 이 tau 를 넘음
REPAIRED                  2    성공 (그중 supervised keypoint 는 1 개)
```

**복원 실패의 대부분이 AMBIGUOUS_VIEW 와 HYPOTHESIS_DISAGREE 다.**  이건 우연이 아니라
구조다 — 불확실한 코너는 바로 **물체 축 hypothesis 들이 서로 다른 답을 내는 곳**이다.
"신뢰 코너 + 알려진 기하가 불확실한 코너를 결정한다" 는 전제가 이 물체·이 시점에서는
성립하지 않는다.

hypothesis 를 GT 로 고르면 복원할 수 있지만, §9·§11 이 그것을 금지한다.  그 금지는
옳다 — GT 로 축을 고르면 배포 불가능한 방법이 된다.

## 유일하게 측정된 복원은 나빠졌다

```text
group                      n   raw NME   rep NME          Δ  raw gross  rep gross
ALL                        1   0.02069   0.02599   +0.00530      0.000      1.000
teacher_conf_0.80_0.95     1   0.02069   0.02599   +0.00530      0.000      1.000
nighttime                  0   (표본 없음)
```

n = 1 이므로 이것으로 "복원이 해롭다" 고 주장하지 않는다.  방향만 기록한다.

## Proxy gate

```text
FAIL  P1  Night repaired median < raw     야간 복원 표본 0 개 — 비교 불가
FAIL  P2  Night gross 악화 없음            야간 복원 표본 0 개 — 비교 불가
FAIL  P3  검정력                           ALL n=1  Night n=0   LOW_POWER
```

## 진단이 가리키던 것과의 관계

`V3_NIGHT_RESIDUAL_DIAGNOSIS.md` 는 야간 잔차가 teacher confidence 0.80~0.95 구간에
몰린다고 했다 (Δ NME +0.02170, n=23, conf ≥0.95 의 10 배).

그 신호는 **여전히 유효하다**.  다만 이번 감사가 덧붙이는 것은 규모다 — 그 구간은
야간 supervised keypoint 216 개 중 4 개(1.9%)뿐이다.  **item 당 손해는 크지만 집단이
작아서, 그것만으로 야간 잔차 전체를 설명하지 못한다.**  그리고 그 소수마저 기하로
복원할 수 없다.

## 남은 선택지 (§26)

같은 PAPER_EVAL 319 를 보고 V5 를 설계하지 않는다.  V1~V4 가 모두 이 모집단의
진단을 소비했다.

```text
NEW_DEV_DATA_REQUIRED       새 development set 을 만든 뒤 새 방법을 개발
PAPER_CLAIM_PIVOT_REQUIRED  self-training localisation claim 을 포기하고
                            detection/ranking 이득 + 실패 분석으로 서술
TEMPORAL_MULTIFRAME_TRACK   단일 프레임이 아닌 시계열/다중 프레임 방법으로 전환
```

세 트랙 모두에서 **V1~V4 의 음성 결과는 그대로 논문에 남긴다** — frame-level
필터링의 부족, `visibility=0` 이 true ignore 가 아니라는 발견, true-ignore 가
kp_conf 를 회복시키되 localisation 은 못 고친다는 것, 그리고 이번의 "불확실한 코너는
기하로 복원되지 않는다".

## 그럼에도 남는 확정된 이득

이번 트랙에서 새로 학습한 모델은 없지만, 앞선 트랙에서 확정된 것은 유지된다.

```text
detection   야간 0.840 -> 0.960, 주간 1.000 유지          (V3-B)
축 순열     전체 0.047 -> 0.041, q>=0.75 0.096 -> 0.084   (V3-B, R0 를 이긴 유일 arm)
kp_conf     true-ignore 가 V2 의 억압을 완전히 회복        (V3)
```

localisation 만 R0 를 넘지 못했다.
