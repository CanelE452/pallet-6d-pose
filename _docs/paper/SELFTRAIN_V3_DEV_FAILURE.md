# V3 development — FAILED, 그리고 가설 하나가 정확히 갈렸다

`V3_METHOD_STATUS = FAILED`.  §25 에 따라 threshold·q·LR·epoch·pseudo fraction 을
sweep 하지 않았고, Proposed 를 V3-A 로 바꾸지 않았으며, untouched confirmation 을
열지 않았다.

V1·V2 무결성: 봉인한 82 파일 전부 해시 일치 (`V1_V2_IMMUTABILITY_LOCK.json`).

## Gate

```text
   PASS  G1  Night detection > R0        0.840 -> 0.960
   PASS  G2  Day detection 유지          1.000 -> 1.000
   FAIL  G3  ALL NME < R0                Δframe +0.00067
   FAIL  G4  Night NME <= R0             Δframe +0.00168
   FAIL  G5  V3-A NME < V2B NME          0.0217 -> 0.0218
   PASS  G6  ambiguity 가 축을 돕는다     q>=0.75  0.120 -> 0.084
   PASS  G7  축 순열 <= R0                0.047 -> 0.041
   FAIL  G8  detection-only 아님          (G3/G4 실패)
```

## 전체 표

```text
model                  det ALL  det day  det night      NME  axis all  axis q>=.75
R0                       0.975    1.000      0.840   0.0208     0.047        0.096
V2B_KP_MASK              0.984    0.971      0.960   0.0217     0.050        0.133
V3A_TRUE_IGNORE          0.984    0.986      0.980   0.0218     0.053        0.120
V3B_TRUE_IGNORE_AMBIG    0.991    1.000      0.960   0.0219     0.041        0.084
```

## 가설이 갈렸다 — kp_conf 는 완전히 회복됐고, NME 는 그대로다

§22 가 미리 적어 둔 두 갈래 중 첫 번째가 나왔다.

```text
model                  kp_conf median   p05   kp_conf < 0.5
R0                              0.999  0.976          0.001
V2B_KP_MASK                     0.998  0.826          0.022
V2C_AMBIG                       0.998  0.754          0.028
V2D_FULL                        0.999  0.762          0.031
V3A_TRUE_IGNORE                 0.999  0.981          0.006
V3B_TRUE_IGNORE_AMBIG           0.999  0.982          0.005
```

**true-ignore 는 의도한 것을 정확히 했다.**  V2 에서 0.826~0.754 로 눌렸던 kp_conf
꼬리가 0.981~0.982 로, `< 0.5` 비율이 2.2~3.1% 에서 0.5~0.6% 로 돌아왔다 — R0 수준
이상이다.  gradient 계약이 예고한 그대로다.

그런데 localisation 은 따라오지 않았다.

```text
model                       frames  arm NME    Δframe                CI95
V2B_KP_MASK / ALL              308   0.0217  +0.00030  [-0.00017, +0.00089]
V3A_TRUE_IGNORE / ALL          307   0.0218  +0.00072  [+0.00010, +0.00147]
V3B_TRUE_IGNORE_AMBIG / ALL    310   0.0219  +0.00067  [-0.00008, +0.00146]
```

V3-A 는 V2-B 와 **같은 pseudo 프레임·같은 좌표·같은 신뢰도 선택**을 쓰고 mask
semantics 만 다른데, NME 가 0.0217 -> 0.0218 로 사실상 동일하다.

**따라서 §22 의 결론이 적용된다:**

> V2 masking 의 confidence suppression 문제는 실재했고 V3 가 고쳤다.
> 그러나 야간 localisation 의 원인은 거기가 아니다.

이것은 실패지만 **정보가 있는 실패**다.  keypoint objectness 의 negative supervision
은 kp_conf 를 눌렀을 뿐, NME 를 움직이는 원인이 아니었다.

## 야간은 여전히 남는다

```text
model                       domain      Δframe                CI95
V3B_TRUE_IGNORE_AMBIG       daytime   -0.00081  [-0.00248, +0.00154]   개선
V3B_TRUE_IGNORE_AMBIG       nighttime +0.00168  [+0.00003, +0.00462]   악화
```

주간은 개선(CI 가 0 포함), 야간만 악화가 남고 CI 가 간신히 0 을 배제한다.
`V3_NIGHT_RESIDUAL_DIAGNOSIS.md` 가 짚은 자리 — 가림·저앙각·teacher confidence
0.80~0.95 — 와 일치한다.  V3 는 그 자리를 건드리는 방법이 아니었다.

## 성공한 것 — 축 순열에서 처음으로 R0 를 이겼다

```text
model                   axis all   q>=0.75
R0                         0.047     0.096
V2B_KP_MASK                0.050     0.133
V2C_AMBIG                  0.044     0.120
V2D_FULL                   0.053     0.120
V3A_TRUE_IGNORE            0.053     0.120
V3B_TRUE_IGNORE_AMBIG      0.041     0.084
```

**V3-B 는 전체(0.041)와 모호 subset(0.084) 둘 다에서 R0 를 이긴 유일한 arm 이다.**
V3-A(0.053 / 0.120) 대비 차이는 ambiguity 처리 하나이므로, 그 효과는 이 데이터에서
분리되어 보인다 (G6 PASS).

detection 도 최고다 — 주간 1.000 을 유지하면서 야간 0.840 -> 0.960.

즉 V3-B 는 **detection 과 축 배정 두 축에서 R0 를 이기고 NME 에서만 동률에 머문다.**
gate 는 NME 개선을 요구했고 그것이 없으므로 FAIL 이다.

## 남은 것

- 야간 localisation 잔차의 원인.  V1 의 axis 설명도, V2 의 frame-level 설명도,
  V3 의 kobj 설명도 아니었다.  진단이 가리키는 곳은 **가림 프레임에서 teacher 가
  중간 신뢰도로 낸 코너**다 (Δ +0.02170, n=23 — 나머지의 10 배).
- 그 코너들은 `kp_conf >= 0.5` 를 통과하므로 현재 신뢰도 계약이 걸러내지 못한다.
  이를 고치려면 임계를 움직여야 하는데 §10·§25 가 금지한다.  다음 트랙에서
  **사전등록된 별도 가설**로 다뤄야 한다.
- V3-B 의 축 이득(0.047 -> 0.041, 0.096 -> 0.084)은 confirmation 없이 주장하지
  않는다.  DEV 는 방법 개발 모집단이다.
