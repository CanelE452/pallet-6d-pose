# V2 development results

**개발 모집단은 PAPER_EVAL 319 다.**  V2 방법이 이 모집단의 진단을 보고
설계됐으므로 여기서 최종 성능을 주장하지 않는다
(`_docs/paper/SELFTRAIN_V2_PROTOCOL.md`).

## Coverage 와 Geometry — 두 축을 합치지 않는다

```text
model           det ALL  det day  det night       NME  axis all  axis q>=.75  kp_conf med
----------------------------------------------------------------------------------------
R0                0.975    1.000      0.840    0.0209     0.047        0.096        0.999
V2A_CONF25        0.994    0.986      1.000    0.0219     0.053        0.157        0.999
V2B_KP_MASK       0.984    0.971      0.960    0.0217     0.050        0.133        0.998
V2C_AMBIG         0.991    1.000      0.960    0.0216     0.044        0.120        0.998
V2D_FULL          0.987    0.986      0.960    0.0212     0.053        0.120        0.999
```

`NME` 는 R0 와 그 arm 이 둘 다 검출한 프레임의 같은 supervised keypoint 를
cuboid diagonal 로 나눈 값이다.  R0 열은 V2-D 와의 공통 프레임 기준이다.

`kp_conf med` 는 감시 지표다 — per-keypoint mask 가 keypoint objectness 를
통해 kp_conf 를 누르는지 본다 (배포는 kp_conf >= 0.5 를 쓴다).

## DEV gate

```text
PASS  1_night_detection: R0 0.840 -> V2-D 0.960
PASS  2_day_detection_no_collapse: R0 1.000 -> V2-D 0.986 (허용 -0.02)
FAIL  3_common_nme_below_r0: Δframe median +0.00030 (음수여야 한다)
PASS  4_common_nme_below_v2a: V2-A 0.0219 -> V2-D 0.0212
PASS  5_ambiguous_axis_below_v2a: q>=0.75  V2-A 0.157 -> V2-D 0.120
FAIL  6_axis_at_most_r0: R0 0.047 -> V2-D 0.053
FAIL  7_not_detection_only: detection 만 좋아지고 localisation 이 악화하면 FAIL — gate 3 과 같다
```

**FAIL** — `V2_METHOD_STATUS = FAILED`

DEV 는 방법 개발 모집단이다.  CI 가 0 을 포함해도 방향과 효과 크기로 final 을 진행할 수 있으나, 이것을 paper confirmation 이라 부르지 않는다.

