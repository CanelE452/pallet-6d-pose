# V2 development results

**개발 모집단은 PAPER_EVAL 319 다.**  V2 방법이 이 모집단의 진단을 보고
설계됐으므로 여기서 최종 성능을 주장하지 않는다
(`_docs/paper/SELFTRAIN_V2_PROTOCOL.md`).

## Coverage 와 Geometry — 두 축을 합치지 않는다

```text
model           det ALL  det day  det night       NME  axis all  axis q>=.75  kp_conf med
----------------------------------------------------------------------------------------
R0                0.975    1.000      0.840    0.0207     0.047        0.096        0.999
V3B_TRUE_IGNORE_AMBIG__FULL    0.991    1.000      0.960    0.0219     0.041        0.084        0.999
V5_RELIABILITY_WEIGHTED    0.984    0.986      0.960    0.0220     0.044        0.084        0.999
```

`NME` 는 R0 와 그 arm 이 둘 다 검출한 프레임의 같은 supervised keypoint 를
cuboid diagonal 로 나눈 값이다.  R0 열은 V2-D 와의 공통 프레임 기준이다.

`kp_conf med` 는 감시 지표다 — per-keypoint mask 가 keypoint objectness 를
통해 kp_conf 를 누르는지 본다 (배포는 kp_conf >= 0.5 를 쓴다).

## DEV gate

```text
PASS  G1_night_detection_holds: V3-B 0.960 -> V5 0.960
PASS  G2_day_detection_no_collapse: R0 1.000 -> V5 0.986 (허용 -0.02)
FAIL  G3_all_nme_below_r0: Δframe median +0.00086 (음수여야 한다)
FAIL  G4_night_nme_not_worse_than_r0: Night Δframe +0.00174 (0 이하)
FAIL  G5_better_than_v3b: V3-B 0.0219 -> V5 0.0220
PASS  G6_night_better_than_v3b: Night V3-B 0.0283 -> V5 0.0280
PASS  G7_axis_at_most_r0: R0 0.047 -> V5 0.044
PASS  G8_ambiguous_axis_at_most_r0: q>=0.75  R0 0.096 -> V5 0.084
FAIL  G9_not_detection_only: G1/G2 만 좋아지고 G3/G4 가 실패하면 overall FAIL
```

**FAIL** — `V5_METHOD_STATUS = FAILED`

PAPER_EVAL 은 V1~V4 와 FILTER_SEPARABILITY 에 이미 쓰였다.  PASS 여도 DEV_PASS_REQUIRES_UNTOUCHED_CONFIRMATION 까지만 판정한다.

