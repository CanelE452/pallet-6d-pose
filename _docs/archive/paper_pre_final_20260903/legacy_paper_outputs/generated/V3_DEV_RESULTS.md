# V2 development results

**개발 모집단은 PAPER_EVAL 319 다.**  V2 방법이 이 모집단의 진단을 보고
설계됐으므로 여기서 최종 성능을 주장하지 않는다
(`_docs/paper/SELFTRAIN_V2_PROTOCOL.md`).

## Coverage 와 Geometry — 두 축을 합치지 않는다

```text
model           det ALL  det day  det night       NME  axis all  axis q>=.75  kp_conf med
----------------------------------------------------------------------------------------
R0                0.975    1.000      0.840    0.0208     0.047        0.096        0.999
V2B_KP_MASK__FULL    0.984    0.971      0.960    0.0217     0.050        0.133        0.998
V3A_TRUE_IGNORE    0.984    0.986      0.980    0.0218     0.053        0.120        0.999
V3B_TRUE_IGNORE_AMBIG    0.991    1.000      0.960    0.0219     0.041        0.084        0.999
```

`NME` 는 R0 와 그 arm 이 둘 다 검출한 프레임의 같은 supervised keypoint 를
cuboid diagonal 로 나눈 값이다.  R0 열은 V2-D 와의 공통 프레임 기준이다.

`kp_conf med` 는 감시 지표다 — per-keypoint mask 가 keypoint objectness 를
통해 kp_conf 를 누르는지 본다 (배포는 kp_conf >= 0.5 를 쓴다).

## DEV gate

```text
PASS  G1_night_detection_above_r0: R0 0.840 -> V3-B 0.960
PASS  G2_day_detection_no_collapse: R0 1.000 -> V3-B 1.000 (허용 -0.02)
FAIL  G3_all_nme_below_r0: Δframe median +0.00067 (음수여야 한다)
FAIL  G4_night_nme_not_worse_than_r0: Night Δframe median +0.00168 (0 이하여야 한다)
FAIL  G5_true_ignore_beats_v2_masking: V2B 0.0217 -> V3-A 0.0218
PASS  G6_ambiguity_helps: q>=0.75  V3-A 0.120 -> V3-B 0.084
PASS  G7_axis_at_most_r0: R0 0.047 -> V3-B 0.041
FAIL  G8_not_detection_only: detection 이 좋아져도 G3/G4 가 실패하면 overall FAIL
```

**FAIL** — `V3_METHOD_STATUS = FAILED`

DEV 는 방법 개발 모집단이다.  CI 는 따로 보고하며 점추정 gate 를 결과를 보고 바꾸지 않는다.  이것을 paper confirmation 이라 부르지 않는다.

