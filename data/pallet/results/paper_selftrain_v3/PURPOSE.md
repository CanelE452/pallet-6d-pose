# PURPOSE — paper_selftrain_v3

[소비처] 논문 §Method(True-ignore geometry-aware pseudo-supervision) + Main Table 1·2,
그리고 `_docs/paper/SELFTRAIN_V3_PROTOCOL.md` 의 DEV gate.  V2 가 "ignore" 로 의도한
것이 실제로는 negative visibility supervision 이었음을 loss level 에서 고친다.

[문장] 신뢰할 수 없는 pseudo keypoint 를 location·RLE·keypoint-objectness 어느
항에도 기여시키지 않으면(진짜 no-gradient), box/class adaptation 이득을 유지하면서
V2 에 남은 keypoint localisation 열화와 kp_conf 억압이 사라진다.

## 판단 지표 (결과 보기 전 고정 — G1~G8)

- G1 Night detection > R0
- G2 Day detection >= R0 - 0.02
- G3 ALL paired common-frame NME delta vs R0 < 0
- G4 Night paired NME delta vs R0 <= 0
- G5 V3-A paired NME < V2B paired NME  (true-ignore 자체의 효과)
- G6 V3-B q>=0.75 axis permutation < V3-A
- G7 V3-B axis permutation all <= R0
- G8 G1/G2 가 통과해도 G3/G4 가 실패하면 overall FAIL

## 직접성

architecture·forward 는 그대로.  변경은 **training loss supervision mask** 뿐이다.
balanced synthetic replay 는 V2 에서 실패했으므로 쓰지 않는다.

## 보존

V1·V2 는 수정하지 않는다.  `V1_V2_IMMUTABILITY_LOCK.json` 이 종료 시 해시로 강제.
