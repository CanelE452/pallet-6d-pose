# PURPOSE — paper_selftrain_v2

[소비처] 논문 §Method(Decoupled pseudo-supervision) + Main Table 1·2 와
`_docs/paper/SELFTRAIN_V2_PROTOCOL.md` 의 DEV gate.  V1 이 실패로 남은 자리를
대체하는 제안 방법의 근거가 된다.

[문장] detection 신뢰도와 keypoint 신뢰도를 분리하고, 기하 일관성을 frame 이 아닌
keypoint 단위로 적용하며, 축이 모호한 시점의 real corner 라벨만 억제하고 정확한
synthetic supervision 을 시점 균형으로 재생하면, unlabeled target adaptation 이
detection 이득을 유지하면서 keypoint localisation 열화를 막는다.

## 판단 지표 (결과 보기 전 고정)

- Detection rate (Night >= R0, Day 파국적 열화 없음)
- Common-detected paired NME (V2-D < R0, V2-D < V2-A)
- Axis permutation rate (V2-D <= R0, q>=0.75 subset 에서 V2-D < V2-A)
- 감시 지표: kp_conf 분포 — mask 가 kobj 를 통해 kp_conf 를 누르는지
  (KEYPOINT_MASK_CONTRACT.json 참조)

## 직접성

부차적 매력(새 head·새 backbone·새 loss) 금지.  변경은 **training data /
supervision contract** 뿐이다.

## 보존

V1 은 실패/진단 baseline 으로 그대로 둔다.  수정·삭제 금지.
