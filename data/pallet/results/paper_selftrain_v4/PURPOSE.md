# PURPOSE — paper_selftrain_v4

[소비처] 논문 §Method(Geometry-refined selective self-training) + Main Table 1·2,
그리고 `_docs/paper/SELFTRAIN_V4_PROTOCOL.md` 의 proxy gate 와 DEV gate.
V1(frame-level) · V2(per-keypoint mask) · V3(true-ignore)가 모두 localisation 을
개선하지 못한 자리를 대체한다.

[문장] 중간 신뢰도 pseudo-keypoint 를 버리는 대신, 신뢰할 수 있는 고신뢰 코너와
등록된 물체 기하로부터 그 2D 위치를 복원해 학습시키면, teacher 원본 좌표보다 정확한
supervision 이 되어 R0 의 keypoint localisation 을 넘어선다.

## 판단 지표 (결과 보기 전 고정)

먼저 proxy gate — 학습 전에 복원 자체를 GT 로 채점한다.
- P1 Night repaired median NME < raw teacher median NME
- P2 Night gross rate  repaired <= raw
- P3 표본이 비교 불가능할 만큼 적지 않을 것 (최소값을 새로 정하지 않고 실제 N 보고,
  적으면 LOW_POWER 표시)

그다음 DEV gate G1~G9 (`SELFTRAIN_V4_METHOD_LOCK.json`).
Proposed = V4-C, 결과 전에 고정.

## 직접성

복원은 2D pseudo target 을 고치기 위한 것이다.  pseudo 6D pose·승리 hypothesis·
GT 축을 **저장하지 않는다**.  PnP 는 latent operator 일 뿐이다.

## 보존

V1·V2·V3 는 수정하지 않는다.  `V1_V2_V3_IMMUTABILITY_LOCK.json` 이 종료 시 해시로
강제한다.

## 마지막 재설계

PAPER_EVAL 319 를 쓰는 **마지막 single-frame self-training 재설계**다.  V4 가
실패하면 같은 319 에서 V5 를 설계하지 않는다.
