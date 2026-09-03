# PURPOSE — paper_fast_teacher_v1

[소비처] `_docs/paper/FAST_TEACHER_V1.md` 의 teacher gate.  YOLO26m 60ep(~21시간)를
돌릴 가치가 있는지 판단하는 근거가 된다.

[문장] 새 모델을 학습하지 않고 R0 를 여러 inference view 로 돌려 합치면, R0 단독보다
정확한 2D keypoint teacher 를 만들 수 있다.

## 판단 지표 (결과 보기 전 고정)

candidate 가 값을 낸 keypoint 에서 R0 도 **같은 집합으로** 재는 paired 비교만 쓴다.
- A1/B1  ALL NME < R0
- A2/B2  Night NME <= R0
- A3/B3  p90 < R0
- A4/B4  gross20 < R0
- B5     catastrophic40 <= R0  (FAST-B 만)

## 직접성

새 학습 0 회.  새 학습 가중치 0 개.  confidence 로 좌표를 가중하지 않는다.
teacher 가 R0 를 못 이기면 student 를 시작하지 않는다.

## 함정 방지

- flip cache 는 이미 unflip + index 복원돼 있다.  재mirror 금지(과거 1.9px -> 127px).
- coverage selection 으로 지표가 좋아 보이는 것을 막기 위해 paired 비교만 gate 에 쓴다.
  V1~V5 를 다섯 번 속인 것이 그 효과다.
- imgsz 는 960 한 값만 본다.  OOM 이면 FAST-B = UNAVAILABLE, sweep 금지.

## 보존

V1~V5 와 R0 는 수정하지 않는다.
