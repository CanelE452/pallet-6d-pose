# PURPOSE — paper_strong_teacher_v1

[소비처] 논문 §Method(stronger source-only teacher + cross-teacher consensus) 와
`_docs/paper/STRONG_TEACHER_V1_PROTOCOL.md` 의 teacher gate · student gate.
V1~V5 가 전부 **selection** 을 건드려 실패한 자리에서, 이번엔 **teacher** 를 바꾼다.

[문장] 더 정확한 pseudo-label 을 만드는 source-only teacher(용량 증가 + 교차 합의)를
쓰면, 같은 compact YOLO26n student 가 R0 의 keypoint localisation 을 넘어선다.

## 판단 지표 (결과 보기 전 고정)

teacher gate 가 먼저다 — **teacher 가 R0 를 못 이기면 student 를 학습하지 않는다**.
- T1-G1~G5  medium source teacher 가 R0 대비 NME·야간·gross·p90·검출
- C1~C6     consensus teacher 가 R0 대비 NME·야간·gross·p90·catastrophic·coverage
- S1-G1~G9  student gate (teacher 통과 시에만)

## 직접성

student architecture·init·optimizer·pseudo frame membership·box label 은 V3-B 와
**동일**하게 고정한다.  바뀌는 것은 **keypoint pseudo target 의 출처** 하나다.
V5 reliability weighting 은 끈다 — 이번 질문은 "더 정확한 좌표가 student 를
움직이는가" 뿐이다.

## 금지

real keypoint / real pose supervision 이 들어간 모델을 teacher 로 쓰지 않는다
(`release/*-ft`, `*-livegt`, `runs_ft`, `runs_live_gt`).  seed 만 바꾼 모델을 독립
teacher 라 부르지 않는다.  TTA 는 original + horizontal flip 두 view 뿐이다.

## 보존

V1~V5 는 수정하지 않는다.  `PRIOR_TRACK_IMMUTABILITY_LOCK.json` 이 종료 시 해시로
강제한다.

## 연구 상태

PAPER_EVAL 319 는 V1~V5 에 이미 쓰였다.  여기서는 `TEACHER_DEV_ONLY` 이며,
PASS 여도 final claim 을 쓰지 않는다.
