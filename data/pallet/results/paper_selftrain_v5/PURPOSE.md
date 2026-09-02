# PURPOSE — paper_selftrain_v5 (V5_EXPLORATORY_RELIABILITY)

[소비처] 논문 §Method(GT-free pseudo-label reliability weighting) + Main Table,
그리고 `_docs/paper/SELFTRAIN_V5_PROTOCOL.md` 의 mechanism gate 와 DEV gate.
V1~V4 가 전부 실패한 자리에서 마지막으로 **노출 분포**를 건드려 본다.

[문장] 나쁜 pseudo frame 을 버리는 대신 detection·기하·keypoint 신호를 GT 없이
하나의 reliability score 로 묶어 **덜 반복 노출**하면, V3-B 의 detection·축 이득을
잃지 않으면서 keypoint localisation 이 R0 아래로 내려간다.

## 판단 지표 (결과 보기 전 고정)

mechanism gate — 학습 전에 확인한다
- M1  R_total 의 frame gross AUC > 0.5
- M2  reliability-weighted expected corner gross < uniform (V3-B)
- M3  reliability-weighted expected frame gross  < uniform (V3-B)

DEV gate G1~G9 는 `RELIABILITY_SCORE_LOCK.json` 에 있다.

## 직접성

새 좌표를 만들지 않는다.  loss·architecture·inference 를 바꾸지 않는다.
V3-B 와의 유일한 차이는 **pseudo frame 반복 횟수** 하나다.

## 금지

kp_conf / cutoff / retention sweep, PAPER_EVAL GT 로 weight 학습·calibration,
logistic regression·random forest·MLP.  score 는 FIXED · UNSUPERVISED ·
MONOTONIC · RANK-FUSION 이다.

## 연구 상태

PAPER_EVAL 319 는 V1~V4 와 FILTER_SEPARABILITY 에 이미 쓰였다.  이번 결과가 좋아도
`DEV_PASS_REQUIRES_UNTOUCHED_CONFIRMATION` 까지만 판정한다.  실패하면 같은
PAPER_EVAL 을 보고 V6 를 설계하지 않는다.

## 보존

V1~V4 는 수정하지 않는다.  `V1_V2_V3_V4_IMMUTABILITY_LOCK.json` 이 종료 시 해시로
강제한다.
