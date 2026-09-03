# PURPOSE — paper_fast6d_screen_v1b

[소비처]
FAST_6D_SCREEN_V1 의 사후 감사에서 드러난 **두 가지 사실오류**를 교정한 결과.
사용자 판단으로 소비된다.  positive 여도 METHOD_CANDIDATE 까지이며 논문 확증
결과가 아니다.

[문장]
"V1 의 S2 는 YOLO 가 학습한 bbox semantics 와 다른 상자를 맞췄고, S5 는 실제로
존재하는 real line inference path 를 못 찾아 BLOCKED 로 닫혔다.  두 arm 을
사실대로 완성하면 결론이 바뀌는가" — 또는 그 반증.

## 이 작업의 지위

POST_STOP_EXPLORATORY_CORRECTION.  `_docs/paper/final/EXPERIMENT_STOP_LOCK.json`
은 수정하지 않는다.  새 confirmatory study 가 아니라 **잘못 구현·잘못 차단된 arm 의
사실적 완성**이다.

## 판단 지표 (결과 보기 전 고정 — FAST_6D_SCREEN_V1B_LOCK.json)

```
primary    IoU3D median ↑ · ADDsym AUC ↑
C1 gate    ΔIoU3D >= +0.020 OR ΔADDsym >= +0.020, 다른 지표 악화 없음
L3 gate    두 seed 모두 Δ >= 0, median(Δ) >= +0.020 (둘 중 하나), coverage drop <= 0.01,
           R/t median 5% 초과 악화 없음
```

★ 목적함수(predicted reprojection)가 줄었다는 이유만으로 PASS 하지 않는다.

## 적용범위 한계 (측정 전 선언)

population 은 PAPER_EVAL positive 319 로 **이미 개발에 반복 사용된 셋**이다.
held-out · independent · final · confirmed 라고 부르지 않는다.

## 실패 시

C1 STOP AND L3 STOP 이면 새 후보를 만들지 않고 PAPER_FRAMING_CLOSURE_V1 로 넘어간다.

NEXT_ACTION = USER_REVIEW_OVERNIGHT_RESULT
