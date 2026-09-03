# PURPOSE — temporal evaluation closure v1

[소비처]
temporal pilot 을 논문에서 어떻게 부를지 정하는 근거.  기존 pilot 결과가
formal result 로 쓸 수 없는 상태이므로, 좌표는 그대로 두고 **채점만** 바로잡아
공식 판정을 확정한다.  다음 단계는 paper framing decision 뿐이다.

[문장]
"기존 refinement 좌표를 그대로 두고 lock 이 실제로 요구한 population,
pooled corner 2D metric, frozen prediction-only pose evaluator 로 다시 채점했을 때
temporal method 의 formal verdict 는 X 다" — X ∈ {FAILED_TO_IMPROVE,
POPULATION_LIMITED, OBSERVED_IMPROVEMENT}.

## 왜 다시 채점하는가 — 내 채점의 결함 4개

```
1 population  lock 은 evaluation-ineligible 을 배제하라 했는데 FT_OVERLAP 만 걸렀다
2 2D metric   프레임 요약의 요약(median of medians)이었다. 코너 분포가 아니다
3 6D 경로     frozen predict_pose_without_gt 대신 자체 W/D selector 를 썼다
4 coverage    0.90 임계를 사전 등록 없이 평가 코드에서 정했다
```

전부 method 가 아니라 **내 평가**의 문제다.

## 이번 단계 범위

teacher inference 0 · R0 cache 재생성 0 · LK 재실행 0 · refinement 재실행 0 ·
tracklet/window/FB/median 변경 0 · student 학습 0 · depth 0 ·
**기존 refinement 좌표 수정 0**.  eligibility 감사 + 재집계 + frozen evaluator 재채점만.

## 판단 지표

```
2D   pooled corner median · p90 · gross20  (총 코너 수 함께 보고)
6D   frozen predict_pose_without_gt 경로.  PoseCov · R · yaw · t · IoU3D · ADDsym AUC
불확실성  frame paired bootstrap + recording-cluster paired bootstrap, cluster K 명시
coverage  숫자만 보고, 사후 임계 라벨 금지
```

## 완화 금지

formal eligible N 이 0 이거나 평가 불가하면 **POPULATION_LIMITED** 로 끝낸다.
쓸 만한 N 을 만들려고 eligibility 를 완화하지 않는다.  결과를 보고 오차 큰
recording 을 새로 빼지 않는다 — 사전 기록된 known-broken 만 제외한다.

기존 109장 결과는 지우지 않고 `EXPLORATORY_DIAGNOSTIC_ONLY` 로 분류한다.
그 `FAILED_TO_IMPROVE` 를 preregistered formal result 라고 부르지 않는다.

NEXT_ACTION = PAPER_FRAMING_DECISION
