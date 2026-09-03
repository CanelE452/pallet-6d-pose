# PURPOSE — sensor_validation_v1

[소비처]
depth-assisted correction track 을 계속할지 말지의 근거.  Gate 0B 가
`NOT_READY_FOR_GATE1` 로 막힌 이유가 depth 자체가 아니라 **acquisition 기록의
부재**였으므로, 그 공백을 문서 대신 실측으로 메울 수 있는지 본다.  결과는
사용자 판단으로 소비된다 — 통과해도 Gate 1 이 자동으로 열리지 않는다.

[문장]
"수동 어노 2D cuboid 키포인트 · 알려진 팔레트 치수 · depth · cam_K 가 함께 있는
프레임에서, 저장된 depth 는 그 RGB 기하와 metric 수준에서 양립하며, 알려진
치수는 팔레트 표면을 주변 지면·배경과 구분해 낸다" — 또는 그 반증.

## 이번 단계 범위

student training 0 · pseudo-label correction 0 · Gate 1 NOT RUN ·
new filter 0 · threshold tuning 0 · parameter sweep 0.
teacher 예측을 쓰지 않는다.  기준 기하는 **수동 어노 + 등록 치수** 다.

## 판단 지표 (결과 보기 전 고정 — SENSOR_VALIDATION_LOCK.json 과 동일)

```
Q1  저장된 depth·cam_K 가 수동 어노 RGB 기하와 양립하는가
Q2  알려진 치수가 팔레트 표면을 우연한 지면·배경 평면과 구분하는가

측정  면 내부 depth 통계 · 기준 cuboid 표면까지의 거리 잔차 ·
      ray 기대 z 대비 depth 잔차 · face vs ring · 수동 경계 ↔ depth 불연속 ·
      day/night K 별 일관성 · 고정 scale 0.001 의 양립성
주 통계  fraction(face residual < ring residual)
```

scale 은 **적합하지 않는다** — 0.001 만 평가하고 sweep·최적값 탐색 금지.

## 누수 계약

이 population 은 **development sensor validation 전용**이다.  나중에 method
성공의 독립 확인 population 으로 재사용하지 않는다(순환이 된다).
PAPER_EVAL 과 겹치면 `OVERLAPS_EXISTING_EVAL` 로 명시한다.

수동 라벨을 보고 shrink 비율·clipping·loss weight·hyperparameter 를 고치는 것
전부 금지.  "20% 가 안 좋으니 10%" 같은 수정이 정확히 막으려는 행동이다.

## PASS 가 뜻하지 않는 것

"depth 교정이 정확도를 올린다" 를 뜻하지 않는다.  저장된 depth 가 수동 어노 RGB
기하와 기하적으로 일관된다는 것까지다 — Gate 0B 가 기록으로는 얻지 못한 증거.

NEXT_ACTION = USER_REVIEW_SENSOR_VALIDATION
