# PURPOSE — paper_temporal_selftrain_v1

[소비처]
논문의 마지막 positive-method 후보를 열지 말지의 근거.  depth track 이 main
method 후보에서 중단됐으므로, **원 논문 조건(RGB only)을 그대로 지키는** 방향이
하나 남았다.  이 pilot 의 결과는 사용자 판단으로 소비되며, 좋아도 student 학습이
자동으로 시작되지 않는다.

[문장]
"서로 다른 실제 RGB 프레임의 temporal 대응을 쓰면 R0 teacher 의 잘못된 keypoint
좌표 자체가 GT 에 더 가까워지고, 알려진 cuboid 로 제약하면 더 나아지며, 그 개선이
downstream 6D 까지 전달된다" — 또는 그 반증.

## 왜 이 방향인가

R1~R5 · V2~V5 는 전부 **어떤 pseudo-label 을 쓸지** 만 바꿨고, 채택된 라벨의 2D
좌표는 teacher 예측을 그대로 supervision 으로 썼다.  그래서 teacher 의 구조적
keypoint 오차를 student 가 그대로 물려받는다.  이번엔 **좌표 자체를** 고친다.

## 이번 단계 범위

student 학습 0 · pseudo-real 데이터셋 0 · 새 checkpoint 0 · teacher refresh 0 ·
threshold sweep 0 · tracklet sweep 0 · flow parameter sweep 0 · depth 0.
평가 GT 를 refinement 입력으로 쓰지 않는다(생성기와 평가기를 물리적으로 분리).

## 판단 지표 (결과 보기 전 고정 — TEMPORAL_METHOD_LOCK.json 과 동일)

```
2D   corner median px · corner p90 px · gross20
6D   pose coverage · IoU3D median · ADDsym AUC (+ R/yaw/t)
결정적 넷   p90 · gross20 · IoU3D · ADDsym AUC
불확실성    frame paired bootstrap + recording-cluster paired bootstrap
```

metric 정의는 전부 기존 것 재사용 — 새 정의 금지.

## 적용범위 한계 (측정 전에 선언)

PAPER_EVAL 319 에 프레임을 대는 recording 을 통째로 빼면 **주간이 하나도 안 남는다**.
이 pilot 은 야간·plastic 만의 조각을 재는 것이고, 결과는 그 범위로만 읽어야 한다.

## 실패 시

`TEMPORAL_METHOD_PILOT = FAILED_TO_IMPROVE` 로 끝낸다.  window 를 바꾸거나
tracklet 을 늘리거나 flow 를 학습 모델로 갈아끼우지 않는다.  그 다음 새 method
탐색도 열지 않는다.

NEXT_ACTION = USER_REVIEW_TEMPORAL_PILOT
