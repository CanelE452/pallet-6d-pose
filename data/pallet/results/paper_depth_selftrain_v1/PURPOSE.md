# PURPOSE — paper_depth_selftrain_v1

[소비처]
논문의 새 method track "depth-assisted pseudo-label correction" 을 열지 말지
결정하는 근거.  현 단계(GATE 0)의 소비처는 **사용자 판단** 이며, 통과해도
자동으로 구현으로 넘어가지 않는다.  Gate 0 결과 = 이 track 을 계속할지 여부.

[문장]
"기존 unlabeled real RGB-D 촬영본의 depth 는 동기화·metric scale·calibration·
pallet ROI coverage 가 확보되어 있어 teacher 의 pseudo-label 좌표를 기하로
교정하는 데 쓸 수 있다" — 또는 그 반증.

## 왜 이 방향인가

지금까지의 self-training arm 은 전부 **어느 프레임을 남길지** 만 바꿨고
teacher 가 찍은 좌표 자체는 건드리지 않았다.  그래서 student 가 teacher 의
구조적 기하 오차를 그대로 물려받는다.  depth 는 teacher 와 독립인 측정이므로,
프레임 재선택이 아니라 **좌표 교정** 을 시도할 수 있는 첫 신호다.

## 이번 단계 범위 (GATE 0 — 센서 계약 감사만)

student training 0 · pseudo-label 생성 0 · 좌표 교정 0 · cuboid fitting 0 ·
ICP/RANSAC sweep 0 · 새 loss 0 · 새 architecture 0 · threshold tuning 0 ·
평가 metric 변경 0.  기존 V1~V5 · pose closure · SITE_A artifact 는 읽기 전용.

`GT_USED_FOR_GATE0 = false` — 평가 GT(2D keypoint · 6D pose · error) 를
어느 경로에서도 읽지 않는다.  기존 R0 prediction cache 는 **pallet ROI 정의
용도로만** 쓴다.

## 판단 지표 (결과 보기 전 고정 — METHOD_INTENT_LOCK.json 과 동일)

```
calibration   depth metric scale 확보 AND (aligned-to-color 확증 OR depth K + extrinsics)
pairing       recording 별 RGB-depth paired rate >= 0.95
ROI support   usable frame rate 전체 >= 0.70, Day >= 0.50, Night >= 0.50
              usable = valid depth pixel >= max(200, ROI 면적의 5%)
alignment     visual audit 에서 체계적 어긋남이 반복되면 자동 점수와 무관하게 FAIL
```

`usable support` 는 "정확한 pose fitting 가능" 이 아니라 "기하 fitting 을
시도할 최소 표본이 있다" 는 뜻이다.

## PASS 가 뜻하지 않는 것

"depth method 가 성능을 개선할 것이다" 를 뜻하지 않는다.  PASS 는
**실험할 기술적 근거가 생겼다** 까지다.  pseudo-label 교정 품질은 Gate 1 의
별도 질문이다.

NEXT_ACTION_AFTER_GATE0 = USER_REVIEW_DEPTH_GATE
