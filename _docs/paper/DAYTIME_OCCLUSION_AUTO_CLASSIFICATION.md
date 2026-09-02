# Daytime occlusion — automatic classification

630 개 keypoint 를 전부 사람이 보게 하지 않는다.  기하로 결정되는 것은 기하로 정했다.

## 규칙 (사전 고정)

```text
truncation      GT v2 의 기존 규칙.  in_frame == False -> AUTO_TRUNCATED
                사람에게 묻지 않는다.
self-occlusion  back-face culling.  코너가 속한 세 면 중 하나라도 정면이면 보인다.
                signed-axis 후보 둘을 모두 풀어 일치할 때만 확정한다.
external        depth 로 판단.  expected - observed > max(0.15 m, 0.04 x expected)
                threshold 는 센서 노이즈에서 왔고 모델 결과와 무관하다.
M5 Occlusion    external only.  self-occlusion 을 그 태그에 넣지 않는다.
```

## 결과

```text
total keypoints                        630
AUTO_CENTROID_OCCLUDED                 70
AUTO_SELF_OCCLUDED                     95
AUTO_TRUNCATED                         21
EXTERNAL_OCCLUSION_CANDIDATE           119
SELF_VISIBLE_CANDIDATE                 325

requires_human                         119  (18.9%)
truncation 선언 불일치                      0
```

## 남은 사람 작업

119 개다.  630 개 전부가 아니다.

`UNKNOWN_SELF_VISIBILITY_DISAGREES` 는 signed-axis 가 미해결이라 두 후보의
self-visibility 가 엇갈리는 코너다 — pose selector 가 풀리면 자동으로 줄어든다.

`EXTERNAL_OCCLUSION_CANDIDATE` 는 depth 가 가림을 시사하는 코너다.  자동으로
occluded 로 확정하지 않고 사람이 확인한다 — depth 노이즈와 실제 가림을
센서만으로 가르지 않는다.

다만 이 신호는 사람이 매긴 프레임 태그와 잘 맞는다.  ext 후보가 있는 64 프레임 중
62 개가 `occlusion=medium` 이다 (Daytime 70 중 medium 은 65).  임계값이 헛돌고
있지 않다는 교차 검증이다.

`AUTO_CENTROID_OCCLUDED` 는 새 규약이 아니다.  이 저장소의 신규 어노 146 장이
예외 없이 centroid 를 visibility=1 / source=centroid_auto 로 적는다 — 물체 내부의
점이라 직접 보이는 일이 없기 때문이다.  그 규약을 그대로 적용했다.

```text
queue   data/evaluation/pallet_eval_v1/review/DAYTIME_OCCLUSION_REVIEW_QUEUE.csv
```
