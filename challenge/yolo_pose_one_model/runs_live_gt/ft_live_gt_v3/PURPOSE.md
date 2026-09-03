# PURPOSE — 수동 어노 402장만으로 기존 keypoint 모델 FT

## [소비처]
forklift alignment FSM (`depth_cam/calib/fsm*`) 이 쓰는 keypoint → PnP → x,z,yaw 경로.
출력 구조는 그대로다. 가중치만 현장 팔레트에 맞춘다.

## [문장]
> 현장에서 직접 라벨링한 402장으로 이어서 미세조정하면, 배포 중인 YOLO26n-pose 가
> 이 팔레트·이 현장에서 keypoint 를 더 정확히 찍는다.

## 데이터
`challenge/data/01_real/live_capture_gt` (4-fold 정규화 완료) → `datasets/live_gt_v1`
```
train  handheld_20260902        344장
val    forklift_v4_20260901      58장   ← 촬영 단위 split (세션 누수 없음)
```
negative 도 synthetic 도 섞지 않는다 — 사용자가 "어노테이션한 것만" 으로 명시했다.

## 알려진 위험 (사용자 확인 후 진행)
- real 402장만이라 과적합·오검출 증가 가능. 선례 `runs_ft/ft_a` 는 real 157 에
  negative 259 + synthetic 12k 를 섞었다.
- val 이 train 과 다른 촬영(저앙각 rec)이라 도메인 갭이 있다. val 이 나쁘게 나와도
  그것이 곧 과적합의 증거는 아니다.

## 판정 지표
val 의 `pose mAP50-95` 와 `box mAP50-95` 를 base 대비 비교한다.
개선이 없으면 데이터가 부족한 것으로 보고 negative/synthetic 혼합으로 돌아간다.
