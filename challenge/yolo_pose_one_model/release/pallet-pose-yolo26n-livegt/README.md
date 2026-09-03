---
license: agpl-3.0
tags:
  - object-detection
  - pose-estimation
  - keypoint-detection
  - ultralytics
  - yolo
  - pallet
library_name: ultralytics
pipeline_tag: keypoint-detection
---

# pallet-pose-yolo26n-livegt

지게차 현장에서 직접 라벨링한 팔레트 402장 + 그로부터 만든 truncation crop 996장으로
[`CanelE452/pallet-pose-yolo26n-ft`](https://huggingface.co/CanelE452/pallet-pose-yolo26n-ft)
를 이어서 미세조정한 YOLO26n-pose 모델이다.

**화면 가장자리에서 잘린 팔레트에 강하다.**  지게차가 접근하며 패닝할 때 팔레트가
프레임을 벗어나는 상황을 겨냥했다.

## 성능

같은 촬영 분포에서 6장마다 1장을 뺀 held-out 으로 잰다.  `clean` 은 원본 프레임,
`truncated` 는 그 held-out 원본을 잘라 만든 것이다 (학습에 쓰지 않았다).

**clean (70 프레임)**

| | box mAP50-95 | pose mAP50-95 |
|---|---|---|
| base (`-ft`) | 0.8877 | 0.3523 |
| **이 모델** | **0.9552** | **0.9703** |

**truncated (210 프레임)** — 이 모델을 만든 이유

| | box mAP50-95 | pose mAP50 | pose mAP50-95 | box recall |
|---|---|---|---|---|
| base (`-ft`) | 0.5318 | 0.2019 | 0.1376 | 0.7657 |
| crop 없이 학습 | 0.5000 | 0.4630 | 0.2847 | 0.6880 |
| **이 모델** | **0.8788** | **0.9457** | **0.9002** | **0.9274** |

잘린 프레임에서 pose mAP50-95 가 base 대비 **6.5배**, crop 없이 학습한 것 대비
**3.2배**다.  clean 에서는 crop 없는 쪽이 0.9837 로 근소하게 높다 — crop 을 넣으면
clean 정밀도를 조금 내주고 truncation 강건성을 크게 얻는 교환이다.

> ⚠️ 위 수치는 **같은 촬영 분포**에서 잰 것이다.  처음 보는 현장·조명·팔레트에서의
> 일반화를 보장하지 않는다.  이 모델은 특정 현장에 맞추는 것이 목적이었다.

## 대상 물체가 base 와 다르다

| | base (`-ft`) | 이 모델 |
|---|---|---|
| 팔레트 | 1.10 × 1.30 × 0.11 m (직사각) | **1.10 × 1.10 × 0.15 m (정사각)** |
| 대칭 | 180° 등가 | **90° 등가 (4방향 포크 진입)** |

정사각이라 yaw 가 90° 배수로 모호하다.  학습 라벨은 "어느 면을 앞면(keypoint 0~3)으로
볼지" 를 한 규칙으로 통일해 두었다 — 90° 회전은 등가이므로 정보 손실 없이 통일된다.
서로 다른 두 사람이 라벨링한 데이터를 이 규칙으로 정렬해 합쳤다.

## 추론 계약 — 지키지 않으면 성능이 나오지 않는다

`inference_config.yaml` 이 정본이다.  핵심 두 가지:

**1) 100px reflect padding 이 필수다.**  학습 이미지가 전부 그렇게 만들어졌다.

```python
import cv2
padded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
# 예측 좌표에서 100 을 빼고 원본 K 로 기하 계산
```

**2) keypoint 는 camera-facing 0123 규약이다** (물체 고정이 아니다).

```
0~3  카메라를 향한 면      {0,1,4,5} 위 / {2,3,6,7} 아래
4~7  그 반대 면            8 centroid
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]
```

## 사용법

```python
from ultralytics import YOLO
import cv2

model = YOLO("pallet_yolo26n_pose_livegt.pt")
img = cv2.imread("frame.png")                      # 640x480 BGR
padded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)

r = model.predict(padded, imgsz=640, conf=0.4, verbose=False)[0]
if len(r.boxes):
    kp = r.keypoints.data[0][:, :2].cpu().numpy() - 100   # 패딩 보정
    # kp -> solvePnP 로 6DoF
```

`ultralytics >= 8.4.60` 이 필요하다 (YOLO26 `Pose26` head).  구버전은 로드 중 죽는다.

## 학습

```
base        pallet-pose-yolo26n-ft
데이터       수동 라벨 402장 + truncation crop 996장 = train 1,328 / val 70
            crop 방향  측면(L/R) 71.5% · 하단 24.5% · 상단 4.1%
            (지게차는 좌우로 패닝하므로 옆이 잘린다.  위쪽 잘림은 거의 배제)
            held-out 원본에서 파생된 crop 210장은 train 에서 제외 (누수 차단)
            negative·synthetic 미포함
epochs 40   batch 32   imgsz 640   SGD lr0 0.01  lrf 0.01  cos_lr  warmup 3
augment     mosaic 0.3  close_mosaic 10  scale 0.25  fliplr 0.0  flipud 0.0
            hsv [0.015, 0.5, 0.35]
장비        RTX 3080, 6.9분 (10.4초/epoch)
```

augment 값은 base 모델의 학습 계약을 그대로 따랐다.  ultralytics 기본값
(`mosaic 1.0` / `fliplr 0.5`)으로 돌리면 소량 데이터에서 오히려 base 보다 나빠진다 —
실제로 그렇게 한 첫 시도는 pose mAP50-95 가 base 아래로 떨어졌다.

## 알려진 한계

- **같은 촬영 분포 기준 성능**이다.  다른 현장·다른 팔레트 일반화는 검증되지 않았다.
- 402장이라 배경·조명 다양성이 좁다.
- clean 프레임 정밀도는 crop 없이 학습한 쪽이 근소하게 낫다.
- 정사각 팔레트 전용 라벨 규약이라, 직사각 팔레트에는 yaw 해석이 달라진다.
- truncated 평가는 원본을 잘라 만든 합성 crop 이다.  실제로 잘려 찍힌 프레임과는
  통계가 다를 수 있다.

## 라이선스

AGPL-3.0 (Ultralytics 파생).
