---
license: agpl-3.0
tags:
  - object-detection
  - pose-estimation
  - keypoint-detection
  - ultralytics
  - yolo
  - pallet
  - symmetry
library_name: ultralytics
pipeline_tag: keypoint-detection
---

# pallet-pose-yolo26n-c4

정사각 팔레트의 **C4 대칭(yaw 90도 등가)** 을 학습 손실에서 인정하도록 만든
YOLO26n-pose 모델이다.
[`CanelE452/pallet-pose-yolo26n-livegt`](https://huggingface.co/CanelE452/pallet-pose-yolo26n-livegt)
에서 이어서 미세조정했다.

## 무엇을 바꿨나

**keypoint 정의는 바뀌지 않았다.**  9 keypoint · 0~3 앞면 · {0,1,4,5} 위 /
{2,3,6,7} 아래 · 8 centroid · camera-facing 규약 그대로다.  바뀐 것은 **학습 중
정답으로 인정하는 라벨 집합**이다.

1.10 × 1.10 m 정사각 팔레트는 네 면이 물리적으로 같아서 "어느 면이 앞면인가" 가
임의다.  기존 indexed loss 는 코너를 정확히 맞춘 예측에도 번호가 90도 돌았다는
이유로 큰 벌점을 준다.  그래서 네 개의 cuboid permutation 중 손실이 최소인 것을
target 으로 삼는다.

```
Lsym = min( L(pred, GT_0deg), L(pred, GT_90deg),
            L(pred, GT_180deg), L(pred, GT_270deg) )
```

permutation 은 손으로 적지 않고 3D vertex 를 Ry(θ) 로 회전시켜 좌표 매칭(<1e-9)으로
유도했다.  bijection · centroid 고정 · top/bottom 보존 · cuboid 12-edge 보존 ·
군 성질(90x2=180, 90x4=identity, inverse) 을 전수 검증했다.

```
  0도  [0, 1, 2, 3, 4, 5, 6, 7, 8]
 90도  [1, 5, 6, 2, 0, 4, 7, 3, 8]
180도  [5, 4, 7, 6, 1, 0, 3, 2, 8]
270도  [4, 0, 3, 7, 5, 1, 2, 6, 8]
```

**하지 않은 것**: nearest-neighbor 매칭, Hungarian 매칭, keypoint 별 독립 순열.
선택 단위는 object 하나이며, 같은 object 를 담당하는 positive anchor 는 모두 같은
branch 를 쓴다.  좌표와 visibility 는 한 번의 gather 로 함께 이동한다.
box / cls / dfl / anchor assignment 는 stock 그대로다.

## 성능

같은 촬영 분포에서 6장마다 1장을 뺀 held-out 155 프레임.  두 가지를 **분리해서**
본다 — 기존 번호 그대로(fixed-index)와, 90도 배수 회전을 같은 정답으로 인정한 것
(C4-equivalent).

| | 검출 | fixed median | fixed >20px | C4 median | C4 >20px | true collapse |
|---|---|---|---|---|---|---|
| livegt (indexed, base) | 154/155 | **1.90 px** | **3.2%** | 1.90 px | 1.3% | 2 |
| **이 모델 (C4)** | 154/155 | 2.90 px | 7.1% | 2.64 px | **0.0%** | **0** |

`true collapse` 는 네 permutation 중 최선으로 봐도 20px 를 넘는, 즉 **실제로 위치를
못 맞춘** 프레임 수다.  이 모델은 0 이다.

fixed-index 지표는 base 가 낫다.  이 모델이 155 장 중 11 장에서 다른 면을 앞면으로
고르기 때문이며, 그 프레임들도 C4 기준으로는 정확하다.

### 프레임 간 안정성 (GT 미사용, 연속 촬영)

이웃한 두 프레임의 예측끼리만 비교해 "다음 프레임이 이전 대비 몇 도 돌아 보이는가"
를 셌다.  연속 촬영이면 정답은 항상 0도다.

| 세션 | livegt | 이 모델 |
|---|---|---|
| forklift 142318 | 100.0% | 100.0% |
| forklift 103429 | 100.0% | 97.2% |
| handheld 20260902 | 100.0% | 100.0% |

한 접근 시퀀스 안에서는 앞면 선택이 유지된다.

## 추론

`inference_config.yaml` 이 정본이다.  **추론 방식은 base 와 동일하다** — C4 는 학습
시에만 쓰인다.

```python
from ultralytics import YOLO
import cv2

model = YOLO("pallet_yolo26n_pose_c4.pt")
img = cv2.imread("frame.png")                      # 640x480 BGR
padded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)

r = model.predict(padded, imgsz=640, conf=0.4, verbose=False)[0]
if len(r.boxes):
    kp = r.keypoints.data[0][:, :2].cpu().numpy() - 100   # 패딩 보정
    # kp -> solvePnP.  objectPoints 는 1.10 x 0.15 x 1.10 (정사각)
```

`ultralytics >= 8.4.60` 이 필요하다 (YOLO26 `Pose26` head).

## ★ downstream 을 붙이기 전에 확인할 것

PnP 의 `objectPoints` 원점이 **전면 중심**이면, 앞면이 90도 바뀔 때 `tvec` 이 인접
면 중심으로 옮겨간다 — 1.10 m 정사각 기준 `0.55·√2 ≈ 0.78 m` 다.

한 시퀀스 안에서는 앞면이 유지되므로 정렬 루프 도중 튀지는 않는다.  그러나
**세션 사이에는 달라질 수 있다**.  정사각 팔레트는 네 면 어디로도 포크가 들어가므로
그 자체가 물리적 오류는 아니지만, 절대 yaw 를 기록/비교하거나 이전 세션의 자세와
대조하는 로직이 있다면 90도 배수 차이를 감안해야 한다.

또한 `objectPoints` 는 반드시 **1.10 × 0.15 × 1.10** 을 쓸 것.  직사각 값으로 풀면
keypoint 품질과 무관하게 자세가 흔들린다.

## 학습

```
base        pallet-pose-yolo26n-livegt
데이터       수동 라벨 851장 기준 train 2,088 / val 155
            = 원본 696 + 좌우 flip 696 + 센서 노이즈 696
            truncation crop 은 제외했다 (아래 한계 참조)
            val 파생본은 전부 제외 — leakage 실측 0
loss        ChallengeC4PoseLoss (one2many / one2one 양쪽)
epochs 40   batch 32   imgsz 640   SGD lr0 0.01  lrf 0.01  cos_lr  warmup 3
augment     mosaic 0.3  close_mosaic 10  scale 0.25  fliplr 0.0  flipud 0.0
            hsv [0.015, 0.5, 0.35]
장비        RTX 3080, 10.2분
```

학습 중 선택된 branch 분포 (마지막 epoch, one2many):
`identity 10,608 · 90도 2,290 · 180도 0 · 270도 7,980`.
한쪽으로 붕괴하지 않았다.

## 알려진 한계

- **소표본이다.**  base 대비 개선(true collapse 2→0, C4 gross20 1.3%→0.0%)은 전부
  한 자릿수 사건이고 seed 하나짜리 실험이다.
- **truncation crop 을 뺐다.**  base(livegt)는 crop 996장을 포함해 화면 잘림에
  강했다.  이 모델은 그 데이터가 없으므로 **잘린 프레임 성능은 base 보다 낮을 수
  있다**(별도 측정하지 않았다).
- 같은 촬영 분포 기준 성능이다.  val 이 train 과 같은 세션에서 뽑혀 있어 처음 보는
  현장 일반화는 검증되지 않았다.
- 정사각 팔레트 전용이다.  직사각 팔레트에 C4 를 적용하면 90도 회전이 실제 대칭이
  아니므로 거짓 등가를 학습하게 된다.
- fixed-index 지표는 base 가 낫다.  "0~3 = 특정 면" 을 가정하는 파이프라인이라면
  base 를 쓰는 편이 안전하다.

## 라이선스

AGPL-3.0 (Ultralytics 파생).
