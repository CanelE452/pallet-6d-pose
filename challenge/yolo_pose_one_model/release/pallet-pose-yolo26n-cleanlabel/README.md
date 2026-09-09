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

# pallet-pose-yolo26n-cleanlabel

정사각 팔레트용 YOLO26n-pose 모델.  앞선
[`pallet-pose-yolo26n-c4`](https://huggingface.co/CanelE452/pallet-pose-yolo26n-c4)
와 같은 C4 대칭 손실을 쓰되, **학습 라벨의 소스 필드를 교정하고 좌우 flip 증강을
제거**해 다시 학습했다.

## 무엇이 달라졌나 — 요약

이 모델의 이득은 **코너 번호의 일관성**이지 코너 위치의 정확도가 아니다.
그 둘을 섞어 읽으면 이 모델을 잘못 쓰게 된다.

| | 이전 모델들 | 이 모델 |
|---|---|---|
| 예측이 GT 와 같은 phase(0도)로 맞는 비율 | 50.6~57.1% | **98.7%** |
| fixed-index median | 4.59~25.40 px | **1.90 px** |
| fixed-index p90 | 171~181 px | **3.82 px** |
| C4-equivalent median (회전 무시) | 1.89~2.67 px | 1.88 px |

마지막 줄이 핵심이다.  회전을 무시하고 재면 이전 모델과 **거의 차이가 없다**.
달라진 것은 "어느 면을 0~3(앞면)으로 부를지" 가 프레임마다 흔들리지 않게 된 것이다.

## 왜 그런 차이가 났나 — 라벨 소스 교정

학습 GT JSON 안에 같은 코너 좌표가 두 벌 들어 있었다.  둘은 서로의 정확한 복사본인데
**인덱스 배정만 90도 다르다** (잔차 중앙값 0.000 px, 851장 중 46.8%).

```
필드                     camera-facing 0123 규약 위반
projected_cuboid          198 / 851   (23.3%)     <- 이전 학습이 쓰던 것
keypoint_annotations        0 / 847   ( 0.0%)     <- 이 모델이 쓰는 것
```

정사각(1.10 x 1.10 m) 팔레트라 yaw phase 를 정할 기하학적 근거가 없어 어노테이션이
두 규약을 남긴 것이다.  어느 쪽이 배포 규약인지는 **모델에게 물어서** 정했다 —
두 라벨을 각각 4개 회전시켜 기존 배포 모델의 예측과 거리를 쟀다.

```
                                k=0      k=90     k=180    k=270
keypoint_annotations 기준     3.71 px   98.97   157.73   101.24    <- k=0 이 최소
projected_cuboid 기준         8.29 px  143.50   141.13    66.08
```

배포 모델은 `keypoint_annotations` 를 본 적이 없는데도 그쪽과 2.2배 더 잘 맞는다.
이것이 정본 필드를 고른 근거다.

## 좌우 flip 증강을 뺀 이유

이전 학습은 train 의 3분의 1이 좌우 flip 본이었다.  flip 자체는 정확히 구현돼
있었지만(x 미러 + 좌우 짝 순열 `(1,0,3,2,5,4,7,6)`, centroid 불변), **거울 반사는
C4 회전군의 원소가 아니다**.  실측에서 원본이 phase 270 이던 프레임이 flip 본에서는
phase 90 으로 반전됐다.  배포 장면에 거울상 팔레트는 없으므로 뺐다.

대신 확대축소·색감·밝기를 on-the-fly 로 올렸다 (`scale` 0.25→0.40,
`hsv_v` 0.35→0.50, `hsv_s` 0.50→0.60).  파일 증강과 달리 매 epoch 다른 변형이 걸린다.

## 대칭 손실 (이전 모델과 동일)

**keypoint 정의는 바뀌지 않았다.**  9 keypoint · 0~3 앞면 · {0,1,4,5} 위 /
{2,3,6,7} 아래 · 8 centroid · camera-facing 규약 그대로다.

```
Lsym = min( L(pred, GT_0deg), L(pred, GT_90deg),
            L(pred, GT_180deg), L(pred, GT_270deg) )
```

permutation 은 3D vertex 를 Ry(θ) 로 회전시켜 좌표 매칭(<1e-9)으로 유도했고
bijection · centroid 고정 · top/bottom 보존 · 12-edge 보존 · 군 성질을 전수 검증했다.

```
  0도  [0, 1, 2, 3, 4, 5, 6, 7, 8]
 90도  [1, 5, 6, 2, 0, 4, 7, 3, 8]
180도  [5, 4, 7, 6, 1, 0, 3, 2, 8]
270도  [4, 0, 3, 7, 5, 1, 2, 6, 8]
```

**하지 않은 것**: nearest-neighbor 매칭, Hungarian 매칭, keypoint 별 독립 순열.
선택 단위는 object 하나이고, 좌표와 visibility 는 한 번의 gather 로 함께 이동한다.
box / cls / dfl / anchor assignment 는 stock 그대로다.

학습 중 선택된 branch 분포 (마지막 epoch, one2many):
`identity 6,230 · 90도 730 · 180도 0 · 270도 0` — identity 89.5% 로, 라벨이 일관된
덕에 이전 모델(identity 51%, 270도 38%)보다 훨씬 한쪽에 모였다.

## ★ C4 손실의 이득은 이 데이터에서 확인되지 않았다

정직하게 적는다.  같은 데이터로 C4 를 **끄고** 학습한 대조군이 근소하게 더 좋았다.

| | fixed median | C4-equivalent median | pose mAP50-95 |
|---|---|---|---|
| indexed only (대조군) | 1.80 px | 1.79 px | 0.978 |
| **이 모델 (C4)** | 1.90 px | 1.88 px | 0.960 |

차이 0.09 px 는 실험 전에 정해둔 판정 문턱 0.15 px 미만이라 **구분되지 않는다**
(NOT_ESTABLISHED — 해롭다는 뜻이 아니다).  라벨 위반을 23.3%에서 0.8%로 낮추고 나니
C4 가 흡수할 모호성이 거의 남지 않은 것으로 보인다.

**대칭 처리가 downstream 에 필요 없다면 C4 없이 학습한 편이 약간 낫다.**  이 모델은
"대칭 물체를 대칭으로 다룬다" 는 설계를 유지하고 싶을 때 쓴다.

## 성능

같은 촬영 분포에서 6장마다 1장을 뺀 held-out 155 프레임.  네 모델을 **모두 같은
val** 로 채점했다.

| | 검출 | fixed median | fixed p90 | C4 median | true collapse | phase0 비율 |
|---|---|---|---|---|---|---|
| **이 모델** | **155/155** | **1.90 px** | **3.82 px** | 1.88 px | 1 | **98.7%** |
| yolo26n-c4 | 154/155 | 4.59 px | 180.55 px | 2.67 px | 1 | 57.1% |
| livegt (v4) | 154/155 | 25.40 px | 171.48 px | 1.89 px | 4 | 50.6% |

`true collapse` = 네 permutation 중 최선으로 봐도 20px 를 넘는, 실제 위치 실패.

**이 표를 읽을 때 주의**: 이전 두 모델의 fixed-index 수치가 나쁜 것은 그들이 옛
라벨 규약으로 학습됐는데 여기서는 새 규약으로 채점했기 때문이다.  그들이 갑자기
나빠진 것이 아니라 **규약이 다른 것**이다.  규약에 무관한 C4-equivalent 열을 보면
셋이 1.88~2.67 px 로 비슷하다.

## 추론

`inference_config.yaml` 이 정본이다.  추론 방식은 이전 모델들과 동일하다 — C4 는
학습 시에만 쓰인다.

```python
from ultralytics import YOLO
import cv2

model = YOLO("pallet_yolo26n_pose_cleanlabel.pt")
img = cv2.imread("frame.png")                      # 640x480 BGR
padded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)

r = model.predict(padded, imgsz=640, conf=0.4, verbose=False)[0]
if len(r.boxes):
    kp = r.keypoints.data[0][:, :2].cpu().numpy() - 100   # 패딩 보정
    # kp -> solvePnP.  objectPoints 는 1.10 x 0.15 x 1.10 (정사각)
```

**100 px reflect padding 은 필수다.**  없이 추론하면 검출률이 크게 떨어진다.
`ultralytics >= 8.4.60` 이 필요하다 (YOLO26 `Pose26` head).

keypoint 순서는 camera-facing 0123 이다 — `0 near_top_left, 1 near_top_right,
2 near_bottom_right, 3 near_bottom_left, 4~7 far 면 동순, 8 centroid`.
좌우 비대칭이므로 학습·추론 모두 `fliplr` 을 켜면 안 된다.  파일 단위로 flip 할
때의 순열은 `flip_idx = [1, 0, 3, 2, 5, 4, 7, 6, 8]` 이다.

## downstream 을 붙이기 전에

PnP 의 `objectPoints` 원점이 **전면 중심**이면, 앞면이 90도 바뀔 때 `tvec` 이 인접
면 중심으로 옮겨간다 — 1.10 m 정사각 기준 `0.55·√2 ≈ 0.78 m`.  이 모델은 phase0
비율이 98.7% 라 이전 모델들보다 이 위험이 훨씬 낮지만 0 은 아니다.

`objectPoints` 는 반드시 **1.10 × 0.15 × 1.10** 을 쓸 것.  직사각 값으로 풀면
keypoint 품질과 무관하게 자세가 흔들린다.

## 학습

```
1단계  pallet-pose-yolo26n-ft -> [indexed FT 120ep] -> 중간 base
2단계  중간 base              -> [C4 FT 120ep]      -> 이 모델

2단계로 나눈 이유: C4 는 손실이 아니라 초기값이 face phase 를 정한다.  라벨 규약이
바뀌었으므로 그 규약을 indexed 로 먼저 익힌 base 가 필요했다.

데이터   수동 라벨 851장 -> train 696 / val 155 (6장마다 1장을 val)
        파일 증강 없음 — flip·노이즈·truncation crop 전부 제외
        라벨 소스 keypoint_annotations (규약 위반 0.8%)
loss    ChallengeC4PoseLoss (one2many / one2one 양쪽)
epochs 120  batch 32  imgsz 640  SGD lr0 0.01  lrf 0.01  cos_lr  warmup 3
augment mosaic 0.3  close_mosaic 10  scale 0.40  fliplr 0.0  flipud 0.0
        hsv [0.015, 0.6, 0.5]
장비    RTX 3080, 2단계 합쳐 22.6분
```

120 epoch 은 이전 모델(train 2,088 × 40ep)과 총 업데이트 수를 맞춘 값이다.

## 알려진 한계

- **처음 보는 현장 일반화는 검증되지 않았다.**  val 155 가 train 과 같은 세션에서
  6장마다 뽑혀 있다.  이것은 이 프로젝트(특정 현장·특정 팔레트에 맞추는 것)의
  의도된 설정이지만, 새 현장 성능의 근거로 쓰면 안 된다.
- **잘린 팔레트에 약하다.**  truncation crop 을 학습에 넣지 않았다.  화면 가장자리에
  걸린 팔레트는 별도로 측정하지 않았으므로 crop 을 포함해 학습한 모델보다 낮을 수
  있다.
- **소표본이다.**  개선 판정의 근거인 collapse·미검출 차이는 한 자릿수 사건이고
  seed 하나짜리 실험이다.
- **오탐률은 in-sample 수치가 아니라 아예 측정하지 않았다.**  negative 프레임에 대한
  false positive 를 이 실험에서 재지 않았다.
- 정사각 팔레트 전용이다.  직사각 팔레트에 C4 를 적용하면 90도 회전이 실제 대칭이
  아니므로 거짓 등가를 학습하게 된다.
- 위에 적었듯 **C4 손실 자체의 이득은 확인되지 않았다.**  이 데이터에서는 indexed
  단독이 근소하게 나았다.

## 라이선스

AGPL-3.0 (Ultralytics 파생).
