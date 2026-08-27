# Pallet YOLO26n-Pose — 모델 카드

생성 2026-08-26 · Windows(RTX 4070) · ultralytics 8.4.60 / torch 2.1.1+cu118

이 패키지에는 **실측으로 가장 좋았던 2개**만 담았다. 형식은 ultralytics `.pt` 다(`.pth` 아님).

## 무슨 모델인가

```
task          pose (keypoint regression)
아키텍처       YOLO26n-pose 순정 (Pose26 head, 개조 0)  params 3,043,704
클래스         nc=1 (pallet, single_cls 라 이름은 'item' 으로 저장됨)
keypoint      9점 — camera-facing cuboid 8 corner + centroid(index 8)
kpt_shape     [9, 3]   flip_idx [1,0,3,2,5,4,7,6,8]
입력           imgsz 640
init          yolo26n-pose.pt 공식 v8.4.0 (sha eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9)
```

## 두 모델의 차이 — 학습 데이터 하나만 다르다

```
Y0E   G38 38,002 + 그중 9,000장 1회 반복      노출 47,002  고유 38,002
YN    G38 38,002 + synthetic negative 9,000   노출 47,002  고유 47,002

공통  30ep · batch 32 · imgsz 640 · SGD lr0 0.01 lrf 0.01 cos_lr · seed 42
      mosaic 0.3 · scale 0.25 · fliplr/flipud 0 · erasing 0.4 · close_mosaic 10 · warmup 3 · patience 0
      architecture 변경 0 · loss 변경 0 · steps/epoch 1,469 (동일)
안 씀  real 학습 0 · pseudo-label 0 · self-training 0 · 평가용 real negative 2,689 미투입
```
두 모델은 총 노출량을 47,002 로 맞춘 **matched pair** 다. 차이는 '추가 9,000장이 positive 반복이냐 negative 냐' 하나뿐이다.

## 성능 (실측)

### 합성 G38 val (n=1,998) — 구분 안 됨
```
모델     box mAP50  box mAP50-95  pose mAP50  pose mAP50-95
Y0E       0.9938        0.9311      0.9471         0.9141
YN        0.9938        0.9253      0.9472         0.9121
```
합성 val 은 이미 포화라 모델 선택 근거가 못 된다. 실제 차이는 real 에서만 난다.

### real positive 128 (DAY 100 / NIGHT 28)
```
모델    dom     top1-cbox  any-cbox  det recall  det@0.40  kp med   kp p90
Y0E   ALL        0.8828    0.9688      1.0000    0.7812   12.07    74.54
Y0E   DAY        0.9400    0.9900      1.0000    0.8700   10.57    59.86
Y0E   NIGHT      0.6786    0.8929      1.0000    0.4643   23.75   137.61
YN    ALL        0.8125    0.9219      0.9766    0.6094   11.54    92.55
YN    DAY        0.9200    0.9800      0.9800    0.6900   11.01    94.09
YN    NIGHT      0.4286    0.7143      0.9643    0.3214   15.32    52.86
```
지표 정의는 원 프로젝트 정본 `cf_real_eval.py` 를 그대로 따랐다:
cbox = IoU(pred box, GT cuboid bbox) >= 0.5 · kp err = 8 corner L2(px) · recipe = pad 100 + BORDER_REFLECT_101, imgsz 640, conf 0.001, top-1 by box conf

### real negative heldout 2,689 (positive = real128)
```
모델         AP   AUROC  FPR@TPR95  FP/img@0.40  conf p95  conf p99
Y0E    0.7625  0.9680     0.1473       0.0431    0.3627    0.7439
YN     0.7360  0.9471     0.3444       0.0126    0.0831    0.4567
```

### 같은 recall 로 맞췄을 때의 FP/image
```
 correct-box recall  Y0E FP/img  YN FP/img        차이
               0.50      0.0004     0.0004   +0.0000
               0.60      0.0074     0.0179   +0.0104
               0.70      0.0320     0.0532   +0.0212
               0.75      0.0536     0.0837   +0.0301
               0.80      0.1090     1.1175   +1.0086
               0.85   (양쪽 다 달성 불가)
               0.90   (양쪽 다 달성 불가)
               0.95   (양쪽 다 달성 불가)

recall 상한   Y0E det 1.0000 / cbox 0.8828
              YN  det 0.9766 / cbox 0.8125
```

## 어느 걸 쓸까 — Y0E

**Y0E 를 기본으로 쓴다.** real 전 항목에서 앞선다.

YN 의 `FP/img@0.40 = 0.0126` 하나만 Y0E(0.0431)보다 좋아 보이는데, 이건 실제 개선이 아니다.
YN 은 negative 뿐 아니라 **positive confidence 까지 눌렀다** (real128 top-1 conf median 0.8808 → 0.6870).
그래서 같은 임계 0.40 에서는 FP 가 줄어든 것처럼 보이지만, **같은 recall 로 맞추면 전 구간에서 Y0E 가 이긴다**(위 표).
threshold-free 지표도 YN 이 열세다 — AP 0.7625 → 0.7360, AUROC 0.9680 → 0.9471.
YN 은 recall 상한 자체도 낮다 (cbox 0.8828 → 0.8125).

YN 을 굳이 쓴다면: **conf 임계를 재조정할 수 없는 고정 파이프라인에서 FP 를 낮추는 것이 recall 보다 중요할 때**뿐이다.

## 쓰는 법

```python
from ultralytics import YOLO
m = YOLO('weights/Y0E_best.pt', task='pose')
r = m.predict('frame.png', imgsz=640, conf=0.25, device=0)[0]
kp = r.keypoints.xy.cpu().numpy()   # (N, 9, 2)  0~7 = cuboid corner, 8 = centroid
bx = r.boxes.xyxy.cpu().numpy()     # (N, 4)
```
커스텀 모듈 불필요 — 순정 ultralytics 로 그대로 로드된다.
`names` 가 `item` 으로 나오는 건 `single_cls=True` 로 학습해서다. 클래스는 pallet 하나뿐이다.

## 한계 — 쓰기 전에 반드시 읽을 것

1. **real 로 학습하지 않았다.** 전부 합성 데이터(G38)만 썼다. 논문용 ablation 목적이라 real fine-tune 이 금지된 조건이었다.
   실배포용으로는 real fine-tune 한 별도 모델(challenge 트랙 `yolo26n_pose_v1_ft_*`)이 따로 있다.
2. **NIGHT 이 약하다.** top1-cbox DAY 0.94 vs NIGHT 0.68 (Y0E). 야간 프레임은 신뢰도가 크게 떨어진다.
3. **real128 의 DAY 12장이 학습 세션에서 유출**돼 있다(NIGHT 0장). DAY 수치는 낙관 편향이다.
   근거: 원 데이터 번들의 `eval_positive_real128/provenance/FT_EVAL_LEAK.json`.
4. **6D pose 절대정확도는 측정하지 않았다.** 여기 수치는 전부 2D keypoint/box 기준이다.
   PnP 로 6D 를 풀 경우 flat 팔레트의 광축 depth 가 약제약이라 ADD/5cm5° 는 별도 검증이 필요하다.
5. **평가 표본이 작다.** real positive 128 (NIGHT 28). 신뢰구간이 넓다.
6. `mosaic=0.3` 때문에 YN 학습 시 negative 의 약 19.7% 가 positive 와 합성됐다(마지막 10 epoch 은 mosaic OFF 라 순도 100%).

## 이 패키지에 없는 것

- **Y2 (quality residual head)** — 제외했다. 로드에 커스텀 모듈(`pose26_quality.py`)이 필요하고,
  real 에서 Y0E 를 넘지 못했으며 negative 거절은 오히려 악화했다(AP 0.7092, FPR@TPR95 0.2692).
- **Y0 (vanilla, G38 38,002 그대로)** — 아직 학습하지 않았다. Y0E 는 노출량 대조군이지 baseline 이 아니다.
- 학습 데이터셋 자체(용량 문제).

## 파일
```
  weights/Y0E_best.pt
  Y0E_results.csv
  Y0E_RUNTIME_AUDIT.json
  weights/YN_best.pt
  YN_results.csv
  YN_RUNTIME_AUDIT.json
  MODEL_CARD.md
  SHA256SUMS.txt
```
