# challenge0123

## 학습 설정

```
Weight:      weights/challenge0123/final_net_epoch_0060.pth
초기 weight: scratch (VGG-19 ImageNet pretrained)
Epochs:      60
Batch size:  8
LR:          1e-4
Sigma:       4.0
Image size:  448
Workers:     4
Seed:        2657
```

## 학습 데이터

```
data/pallet/training_data/mixed_v8_train
challenge/data/02_synthetic/training/v1
challenge/data/02_synthetic/training/v2
```

## Loss 설정

```
symmetric_loss : False
struct_loss    : False
geo_loss       : False
rel_loss       : False
```

## 메모

- header 의 `outf` 는 `weights/challenge_camfacing_v4` 였으며, 학습 후 폴더를 `challenge0123` 으로 **rename** 함
- camera-facing convention 변경 흐름의 **v4 시점** 모델
  - v3 까지의 데이터셋별 camera convention 차이 (mixed_v8 = OpenCV +z forward vs v1/v2 = USD -z forward) 문제 해결
  - v4 = image polygon area 큰 face = FRONT (cam-frame 의존 X)
  - 71% perm 변환, 모든 invariant 100%
  - 자세한 근거: memory `project_keypoint_convention_v4_conversion.md`
- `challenge0123_ft_manual`, `challenge0123_ft_v2` 의 시작점
