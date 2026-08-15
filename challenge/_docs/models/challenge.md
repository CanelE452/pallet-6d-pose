# challenge

## 학습 설정

```
Weight:      weights/challenge_track/challenge/final_net_epoch_0060.pth
초기 weight: scratch (VGG-19 ImageNet pretrained)
Epochs:      60
Batch size:  8
LR:          1e-4
Sigma:       4.0
Image size:  448
Workers:     4
Seed:        5192
```

## 학습 데이터

```
data/pallet/training_data/mixed_v8_train     (Isaac Sim 합성)
challenge/data/02_synthetic/training/v1                   (camera-facing v1 convention)
challenge/data/02_synthetic/training/v2                   (camera-facing v2 convention)
```

## Loss 설정

```
symmetric_loss : False
struct_loss    : False    (struct_warmup=10, struct_flip=0.02, struct_edge=0.05, struct_coord=0.1 등은 default 값이며 활성 X)
geo_loss       : False
rel_loss       : False
```

## 메모

- challenge 시리즈의 첫 정식 학습
- 이후 `challenge_camfacing_ft` 의 시작점
- camera-facing convention 도입 (v1 + v2 데이터 활용)
