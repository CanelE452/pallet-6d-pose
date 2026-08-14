# challenge_camfacing_scratch

## 학습 설정

```
Weight:      weights/challenge_camfacing_scratch/final_net_epoch_0060.pth
초기 weight: scratch (VGG-19 ImageNet pretrained)
Epochs:      60
Batch size:  8
LR:          1e-4
Sigma:       4.0
Image size:  448
Workers:     4
Seed:        7769
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

- `challenge` 와 동일 데이터/하이퍼, **seed 만 다른** 재학습 (5192 → 7769)
- 재현성/seed sensitivity 확인 목적으로 추정
- `outf` 는 `challenge_camfacing_scratch` 그대로
