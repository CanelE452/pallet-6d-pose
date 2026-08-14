# challenge_camfacing_ft

## 학습 설정

```
Weight:      weights/challenge_camfacing_ft/final_net_epoch_0070.pth
초기 weight: weights/challenge/final_net_epoch_0060.pth
Epochs:      70 (60 → 70, 10 ep refine)
Batch size:  8
LR:          1e-5            (challenge 의 1/10)
Sigma:       4.0
Image size:  448
Workers:     4
Seed:        5435
```

## 학습 데이터

```
data/pallet/training_data/mixed_v8_train
challenge/data/02_synthetic/training/v1
challenge/data/02_synthetic/training/v2
```

(challenge 와 동일)

## Loss 설정

```
symmetric_loss : False
struct_loss    : False
geo_loss       : False
rel_loss       : False
```

## 메모

- `challenge` 결과를 **같은 데이터로 LR 만 낮춰** 10 ep 미세 조정
- 추가 데이터 없이 수렴 안정화/일반화 효과를 노린 refine
