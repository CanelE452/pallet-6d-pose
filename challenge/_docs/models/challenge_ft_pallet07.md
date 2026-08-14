# challenge_ft_pallet07

## 학습 설정

```
Weight:      weights/challenge_ft_pallet07/final_net_epoch_0091.pth
초기 weight: challenge/weights/baseline_v8_A.pth
Epochs:      91
Batch size:  4
LR:          5e-5
Sigma:       4.0
Image size:  448
Workers:     0
Seed:        5393
```

## 학습 데이터

```
challenge/data/03_derived/_train_capturepallet07/train     (pallet07 단일 capture)
```

## Loss 설정

```
symmetric_loss : True            (flip 모호성 해결)
struct_loss    : True
  struct_lambda  : 1.0
  struct_coord   : 0.003
  struct_edge    : 0.05
  struct_flip    : 0.02
  struct_warmup  : 10
geo_loss       : False
rel_loss       : False
```

## 메모

- 합성 baseline (`baseline_v8_A`) 을 **pallet07 sequence 1 개로 도메인 적응**
- `symmetric_loss=True` 로 앞/뒤 face 모호성 해소 시도
- 단일 capture 학습이라 over-fit 위험 있음
- 평가 곡선: `_docs/loss_curve_pallet07_ft.png`
