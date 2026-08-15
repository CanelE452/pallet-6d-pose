# challenge0123_ft_manual

## 학습 설정

```
Weight:      weights/challenge_track/challenge0123_ft_manual/final_net_epoch_0080.pth
초기 weight: weights/challenge_track/challenge0123/final_net_epoch_0060.pth
Epochs:      80 (60 → 80, 20 ep ft)
Batch size:  8
LR:          1e-4
Sigma:       4.0
Image size:  448
Workers:     4
Seed:        4139
```

## 학습 데이터 (6 manual GT)

```
challenge/data/01_real/manual_gt/capturepallet03_manual_gt
challenge/data/01_real/manual_gt/capturepallet04_manual_gt
challenge/data/01_real/manual_gt/capturepallet05_manual_gt
challenge/data/01_real/manual_gt/capturepallet07_manual_gt
challenge/data/01_real/manual_gt/capturepallet09_manual_gt
challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt
```

## Loss 설정

```
symmetric_loss : False
struct_loss    : False
geo_loss       : False
rel_loss       : False
```

## 메모

- `challenge0123` 을 실측 manual GT 6 개로 1차 도메인 적응
- 낮 (pallet) capture 만 포함, 야간 없음
- 야간 capture 까지 확장한 후속 모델 = `challenge0123_ft_v2`
