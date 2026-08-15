# challenge0123_ft_v2

## 학습 설정

```
Weight:      weights/challenge0123_ft_v2/net_epoch_0080.pth (확인 시점; 학습 목표 120)
초기 weight: weights/challenge_track/challenge0123/final_net_epoch_0060.pth
Epochs:      120 (60 → 120, 60 ep ft)
Batch size:  8
LR:          1e-4
Sigma:       4.0
Image size:  448
Workers:     4
Seed:        8055
```

## 학습 데이터 (14 manual GT)

낮 8 capture:
```
challenge/data/01_real/manual_gt/capturepallet02_manual_gt
challenge/data/01_real/manual_gt/capturepallet03_manual_gt
challenge/data/01_real/manual_gt/capturepallet04_manual_gt
challenge/data/01_real/manual_gt/capturepallet05_manual_gt
challenge/data/01_real/manual_gt/capturepallet07_manual_gt
challenge/data/01_real/manual_gt/capturepallet08_manual_gt
challenge/data/01_real/manual_gt/capturepallet09_manual_gt
challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt
```

야간 6 capture:
```
challenge/data/01_real/manual_gt/capturenight04_manual_gt
challenge/data/01_real/manual_gt/capturenight05_manual_gt
challenge/data/01_real/manual_gt/capturenight06_manual_gt
challenge/data/01_real/manual_gt/capturenight07_manual_gt
challenge/data/01_real/manual_gt/capturenight08_manual_gt
challenge/data/01_real/manual_gt/capturenight09_manual_gt
```

## Loss 설정

```
symmetric_loss : False
struct_loss    : False
geo_loss       : False
rel_loss       : False
```

## 메모

- `_ft_manual` 대비 **야간 capture 6 개 + 낮 capture 추가 (02, 08)** 까지 포함한 더 큰 manual GT pool 로 장기 ft
- 야간 일반화 목적
- 학습 진행 중 — 확인 시점 기준 `net_epoch_0080.pth` 까지 저장됨
- 최종 가중치는 학습 완료 후 `final_net_epoch_0120.pth` 가 저장될 예정
