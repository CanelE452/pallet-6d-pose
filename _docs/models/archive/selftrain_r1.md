# selftrain_r1

## 학습 설정

```
Weight:     weights/selftrain/selftrain_r1/final_net_epoch_0070.pth
초기 weight: mixed_v1 ep60
Epochs:     61 → 70 (10 epoch 추가)
Batch size: 4
LR:         5e-5
Sigma:      4.0
Image size: 448
Seed:       9416
```

## 학습 데이터

```
데이터:     data/pallet/training_data/selftrain_r1/
이미지 수:  8,751
구성:       mixed_v1_train 8K (syn) + pseudo-label 751장 (real, geo filter passed)
```

### Pseudo-label 생성 과정

```
1. mixed_v1로 real_data (1924장) 추론
2. 3단계 geometric filter (A: Flip, B: Diagonal, C: LOO PnP)
3. 751장 통과 → PnP reproject → NDDS JSON 생성
4. 합성 8K + pseudo 751 병합
```

## Real Data 추론 (필터 재검증)

```
검증 항목         mixed_v1 (before)   selftrain_r1 (after)
──────────────────────────────────────────────────────────────
PnP 성공률        80.6%               85.0%
Filter passed     751장               791장
```

## 비고

- Self-training Round 1 — pseudo-label 기반 finetuning
- PnP 성공률 80.6% → 85.0%, filter passed 751 → 791장으로 개선 확인
- mixed_v1 대비 real data에서 소폭 개선
- Round 2 (791장으로 재학습) 가능하지만 아직 미실행
