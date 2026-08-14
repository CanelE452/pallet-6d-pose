# Training

```
dataset        29308 frames, 2443 batches, batch 12
roots          mixed_v8_train, v4_split_base, aug_squash_v2, aug_trunc_v2, aug_scale_v2, paper_4pallet_mask_v1
sampler        mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:60,paper_4pallet_mask_v1:40
init           ep57 (SHA c0055fe7...), scratch 아님
trainable      42 tensors / 12,567,579 params (belief stage 4~6 만)
frozen audit   {'vgg_trainable': 0, 'belief123_trainable': 0, 'affinity_trainable': 0}  (VGG / belief 1~3 / affinity 전부 0)
optimiser      Adam lr 5e-05, scheduler 없음, AMP 미사용
epochs         5 고정, early stop 없음, checkpoint selection 없음
seed           1
```

## Throughput

```
epoch   wall(min)   peak GPU   total loss   legacy   mass    rank
  1        11.5     2189 MB   0.00552    0.00534  0.6635  0.1036
  2        11.5     2189 MB   0.00557    0.00539  0.6415  0.0971
  3        11.5     2189 MB   0.00552    0.00535  0.6078  0.0945
  4        11.7     2189 MB   0.00542    0.00526  0.5716  0.0881
  5        12.6     2189 MB   0.00542    0.00526  0.5611  0.0900
total 58.9 min   (batch 12, OOM fallback 불필요)
```

[확인] synthetic mass loss 1.27(초기) → 0.5611(epoch5) 로 하강 = GT mass 상승.
[확인] legacy loss 는 0.00534 → 0.00526 로 거의 평평 —
신규 loss 가 legacy 를 망가뜨리지 않았다.
