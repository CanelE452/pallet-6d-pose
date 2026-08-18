# Training

```
dataset          29308 frames (loader count), 2443 batches, batch 12
roots            mixed_v8_train, v4_split_base, aug_squash_v2, aug_trunc_v2, aug_scale_v2, paper_4pallet_mask_v1
balance_groups   mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:60,paper_4pallet_mask_v1:40
sampler          WeightedRandomSampler, epoch_size = dataset size
init             ep57 (SHA c0055fe7...), scratch 아님
epochs           5 (고정), early stop 없음, checkpoint selection 없음
optimiser        AdamW  branch 3e-4 / stage4-6 5e-5 / last VGG 1e-5, wd 1e-4
scheduler        없음        AMP  미사용(기존 trainer 에 검증된 경로 없음)
seed             1
trainable        vgg_last 5,014,912 + stage4_6 12,567,579
                 + branch 334,081 = 17,916,572
features         F100 = vgg[17] 256ch,
                 F50 = vgg[26] 128ch (runtime 탐색)
```

## Throughput

```
epoch   wall(min)   samples/s   peak GPU
  1        14.9        32.8     2862 MB
  2        15.6        31.4     2862 MB
  3        15.6        31.3     2862 MB
  4        16.5        29.6     2862 MB
  5        15.2        32.0     2862 MB
total training 77.9 min   (batch 12, OOM fallback 불필요)
```

## Loss trajectory (raw, 미가중)

```
epoch   L_DOPE(4-6)   L_proposal   L_refined   mean gate
  1      0.001088     1.2996      0.4728     1.746e-04
  2      0.001092     1.1452      0.4028     2.513e-07
  3      0.001078     1.0813      0.3914     7.158e-08
  4      0.001024     1.0438      0.3667     2.614e-08
  5      0.001018     1.0213      0.3644     1.176e-08
```

calibration (20 batch, update 없음, train only):
lambda_proposal = 3.672e-05, lambda_refined = 3.073e-04,
lambda_gate = 0.01 (사전 상수).  5 epoch 동안 고정, 결과 보고 재조정 없음.
