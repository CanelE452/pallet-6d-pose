# RUN_PROVENANCE — PAPER_S2 target-semantics audit

생성: 2026-07-29T12:19:39+09:00

## Source identity
```
git branch : main
git HEAD   : 5f45b5c5d46196168c7a2af56933d91564e818f4
git remote : https://github.com/CanelE452/pallet-6d-pose.git
git status : 1 modified/untracked entries
?? scripts/stage0/paper_s2_target_semantics_audit.py
```

## Environment
```
python 3.10.20
torch 2.1.1+cu118 cuda 11.8
opencv 4.9.0
numpy 1.26.4
pandas 2.3.3
conda env : pallet-pose
GPU       : NVIDIA GeForce RTX 3080
```

## Checkpoint (기준, 읽기 전용)
```
path     : weights/paper_s2_stageB/net_epoch_0057.pth
sha256   : c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
expected : c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
MATCH    : YES
```

## Analysis script
```
b239d03879d85823c152e8d14d74b6cbcfd49a8eddd2a95bfacb4f4946bab6f1  scripts/stage0/paper_s2_target_semantics_audit.py
```

## Audited source files (ep57 학습 경로)
```
794acca9a402beb6e8f426f20dc493777c636fe14de396764f2f456d3040d8c0  Deep_Object_Pose/common/utils_belief.py
0b40c2e0b2bc7cfb20ef652efd2e9942043eb3e68c5e1c6187ec3c922033e0c6  Deep_Object_Pose/common/utils_dataset.py
0f3807bd08c52796930fc0c0d56f4322b7710af06eae094ce65db43da8a7f7c5  Deep_Object_Pose/common/heatmap_refinement.py
0ddbeeb83932670cb97f32e4ad963b0fb056c41b2c34902a6d3d883f15195179  Deep_Object_Pose/train/train.py
0f4ada838ea1dc4dd9cd81febee696d95e2c7ad76381e595cc17e4dcdc93abd1  Deep_Object_Pose/train/diffpnp3d_loss.py
e507a636f1d6e21d96ac32872ae580f8415f58f3d6e4109c9ef20149f63b28d0  challenge/scripts/gen_truncation_crops.py
8c171b3d0b31e86dd8304cf10fb0ad723bf09100f1a9993fe6265fe916ec75b7  challenge/scripts/pad_truncation_crops.py
```

## ep57 training args (weights/paper_s2_stageB/header.txt 발췌)
```
sigma=2.0  imagesize=400  output_size=50 (train.py:987)
truncation_aug_prob=0.0   <- 런타임 truncation aug 미사용 (pre-generated aug_trunc_v2만)
clip_belief_border : 미지정 -> False (CreateBeliefMap legacy all-zero 경로)
mask_aux=True mask_weight=0.01
diffpnp=True lambda=0.005 warmup=0 ramp=1000 temp=0.1  -> aspect_resize=True
net_path=weights/paper_s2/paper_s2_stageA/net_epoch_0042.pth  epochs=57  seed=42  batch=12  lr=5e-05
balance_groups=mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:60,paper_4pallet_mask_v1:40
```

## Data manifest (읽기 전용)
```
dataset                  json_frames
mixed_v8_train               9000
v4_split_base                4000
aug_squash_v2                2212
aug_trunc_v2                 2971
aug_scale_v2                 1125
paper_4pallet_mask_v1       10000
diffpnp index dir: data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index (30808 frames)
```

## 금지사항 준수
- final-test: 이번 작업에서 open count = 0 (접근 없음)
- 기존 checkpoint/데이터 JSON/PNG: 읽기 전용, 수정·재저장 없음
- 신규 full training: 미실행
