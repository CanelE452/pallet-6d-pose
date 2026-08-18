---
license: mit
tags:
- 6d-pose-estimation
- dope
- pallet
- forklift
- vgg19
library_name: pytorch
pipeline_tag: keypoint-detection
---

# pallet-dope-challenge0123-ft-manual

DOPE (Deep Object Pose) 6D pose estimation model for **forklift pallet detection**.
Fine-tuned on 6 real manual-GT capture sequences starting from `challenge0123` (Isaac Sim + camera-facing v4 convention baseline).

Part of the [Pallet 6D Pose Estimation](https://github.com/CanelE452/pallet-6d-pose) project — geometry-aware self-training for automatic forklift alignment.

## Model overview

```
arch         : DOPE (VGG-19 backbone + belief maps + affinity fields)
keypoints    : 9  (8 cuboid corners + centroid, camera-facing v4 convention)
input        : 448x448 RGB
output       : belief maps (9ch) + affinity fields (16ch)
pallet dim   : 1.10 x 1.30 x 0.11 m  (KS T-11형)
```

## Training

```
Weight       : final_net_epoch_0080.pth
Init weight  : challenge0123/final_net_epoch_0060.pth (scratch -> 60 ep on mixed_v8 + chal_v1 + chal_v2)
Epochs       : 80  (60 -> 80, 20 ep fine-tune)
Batch size   : 8
LR           : 1e-4
Sigma        : 4.0 (belief Gaussian)
Image size   : 448
Workers      : 4
Seed         : 4139
Loss         : belief MSE + affinity (no symmetric / no geo / no struct / no rel)
```

### Fine-tune data (6 manual-GT capture sequences)

```
capturepallet03_manual_gt
capturepallet04_manual_gt
capturepallet05_manual_gt
capturepallet07_manual_gt
capturepallet09_manual_gt
capturepalletcad_manual_gt
```

All captured with Intel RealSense D435i (640x480 @ fx=614.18, fy=614.31, cx=329.28, cy=234.53).

## Files

```
final_net_epoch_0080.pth   trained model weight
header.txt                 raw training Namespace + seed
README.md                  this file
```

## Inference (depth_cam pipeline)

The matching pose-solving contract verified by `twin_pnp_check.py` (50/50 frame, reproj 2.89px, |dt|=0.085m):

```python
PALLET_WIDTH_M  = 1.0    # mixed_v8_train label dim
PALLET_DEPTH_M  = 1.2
PALLET_HEIGHT_M = 0.15
PALLET_PNP_CONTRACT_Z180 = True   # Cuboid3d.vertices @ diag([-1,-1,+1])
```

`task.yaml`'s nominal `(1.1, 1.3, 0.11)` is the spec value and does **not** match the actual label dim used in training.

## Lineage

```
ImageNet pretrained VGG-19
        |
        v
challenge0123                     scratch, 60 ep, mixed_v8_train + chal_v1 + chal_v2
        |
        v
challenge0123_ft_manual  <--- this model, 20 ep on 6 manual-GT sequences
```

## Related

- GitHub: https://github.com/CanelE452/pallet-6d-pose
- Camera-facing v4 convention rationale: see project memory `project_keypoint_convention_v4_conversion.md`
- annotate.py PnP fix v4 (gravity invariant): see `project_annotate_pnp_fix_v4_gravity.md`

## License

MIT
