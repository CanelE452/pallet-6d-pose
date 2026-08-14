# Provenance

```
HEAD at start           9c329fcba5e5abddabb837dba7e8710de16f0e54
HEAD at write           9c329fcba5e5abddabb837dba7e8710de16f0e54
branch                  main
ep57 SHA256             c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
python 3.10.20   torch 2.1.1+cu118 cuda 11.8   opencv 4.9.0   numpy 1.26.4
gpu                     NVIDIA GeForce RTX 3080
training steps          0
optimizers constructed  0
checkpoints written     0
tensor dtype            float32
final-test sessions     not read (N87 filtered on is_final_test == False and on the sealed token list)
```

## Sets

```
N87   mechanism_val_manifest.json, domain in (outside, night), is_final_test False
      87 frames   membership sha 8d086cd7f8a20cf6fd76af4b26fd5b2e
eval56  56 frames  sha d4eb5ebe4f30d87bc23fe356482f9aa0   NOT SPENT
wood    45 frames  sha ebcc4164779593cf980459db98c03969   NOT SPENT
N87 ∩ eval56 = 12 frames (all outside)      N87 ∩ wood = 0 frames
```

## Control checkpoints

```
M1 weights/challenge0123/net_epoch_0060.pth
M2 weights/challengenight/net_epoch_0120.pth
```
Used only to measure blob width and to show the wrapper decodes normally on a
wide-target model.  Neither is an evaluation arm; v8-era weights remain barred
from results.

## Varied

```
config.sigma only, through decoder_paths.config_with_sigma, which copies every
other field of the DeploymentConfig unchanged.
grid: 0.0 0.5 1.0 1.5 2.0 2.5 3.0     fixed before the sweep, none added after
```

## Held fixed

```
thresh_map 0.30   thresh_points 0.30   threshold 0.30   thresh_angle 0.50
NMS, 11x11 window, +0.4395, affinity grouping, EPNP solver, live gates,
K, dimensions, checkpoint
```
A guard in `cal_run` aborts if any of the four thresholds differs from the
recorded values before the sweep starts.
