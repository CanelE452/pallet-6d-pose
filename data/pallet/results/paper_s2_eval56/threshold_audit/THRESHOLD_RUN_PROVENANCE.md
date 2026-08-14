# Provenance

```
HEAD at start              769041102221c012d6666f7a471e69e3b8197837
HEAD at write              769041102221c012d6666f7a471e69e3b8197837
ep57 checkpoint            weights/paper_s2_stageB/net_epoch_0057.pth
ep57 SHA256                c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
python                     3.10.20
torch                      2.1.1+cu118
numpy                      1.26.4
seed                       1  (bootstrap only)
training steps             0
optimizers constructed     0
checkpoints written        0
belief tensors modified    0
```

## Inputs

```
eval56   data/pallet/results/paper_s2_eval56/eval56_manifest.json      56 frames
         data/pallet/results/paper_s2_eval56/eval56_ep57_belief.npz    cached stages
wood     data/pallet/results/paper_s2_eval56/wood_manifest.json        45 frames
         data/pallet/results/paper_s2_eval56/wood_ep57_belief.npz      cached stages
```

Both caches were produced by earlier runs of the same forward path and are
reused unchanged; no model was run in this audit.  The sealed sessions
(`capturenight08`, `capturenight09`, `capturepallet07`, `capturepallet09`,
`testset_full8_manifest`, `handannot17`) were not read.

## Held fixed

```
belief / affinity tensors        unmodified
Gaussian sigma                   not on this path (D2 only)
NMS                              not on this path (D2 only)
7x7 local softargmax, T = 0.1    unmodified   (LOCAL_RADIUS 3, LOCAL_TEMPERATURE 0.1)
+0.4395 offset                   not on this path (D2 only)
affinity grouping                not on this path
centroid threshold               0.30 in every arm
PnP                              current_solve, auto_swap_dims, 9 correspondences
K, dimensions                    from each frame's JSON
```

## Varied

```
corner acceptance comparison at paper_s2_frozen_diagnostic.py:661,
re-applied per channel in scripts/stage0/paper_s2_eval56.py decode_thresholded
```

Nine arms, fixed before execution, none added afterwards:

```
T0 0.300/0.300   T1 0.275/0.275   T2 0.250/0.250   T3 0.225/0.225   T4 0.200/0.200
R1 0.275/0.300   R2 0.250/0.300   R3 0.225/0.300   C1 0.300/0.250
```
(near/far; centroid 0.300 throughout)

## Outputs

```
THRESHOLD_ARCHITECTURE.md  THRESHOLD_CORNER_RECOVERY.md  THRESHOLD_POSE_RESULT.md
THRESHOLD_COMMON_SUCCESS.md  THRESHOLD_RESCUE_FRAMES.md  THRESHOLD_GO_STOP_GATE.md
threshold_corner_rows.csv  threshold_frame_rows.csv  threshold_arm_metrics.csv
threshold_rescue_frames.csv  threshold_common_success.csv  threshold_gate.json
threshold_precision_recall.png  threshold_pnp_curve.png  threshold_common_success.png
threshold_rescue_examples.png  threshold_false_corner_examples.png
```
