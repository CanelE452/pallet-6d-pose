# Provenance

```
HEAD at start          88d25c55be0a9ef9275781177b7eb248ba96f648
HEAD at write          88d25c55be0a9ef9275781177b7eb248ba96f648
branch                 main
ep57                   weights/paper_s2_stageB/net_epoch_0057.pth
ep57 SHA256            c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
python                 3.10.20
torch                  2.1.1+cu118  cuda 11.8
opencv                 4.9.0
numpy                  1.26.4
gpu                    NVIDIA GeForce RTX 3080
training steps         0
optimizers constructed 0
checkpoints written    0
model forwards         one per set x arm (12), reused by all three decoders
tensor dtype           float32 throughout, asserted before hashing
final-test sessions    not read
```

## Arm checkpoints, from run_state.json not from guesswork

```
B0  weights/paper_s2_stageB/net_epoch_0057.pth                    c0055fe7  (H6, A6)
E2  same forward as B0, near H6 / far H5 / centroid H6            c0055fe7
S1  weights/paper_s2_stagewise_bias_screen/epoch_005.pth          99584084  epoch 5/5
C1  weights/paper_s2_corner_replacement_screen/epoch_005.pth      aad97f6b  epoch 5/5
N2  weights/paper_s2_pfdr/N2/epoch_003.pth                        4b644fd8  epoch 3/3
N3  weights/paper_s2_pfdr/N3/epoch_003.pth                        9db513f3  epoch 3/3
```

N1 is excluded, and that exclusion was recorded as a limitation of the original
PFDR verdict (its anchor lambda clamped at 10, so the residual never moved),
not chosen after seeing anything here.

## Held fixed

```
decoder thresholds     0.30 / 0.30 / 0.30 / 0.50 as read from task.yaml
Gaussian sigma         3 (deployment), 2 (nowhere on P0/P1)
NMS                    detector.py 4-neighbour, untouched
local window           11x11 (P2/P1), 7x7 (P0), untouched
affinity thresholds    thresh_angle 0.50, untouched
PnP                    SOLVEPNP_EPNP via CuboidPNPSolver (P2),
                       current_solve/annotate_pnp (P0, P1), untouched
K                      per frame from the JSON; squash-scaled for P2 only
dimensions             per frame from the JSON, centimetres for P2 as production does
input resize           squash 400x400 for all three paths
```

## Outputs

```
DECODER_PATH_TRACE.md  DECODER_PATH_PARITY.md  DECODER_BASELINE.md
DECODER_COORDINATE_COMPARISON.md  DECODER_ASSOCIATION_COMPARISON.md
DECODER_MEMBERSHIP.md  DECODER_PAIRED_POSE.md  DECODER_VERDICT_MATRIX.md
DECODER_FINAL_DECISION.md  DECODER_RUN_PROVENANCE.md
decoder_arm_metrics.csv  decoder_frame_membership.csv  decoder_corner_p0_p1.csv
decoder_association_p1_p2.csv  decoder_paired_pose.csv  decoder_verdict_matrix.csv
decoder_gate_registry.json  decoder_direct_cache_parity.csv
decoder_frames.parquet (local only)  figures/ (9 png, local only)
```
