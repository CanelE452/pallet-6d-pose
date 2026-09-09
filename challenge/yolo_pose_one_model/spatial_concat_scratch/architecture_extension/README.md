# Frozen-cache architecture extension

```text
TRAINING_STATUS = COMPLETE
NEW_RUNS = 15/15
YOLO_PARAMETER_UPDATES = 0
```

This namespace prepares the remaining cheap architecture comparisons on the
locked clean-start 60k cache. It trains five post-hoc branches:

- `S2_SPATIAL_FILM`
- `S3_SPATIAL_CROSS_ATTENTION`
- `D0_DIMS_ONLY` (`4 -> 128 -> 32 -> 2`)
- `K0_KP_ONLY` (`26 -> 128 -> 32 -> 2`)
- `C1_CENTER_CONCAT` (the exact selected center cell instead of the 7x7 pool)

The completed `S0_SPATIAL_NO_DIMS` and `S1_SPATIAL_CONCAT` median artifacts are
loaded as frozen reference rows, so the resulting matrix contains all seven
architectures without retraining S0/S1.

The existing S0/S1 contracts, runners, checkpoints, and results are read-only.
Every new artifact is written exclusively below this directory and a stopped
run resumes from complete seed checkpoints. The cache, YOLO checkpoint, fixed
XYZ dimension provenance, seed schedules, and reference S0/S1 artifacts are
hash-bound by `ARCHITECTURE_EXTENSION_CONTRACT.json`.

## Commands

Run the non-mutating checks first:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/run_architecture_extension.py validate-contract
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/run_architecture_extension.py unit-smoke --device cpu
```

Train/resume all 15 runs and materialize the deterministic DEV summary:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/run_architecture_extension.py train --device cuda:0
```

Optionally measure batch-1 branch-only CUDA latency after training (20 warmups,
200 measurements, p50/p95):

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/run_architecture_extension.py latency --device cuda:0
```

The summary includes median-DEV locked artifacts, correct and shuffled fixed-XYZ
controls for every dimension-conditioned branch, per-source metrics, 3-seed
mean and sample SD, exact sampler-schedule checks, S1/S2/S3 step-zero checks,
parameter counts, and 10,000-replicate paired cluster-bootstrap deltas versus
S1. P0 and TEX rows sharing a scenario `pair_group_id` are one bootstrap
cluster.

Completed outputs:

- `ARCHITECTURE_EXTENSION_SUMMARY.json`
- `ARCHITECTURE_EXTENSION_REPORT.md`
- `ARCHITECTURE_EXTENSION_DEV_PREDICTIONS.npz`
- `CUDA_BRANCH_LATENCY.json`
- `runs/<arm>/seed{0,1,2}.{pt,json}`

The immutable summary was created before the optional latency command, so its
`NOT_REQUESTED` latency field records summary-creation time. The later latency
file binds that exact summary by SHA-256. The post-run
[completion manifest](../FOLLOWUP_EXPERIMENT_COMPLETION.json) binds both files
without rewriting the training provenance.

## Evidence boundary

These are exploratory mixed synthetic DEV results. There is no independent
TEST evaluation, no real-data evaluation, and no authorization to populate a
paper `FINAL` table from these outputs. YOLO is frozen and receives zero
parameter updates.
