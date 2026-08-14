# PAPER_S2 diagnostic inventory

Snapshot: `2026-07-28T16:02:42+09:00`

This inventory is intentionally scoped to the camera-facing PAPER_S2 lineage.
The sealed final-test was identified from the split-lock only; its images and
annotations were not opened or used for diagnosis.

## Repository and runtime

| item | value |
|---|---|
| repository | `/home/minjae/Documents/github/pallet-pose` |
| branch / HEAD | `main` / `0baa6dfc2ba850dd498f59b74e42663828d166c7` |
| worktree at inventory snapshot | `116 D`, `31 M`, `37 R`, `203 ??` |
| reproducibility warning | PAPER_S2 training code and July scripts are uncommitted in this checkout; HEAD alone does not reconstruct the run |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB |
| driver | `580.173.02` |
| Python environment | conda `pallet-pose`, Python `3.10.20` |
| PyTorch / CUDA | `2.1.1+cu118`, CUDA available |
| torchvision / OpenCV | `0.16.1+cu118` / `4.9.0` |
| NumPy / SciPy | `1.26.4` / `1.12.0` |
| pandas / matplotlib / albumentations | `2.3.3` / `3.10.9` / `1.4.2` |
| free disk at snapshot | about `1.2 TiB` |

The checkout is heavily dirty and contains user-owned work. No existing data,
GT, checkpoint, or result was overwritten.

## Data and split lock

| role | source | count | use in this audit |
|---|---|---:|---|
| synthetic train Arm A | `mixed_v8_train`, `v4_split_base`, `aug_{squash,trunc,scale}_v2` | 19,308 | lineage/audit only |
| synthetic train Arm B | `paper_4pallet_mask_v1` | 10,000 | lineage/audit only |
| synthetic validation | `training_data/val` | 1,500 | order-free aggregate only; legacy/object-order warning |
| fixed Q1 validation | `q1_split/val_list.json` | 500 unique frames | order-free recheck; no kp-id claim |
| strict real filter-val | split-lock Outdoor–Day 44 + Night 43 | 87 | primary frozen diagnosis |
| legacy manual set | `stage0_gt_candidates/manual_gt` from PL-pool `capturepallet11` | 36 | exploratory only |
| real unlabeled train pool | split-lock outside 2,227 + night 5,804 frames | 8,031 | membership/leak audit only |
| sealed final-test | outside 3,512/63 GT + night 1,706/42 GT | 5,218/105 GT | **not opened; not evaluated** |

Per-source synthetic counts are `9000 + 4000 + 2212 + 2971 + 1125 +
10000 = 29308`. Split-lock SHA-256 is
`2ed92037ef1816d2adc9d514934de98bcdf790baf2ab5118c4d59028a761c67f`.
The fixed Q1 list SHA-256 is
`5a88384f045faf22dda48465b440e69dba78bc94420f10a3db5217390befb56d`.

Important correction: the historical `filterval N=123` aggregate is not one
clean validation split. Its 36 manual frames belong to a PL-pool session.
Primary numbers in this audit therefore use N=87 and show N=36 separately.

The six Stage-B training sources are nearly fully camera-facing by LR/TB
screen-order checks, but held-out `training_data/val` is not: only 690/1,500
(46.0%) pass all LR pairs. It appears to be a pre-conversion legacy/object-order
set. Existing Hungarian/order-free aggregate metrics remain usable as a
convention-invariant recheck; per-channel front/rear and keypoint-5 conclusions
from this synthetic set are invalid and are not used here.

## Checkpoint lineage

| role | checkpoint | convention | training/selection | SHA-256 | audit status |
|---|---|---|---|---|---|
| canonical primary | `weights/paper_s2_stageB/net_epoch_0057.pth` | camera-facing 0123 | Stage A synthetic Arm A, then Stage B Arm A:B sampled 60:40 with mask auxiliary; selected on synthetic val | `c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896` | **primary frozen model** |
| Stage A | `weights/paper_s2_stageA/net_epoch_0042.pth` | camera-facing 0123 | scratch Arm A; synthetic-val selected | `86b58da86ef44adb32a3da5dd92d12db9619227a14cf8ea621f89eb99c2bb739` | lineage reference |
| inference export | `weights/paper_s2_stageB/net_epoch_0057_noseg.pth` | camera-facing 0123 | ep57 with only 24 `m_seg*` tensors removed | `f4b455b360fa6b693fb38ac27f79098a9bd27592cb761bfb50ce82c2d19f8a41` | not an independent model |
| later self-train | `data/pallet/results/ralph_selftrain/h3_s2_night/round_02.pth` | camera-facing 0123 | ep57-derived strict night PL, called best on 2026-07-15 | `d6ddd8b5d97bb9ed176761c9cf56354738795a20cc6a42098a40cdd5513e27ad` | excluded: filter-val session leakage; historical selection not paper-safe |
| latest dated derivative | `data/pallet/results/ralph_selftrain/h8_s2_combined_looflip/round_02.pth` | camera-facing 0123 | ep57-derived LOO+flip combined PL | `b6e79ff1a5766f26554172056e79ea652c5ee8b06359f8aade0b4553962b8f99` | excluded: filter-val session leakage and historical outside evaluator included sealed sessions |

`stageB_best.pth` resolves to `net_epoch_0057.pth`. The full ep57 state has 208
tensors and the no-seg export has 184; all 184 common tensors are bitwise
equal. Object-frame v8 checkpoints and results are excluded from the
camera-facing comparison.

## Canonical ep57 configuration

The exact namespace is stored in `weights/paper_s2_stageB/header.txt`
(SHA-256 `f06f7f12c21a4beec782ff31af579e8c9cc4ee0e05e2296aaa8a5dc01b590972`).

Key values:

- input `400x400` anisotropic squash from `640x480`; belief grid `50x50`;
- 9 beliefs, 16 affinities, 6 recurrent stages;
- batch 12, seed 42, sigma 2, Adam LR `5e-5` in Stage B;
- base loss: six belief MSE terms plus six affinity MSE terms;
- mask auxiliary: two segmentation BCE stages, weight `0.01`, valid-mask frames only;
- PAPER_S2 DiffPnP3D: final belief corners 0–7, local 7x7 soft-argmax,
  temperature `0.1`, four unrolled GN iterations, weight `0.005`, 1,000-step
  linear ramp;
- legacy `--geo_loss`, structural, reliability, visibility-coordinate, and
  symmetric losses were all off.

Historical launch and selection evidence is in
`scripts/stage0/diffpnp3d_full_run.sh`,
`full_run_logs/driver.log`, `full_run_logs/stageB_train.log`, and
`stageB_val_select.md`.

## Audit commands

```text
conda run -n pallet-pose python scripts/stage0/paper_s2_geometry_unit_audit.py
PYTHONPATH=. conda run -n pallet-pose pytest -q \
  challenge/tests/test_diffpnp_fit_coverage.py \
  challenge/tests/test_diffpnp_undercoverage.py \
  challenge/tests/test_projected_span_loss.py \
  challenge/tests/test_signed_footprint_loss.py
conda run -n pallet-pose pytest -q \
  challenge/tests/test_paper_s2_geometry_unit_audit.py
conda run -n pallet-pose python scripts/stage0/diffpnp3d_selftest.py
conda run -n pallet-pose python -u scripts/stage0/diffpnp3d_q1_eval.py \
  --weights weights/paper_s2_stageB/net_epoch_0057.pth \
  --tag diagnostic_ep57_20260728
```

The frozen real-data diagnostic command and its complete provenance are stored
with that run's manifest.
