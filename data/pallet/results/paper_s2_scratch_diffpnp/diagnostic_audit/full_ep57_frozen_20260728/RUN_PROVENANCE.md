# Full ep57 frozen diagnostic provenance

## Scope

- Frozen inference only; no training or checkpoint selection.
- Synthetic Q1: 500 frames, order-free aggregate only.
- Strict filter-val primary: Outdoor–Day 44 + Night 43 = 87 frames.
- Exploratory PL-pool manual set: 36 frames, reported separately.
- Sealed final-test and `handannot17`: zero input opens.

## Exact command

Working directory:
`/home/minjae/Documents/github/pallet-pose`

```bash
set -o pipefail
/home/minjae/anaconda3/envs/pallet-pose/bin/python -u \
  scripts/stage0/paper_s2_frozen_diagnostic.py \
  --device cuda \
  --run-name full_ep57_frozen_20260728 \
  2>&1 | tee \
  data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/full_ep57_frozen_20260728.stdout.log
```

The captured stream was copied without modification to `stdout_stderr.log`.

## Frozen identities

| item | value |
|---|---|
| Git branch / HEAD | `main` / `0baa6dfc2ba850dd498f59b74e42663828d166c7` |
| worktree warning | relevant PAPER_S2 sources are uncommitted; HEAD alone is insufficient |
| diagnostic script SHA-256 | `58ba873be26eb9b66af817ec9bce277d864ffe11472ca3230f20bd0e965b704b` |
| ep57 checkpoint SHA-256 | `c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896` |
| Q1 list SHA-256 | `5a88384f045faf22dda48465b440e69dba78bc94420f10a3db5217390befb56d` |
| strict N87 identity SHA-256 | `2795991dbf7f2c3dcc45132ea18a048a1373893aa0023b9e4bbc81266c1123dd` |
| manual N36 identity SHA-256 | `1d8c8998623258c8ca90a3dd5c47eb4c49c6136d17b6dadfa49df9357dcb3f4b` |
| legacy N123 identity SHA-256 | `ee5f766347bd1bf33ceec899c7d167a33bc5e4f0cc4680e860cb78a9efc68766` |

## Runtime

| item | value |
|---|---|
| start UTC | `2026-07-28T07:33:36.891510+00:00` |
| finish UTC | `2026-07-28T07:43:01.382093+00:00` |
| elapsed | `564.490579276011 s` |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB |
| driver | `580.173.02` |
| Python | `3.10.20` |
| PyTorch / CUDA build | `2.1.1+cu118` |
| OpenCV | `4.9.0` |
| seeds | Python-side/Torch/NumPy/OpenCV `0`; bootstrap `20260728` |
| bootstrap | paired 95% CI, 10,000 replicates; session cluster with documented temporal fallback |

The full conda package lock is `conda-explicit.txt`; GPU metadata is
`gpu-environment.csv`. The complete argument/config/output schema is in
`manifest.json`.

## Result validation

| check | result |
|---|---:|
| run status | `complete` |
| processed frames | `623 / 623` |
| frame errors | `0` |
| strict N87 GT-to-PnP | `87 / 87` |
| prohibited/final/handannot input attempts | `0 / 0 / 0` |
| core and related pytest before run | `41 passed` |

## Artifact hashes

| artifact | SHA-256 |
|---|---|
| `manifest.json` | `bfc40f160d0579c450d7251d5ded83e2cd9e4424d04c59cd536115bab1e729f1` |
| `summary.json` | `3508535b08695a3be11daeba2d4a186c8d2151ebde4d30ada3736ab9695acf52` |
| `stdout_stderr.log` | `fc9b5be40723afef6152b432d7194ddfb7ab677622159af78ad1f17ffda939ca` |
| `conda-explicit.txt` | `50860c4aecb4c727be7aa6cf491b2272d9563bd6d0fa0c1ed3b79fe0c4e2ad5d` |
| `gpu-environment.csv` | `3cd93f617448feed4306bde244a29b24a62c5b58953e7ef3201fff2d107864f6` |

## Frozen post-analysis

The full run was post-processed once with:

```bash
/home/minjae/anaconda3/envs/pallet-pose/bin/python -u \
  scripts/stage0/paper_s2_frozen_post_analysis.py \
  data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/full_ep57_frozen_20260728
```

The post-analysis refuses incomplete/non-full runs and existing output files.
It verified the source/checkpoint/data hashes, zero sealed-input counters, and
passed the existing 10,000-replicate bootstrap object through without
recomputing it.

| artifact | SHA-256 |
|---|---|
| post-analysis script | `0118a1e210bceb85a09f2d4b18c47ae20286d6a637ced47cf46ce23fda39edbf` |
| `analysis_summary.json` | `0cdbdd4d372c5a7fa8c8294e4a630dded95ed75097ab479e749527c40f49e2a5` |
| `frozen_tables.md` | `dfb23416cd663a3644382518dde0bc4e5ea6bb90f3ba18281c99f68f9dded619` |
| `frozen_tables.csv` | `31d35148e840be1db18c757023735cb050f57a8e3d02a37df482c640da09081e` |
| post-analysis stdout | `1eb76815b9b8afc324bc358e3cb719b21732b7362ac35d9283644713e4680e89` |
