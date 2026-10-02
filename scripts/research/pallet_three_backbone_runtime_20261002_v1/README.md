# Three-backbone unified runtime benchmark

This isolated namespace measures the already completed YOLO, DOPE and
dimension-aware ResNet-18 estimators on the exact same historical 26 DEV image
bytes.  It never trains or edits a sealed experiment.

The primary rows are YOLO R0/P1, DOPE baseline/P1 and ResNet FULL/P0 seed 1.
ResNet D0, P5_CONSTANT and P5 seed 1 are retained as auxiliary ablations.  Only
those four seed-1 ResNet heads are loaded; the twelve-head evaluation bank is
not part of any timed call.

CPU/static checks (no real images, checkpoints or GPU calls):

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m pytest -q \
  scripts/research/pallet_three_backbone_runtime_20261002_v1/test_runtime.py
```

Read-only completion check (creates no output):

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m \
  scripts.research.pallet_three_backbone_runtime_20261002_v1.benchmark preflight
```

Production measurement, after all ResNet accuracy artifacts are complete and
the GPU is idle:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m \
  scripts.research.pallet_three_backbone_runtime_20261002_v1.benchmark measure
```

The runner also constrains already-loaded native thread pools to one and checks
their observed values after model initialization.  It restores each frozen
source experiment's CUDA numerical contract outside the timer: YOLO uses cuDNN
TF32 while DOPE and ResNet-18 disable it.  This distinction is recorded in the
protocol and Korean report because exact saved-output parity takes precedence.

The command writes a frozen protocol and Korean report under
`_docs/experiments/pallet_three_backbone_runtime_20261002_v1`, plus append-only
raw rows under `data/pallet/results/pallet_three_backbone_runtime_20261002_v1`.
If any dependency, result receipt, selected source rule, image byte, or saved
DEV prediction is missing or stale, it exits before creating either output
directory.

Before measurement, a read-only replay of all 26 frames and nine arms observed
bit-exact YOLO/DOPE outputs and ResNet maxima of `0.0001893` px and `0.0002323`
across stored pose metrics.  The production parity gate is fixed at `0.0005`
for both classes of floating-point values; detection status and selected pose
axis must still match exactly.
