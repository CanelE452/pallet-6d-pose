# ResNet18 dimension-refiner report

This namespace is deliberately outside the sealed training/evaluation namespace.
It reads only completed, hash-bound artifacts from
`pallet_resnet18_dim_refiner_20261002_v1`, verifies the frozen protocol and
population contracts, and then creates a Korean report, publication tables,
claim audit, and PNG/PDF figures.

The reviewed output is written to
`_docs/experiments/pallet_resnet18_dim_refiner_report_20261002_v3`.  V3 keeps
the validated V2 statistics and spacing, and corrects the pose label to
C2-aware corresponding-point ADD rather than unrestricted nearest-neighbor
ADD-S.

Run from the repository root after all five production phases have completed:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m \
  scripts.research.pallet_resnet18_dim_refiner_report_20261002_v1.report
```

The command performs no model forward, optimizer update, or GPU call. If any
required receipt is absent, incomplete, or inconsistent, it exits before
creating the report output directory.

CPU checks:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m pytest -q \
  scripts/research/pallet_resnet18_dim_refiner_report_20261002_v1/test_report.py
```
