# Frozen FULL ResNet dimension refiner

This isolated namespace is a source-only correction experiment built around the
completed, immutable epoch-10 `FULL` DSNT checkpoint. It does not modify the
bound DSNT training code, evaluator, checkpoint, or paper files.

Arms are `D0`, visual `P0`, dimension-conditioned `P5`, and the same P5
architecture with its five inputs forced to zero (`P5_CONSTANT`). All four heads
share each seed's source rows and frozen `layer2/layer3` features. The intended
budget is three seeds, 6,000 updates, batch 16, with synthetic-only calibration
and selection.

The namespace now contains the strict FULL adapter, compact source prediction
cache, online frozen-prefix batches, paired training, synthetic calibration and
selection, GT-free DEV inference, and canonical 9-point/T-R evaluation. The
adapter uses the DSNT spatial-softmax expectation decoder; it never falls back
to the older argmax decoder. Protocol sealing and all GPU phases remain explicit
operator actions.

CPU verification from the repository root:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m pytest -q \
  scripts/research/pallet_resnet18_dim_refiner_20261002_v1/test_contracts.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m \
  scripts.research.pallet_resnet18_dim_refiner_20261002_v1.cpu_smoke
```

Neither command starts CUDA, writes an experiment receipt, or performs a
production optimizer update. The smoke audits source/dimension metadata and
uses invented head tensors only.

After `PROTOCOL.json` is sealed, the production phase order is:

```bash
# Explicit GPU/operator phases; do not run concurrently with another GPU job.
python -m scripts.research.pallet_resnet18_dim_refiner_20261002_v1.source_cache
python -m scripts.research.pallet_resnet18_dim_refiner_20261002_v1.train
python -m scripts.research.pallet_resnet18_dim_refiner_20261002_v1.selection
python -m scripts.research.pallet_resnet18_dim_refiner_20261002_v1.inference
python -m scripts.research.pallet_resnet18_dim_refiner_20261002_v1.evaluation
```

`FULL` alone cannot establish that dimensions caused an improvement. The causal
comparison is `P5 - P5_CONSTANT`; `P5 - P0` separately includes the added module
capacity/path. All claims must retain these controls and the reused-DEV label.
