# Frozen CLEAN19 pose mismatch diagnosis

No new training, weights loading, optimizer, selector tuning or GT edits.
The unchanged prediction-only selector runs before reference metrics.
`hyp_pose` uses the same cuboid / SQPnP / RefineLM / physical-axis conversion as production `pose.infer`.
Official per-frame 2D and pose values, group metrics and common-matched results must reproduce before posthoc analysis.

Run from repository root in the pallet-yolo26 environment:

```sh
python -m scripts.research.pallet_clean19_pose_mismatch_v1.run
python -m scripts.research.pallet_clean19_pose_mismatch_v1.render
python -m scripts.research.pallet_clean19_pose_mismatch_v1.test_contracts
```

Outputs use exclusive-create; never overwrite a prior experiment.
The one narrowly scoped resume flag in run.py preserves the first STOP receipt and fixes only null-list comparison, not numeric results.

## Limits

- Axis correctness means W/D extents parity, not correct 3D orientation.
- With two successful W/D hypotheses one has reference-matching extents by construction; alternate-good parity does not establish a good pose.
- Native ID edge vectors and one-corner substitutions do not use symmetry-min GT relabeling.
- Oracle W/D and reference replacement are NONDEPLOYABLE, posthoc only.
- True LOO cannot pass the existing selector's all-nine-finite contract. We attempt removal and record rejection, without filling the removed point or replacing the algorithm. LOO sensitivity counts are null, not zero.
- Geometry-resolved real reference is not independently measured 6D GT.
- Public diagnostics/annotated derived montages only; raw prediction files, original RGB and private source mapping remain outside commit paths.
