# Crop completion v2: experiment implementation

One authorized crop1.50 FULL fit from PRIOR1, seed1, 300 updates. A/B/C/D definitions, metrics and limitations are in the [illustrated report](../../../_docs/experiments/pallet_posefix_crop_completion_v2/REPORT_KO.md).

This is the implementation and audit trail of an executed local experiment, **not a self-contained public reproduction package**. It imports historical local modules (`pallet_posefix_crop_support_v1`, `pallet_posefix_heatmap_diversity_v1`, `pallet_posefix_full_preserve_v1`, `pallet_posefix_limited_adaptation_pilot_v1` and their dependencies). Their frozen preparation artifacts, models, original RGB, source data, symmetry mappings and private input locks are required. They are not embedded in this publication. Paths and hash contracts intentionally identify that existing local environment; do not synthesize missing inputs or skip verification.

Environment used: `pallet-yolo26` Python environment with PyTorch/CUDA, NumPy, OpenCV, matplotlib and pytest; RTX 3080. Launch from the repository root. `MPLCONFIGDIR=/tmp/pallet-crop2-mpl` avoids a read-only matplotlib cache.

## Executed order (record, not permission to rerun)

Each module below is invoked with `python -m scripts.research.pallet_posefix_crop_completion_v2.<module>`. Outputs use exclusive creation. Existing completed outputs must not be overwritten, removed, or retrained to make these commands pass.

1. `common`: verify old artifacts, lock protocol, populations and diagnostic subsets.
2. `data`: reconstruct original RGB crops and semantic masks, verify baseline tensor parity; freeze the same original-coordinate source corruption for all 300 updates.
3. `pretrain`: freeze critical code and run pretraining contracts.
4. `inference B`: fixed FULL125 weights, crop1.50 inference.
5. `train`: the single C training, FULL+frozen BN, last300 only; no evaluation during training.
6. `inference C`, `inference D`, then `inference freeze`: finalize all predictions/heatmaps before scoring.
7. `review`: reuse the existing blinded review form, add missing cases; never auto-fill human answers.
8. `evaluate score`, `evaluate heatmaps`: official whole-object symmetry scoring and separate fixed-branch mechanism diagnostics.
9. `probes source`, `probes train_fit`: common-support heldout results and explicitly non-validation TRAIN-fit probes.
10. `report`, `audit`: fixed/paired galleries, decision and 32 final tests. `publish` only packages approved outputs; it does not train, commit or push.

The existing A prediction/heatmap cache is verified, not regenerated. `report.py` records the initial generated report; the subsequent editorial conclusion and completion note are preserved in Markdown. No scoring code changed during publication.

## Boundaries

No GT was introduced into training or pseudo labels; GT-based top5 analysis is explicitly a posthoc oracle, not a deployed selector. No center8, detector bbox/score/candidate, decoder, loss, training schedule or final-paper model change. Crop expansion also changes object scale/grid spacing and supervision support, so C−A is a total crop-pipeline effect, not pure support causality. Human coordinate review remains pending.

The latest plan's mechanism/system classification is primary. `TARGET_DATA_AND_TRAIN_FIT` names a proposed follow-up audit only, not another training run.
