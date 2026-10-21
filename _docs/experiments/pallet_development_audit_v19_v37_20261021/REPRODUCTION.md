# Reproduction scope and required inputs

The executable `analysis/reproduce.py` uses public pseudonymous scalar errors. It recomputes mean, median, linear P90/P95, maximum, success, missingness, T>50/T>100/R>45, paired recovery/damage, common-output means and each historical gate. Its gate tolerance is1e-7. A relative tolerance is used only to verify floating-point aggregation of the retained extreme appearance-only translations; it does not relax any method gate or clip outliers.

The public scalar CSV is derived evaluation output, not pose/reference ground truth. Success rows retain the complete denominator. No raw RGB or physical pose tensors can be reconstructed from it.

## Omitted material needed for raw replay

1. Rights-cleared original synthetic/real RGB, immutable image identities, K, physical dimensions and renderer annotation contracts.
2. Exact original observations, state/partial-line/profile packets, cached masks, native coordinate transforms and producer/covariance/CAL stage seals. Historical CAL residuals cannot silently become another producer's CAL model.
3. Frozen YOLO/N3, H1 and optional frozen face encoder checkpoints; original training metadata, own-bank TRAIN-only normalizers and split seals. These weights are not published here.
4. Original candidate banks, per-action labels, selected R/t outputs, historical binding protocols and exact original source bytes. Normalized public snapshots retain logic for review but differ in path bytes and therefore cannot pass historical source-hash bindings unchanged.
5. Required upstream repository dependencies and their versions; an installation or paid resource is not part of this audit.

Every essential base source is linked with SHA256 in `CONTRACT_SOURCE_PROVENANCE.json`. Historical experiment source originals and normalized copies are distinguished in `SOURCE_SNAPSHOTS.json`. Scalar input table hashes are in `INPUT_TABLE_HASHES.json`. A base commit alone does not reconstruct uncommitted experiment overlays or private data.

## Historical execution shape, not permission to re-run

Use a fresh, owned experiment directory with reviewed source/input bindings. Do not point these commands at completed official runs. The placeholders below refer to original, correctly bound files; the normalized snapshots do not bypass their guards.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# v37 feature/coverage stages completed; training NOT executed:
python -B "$EXPERIMENT_ROOT/raw_leave_two_bank_20261021_v37/experiment_v2.py" features
python -B "$EXPERIMENT_ROOT/raw_leave_two_bank_20261021_v37/verify_features.py"
python -B "$EXPERIMENT_ROOT/raw_leave_two_bank_20261021_v37/score_coverage.py"
```

The completed v37 source run used one CPU thread, an unchanged900-second feature-export cap, all28 six-point leave-two-out subsets, fixed original predicted native geometry, the original admission/dedup rules, and own TRAIN768 normalization. Actual feature export814.485988s; stored coverage scoring49.555099s. The aborted first progress-writer attempt remains documented; no training was attempted.

v30 frozen ROI inference used an existing compatible runtime, the same checkpoint/input640, one1.5x predicted-box crop, affine round-trip checks and original full-image K/gray SubPix/move caps. Full-image TRAIN8 replay passed unchanged technical tolerances. This does not establish full1003-frame exporter equivalence or new independent generalization.

## Data needed for the unresolved cause analysis

The coordinate×correspondence2×2 diagnostic is not supplied as a completed experiment. It requires explicit correspondence contracts; independent real owner/visibility labels do not currently exist. Renderer GT may support a synthetic geometric self-hidden split, but v36 pooled all8 points and did not perform this split. Actual external occlusion cannot be inferred merely from geometric self-hidden status. N3's softmax-mean decoder is confirmed in source; its pre-mean evidence was not retained for all audit cases in this package.

Future study design must distinguish a fixed incumbent, multiple unadopted Pareto comparators, and independent confirmation. These development results and CAL reuse do not justify a new independent-test claim or a no-damage guarantee. No inference-time rule in this publication is allowed to consume reference answers.
