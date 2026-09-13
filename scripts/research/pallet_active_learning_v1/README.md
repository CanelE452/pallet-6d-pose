# Label-blind active-learning acquisition pilot

## Authorized retrospective training comparison

The user subsequently authorized immediate student training using existing
GT-v2 labels in a separately session-partitioned simulation. This uses
`simulation.py`, `simulation_evaluate.py`, and `simulation_report.py`.
It does not replace annotations for the original97-image unlabeled preview.

Frozen budget:4 acquisition methods,30 selected labels each,3 student seeds,
300 actual optimizer updates per fit (3600 total). Acquire on the pool subset
only, reveal only selected pool GT, finish all12 fits before evaluation.
Standard PoseLoss26 preserves occluded visibility1; do not substitute the old
pseudo-label TRUE_IGNORE loss. The unchanged canonical evaluator receives only
the reserved positive subset plus fixed DEV negatives. Any literal historical
319/new-training0 metadata in its pose JSON is superseded by each
ACTUAL_POSE_BINDING.json, not interpreted as this run's population or cost.

New evidence is in `retrospective_v1/` under this experiment's doc/raw roots;
`CURRENT_RESULT.json` points to the completed result when available.
These students have trained on part of the old DEV319, so never evaluate them
on the whole oldDEV319 and call it untrained/independent evaluation.

The one-time phases are `simulation.py setup`, `acquire`, `prepare`, and
`driver`, in that order. Do not rerun setup after changing HEAD: protocol
receipts are immutable. `driver` skips audited fits and refuses to overwrite
existing checkpoints. Run `simulation_evaluate.py` only after all12 fits,
then `simulation_report.py`. Evaluation reuses immutable prediction caches.
Use the `pallet-yolo26` Python environment and process-local CUDA library path
below, with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`; no system changes are needed.
2D and pose frame IDs have different spellings; their evaluation join is by
canonical image path with frozen image SHA verification, recorded in
`EVALUATION_AUDIT.json`.

## Original unlabeled acquisition-only pilot

This is a separately authorized acquisition pilot, not a restart of the closed
C diagnostic. Source model is immutable R0. No student trainer is invoked.

Entry point: `run.py initialize`, `extract`, `acquire`, `verify` in the
`pallet-yolo26` environment. Completed inference refuses overwrite. Read frozen
receipts before execution; do not rerun initialization after changing git HEAD.
GPU extraction uses process-only
`LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace`; never alter drivers or foreign
GPU processes. Raw feature arrays stay in ignored `data/pallet/results/pallet_active_learning_v1`.

Protocol, exclusions, preview selections and the blind annotation queue are in
`_docs/experiments/pallet_active_learning_v1`. The four30-image selections have
a97-image union, not30 total annotations. This is a preview, not approved human
labeling cost. No ground truth is fabricated or read for acquisition.

Photometric selected-pose instability is not calibrated error probability.
The weighted k-center is an explicit lightweight comparator, not a reproduction
of CLUE or an established novel method. Label-efficiency/pose results remain
NOT_RUN until manual annotations and a downstream protocol are available.

Tests: `python -m pytest -q scripts/research/pallet_active_learning_v1/test_acquisition.py`.
