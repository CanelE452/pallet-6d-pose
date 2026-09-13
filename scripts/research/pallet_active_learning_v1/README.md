# Label-blind active-learning acquisition pilot

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
