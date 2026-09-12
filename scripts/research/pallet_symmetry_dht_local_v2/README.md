# Symmetry-aware DHT local fusion v2

Final bounded experiment correcting exactly four v1 equation-level mismatches:
anchor-vs-point loss scaling, structural support vs correction utility, hard-bin
vs continuous line supervision, and simultaneous fusion of alternative modes.
It reuses the immutable v1 1,792/256/512 export and does not alter v1 results.

Run modules from the repository root with
`python -m scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.<module>`.
CUDA requests never fall back silently to CPU.

## WLS implementation correction

Commit `187589a` used the first endpoint's residual for both endpoints in
`single_mode_wls`. The current function computes each endpoint's own residual.
Historical checkpoints were trained through the buggy function: loading them
with current code produces corrected-inference/posthoc behavior, not a replay
of their historical predictions. The new correction runs are stored separately
under `data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction`.
See [_docs/experiments/pallet_symmetry_dht_local_v2_wls_correction](../../../_docs/experiments/pallet_symmetry_dht_local_v2_wls_correction)
for the correction status, independent regressions, saved-prediction replay,
controlled oracle, and same-protocol retraining. Original results remain intact.
