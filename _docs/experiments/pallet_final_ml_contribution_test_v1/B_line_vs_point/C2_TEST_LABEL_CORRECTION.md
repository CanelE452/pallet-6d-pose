# C2 test-label correction (no training or prediction change)

The initial `test_canonical_assignment_roundtrip[C2]` used the involution `[2,3,0,1,6,7,4,5,8]`. It correctly tested generic coordinate/target reindexing, but it was **not the repository's canonical yaw-180 permutation**. Calling that fixture C2 yaw was inaccurate.

The repository's `symdht_local/constants.py` defines C2_YAW as `[5,4,7,6,1,0,3,2,8]`. `test_canonical_yaw.py` now explicitly checks this exact reference, its round trip and joint prediction/GT target reindexing. This extra check occurs after training; it is not misrepresented as an initial pretraining test. The original frozen test source is retained.

Neither permutation was ever applied to training or inference: existing L and new P reuse the cache's fixed canonical assignments, with no min-over-permutations loss. Architecture, data, weights, predictions, selection and optimization remain unchanged; no retraining is required or performed. This is a regression-test naming/coverage correction, not a new symmetry method or a performance repair.

The A padding-instance diagnostic initially reused the same mislabeled permutation. Its coordinate evidence is unchanged; `A_task_risk_robustness/CANONICAL_C2_DIAGNOSTIC_CORRECTION.json` provides the properly named yaw counterfactual and preserves the original artifact as a record.
