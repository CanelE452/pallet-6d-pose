# Diagonal pose-sensitive EASY→HARD controlled pilot

Historical v1 is immutable. All outputs are exclusive-create; do not rerun completed stages into this namespace.

Run modules with the existing `pallet-yolo26` environment from the repository root:

1. `prepare`
2. `test_contracts` — three independent TRAIN-only GPU forward/backward runs, no optimizer; fixed atol=1e-5 / rtol=1e-4.
3. `trainer calibrate` — one fresh R0, coordinate-producing modules identified by module identity, existing **total** loss gradient norm, lambda ratio .25.
4. `trainer train --material PLASTIC --arm M0`, then PLASTIC M1, WOOD M0, WOOD M1. No evaluation, restart, or rescue. Exact frozen S1 inputs, original R0, 320 updates each.
5. `evaluate infer` — freeze student-alone evaluation300 before any scoring.
6. `evaluate score` — unchanged production D9 and posthoc oracle diagnostics; no oracle feedback into inference.
7. `evaluate diagnostics` — existing heldout256 and TRAIN19 / fixed synthetic TRAIN probe.
8. `evaluate source_pose` — exact renderer/target projection binding for heldout256 before pose metrics; no GT estimation.
9. `test_contracts archive`, then `report`.

For example: `python -m scripts.research.pallet_clean19_pose_sensitive_diag_v1.trainer calibrate`.

New loss: `mean_enabled_positive(sum(diag(H)*residual²*supervision)/Nvalid)`, corners0..7 only, real/ignored weights0. Effective coefficient is applied after pose gain, with unchanged E2E weighting and batch multiplier. It is a local diagonal covariance-trace proxy, not a full Linear-Covariance reproduction or a guarantee of preservation of other network outputs.

M0 uses the original TrueIgnore criterion. M1 uses the new subclass; the production selector, solver, historic artifacts, targets and geometry caches are never modified. Public report includes all requested charts and posthoc examples; large private caches/checkpoints stay local.
