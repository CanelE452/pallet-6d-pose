# Pose-sensitive easy-to-hard pilot

Phase A: frozen R0/S0/S1/S2, identical two W/D candidate poses. D8 changes only RMSE9 to RMSE8 in the production score; not deployed.

Phase B was STOPPED before training. No optimizer steps or lambda calibration. Strict zero-lambda gradient equality failed at ~2.5e-6 absolute difference while loss/components were exact. User requested tolerance-based parity; this implementation used the stricter zero tolerance. Meaningful gradient inequivalence or CUDA nondeterminism has NOT been established. Do not relax tolerances posthoc or resume training without a new explicit decision.

Completed: renderer/side-table binding512/512, frozen occurrence projection <=0.05px, H finite-difference/PSD, coordinate transform and masks.
Prepared but unvalidated end-to-end: geo_loss.py and trainer.py. They are NOT evidence of completed M0/M1 experiments. trainer.py refuses to train while STOP.json exists.

Executed order:

```sh
python -m scripts.research.pallet_clean19_pose_sensitive_v1.prepare preflight
python -m scripts.research.pallet_clean19_pose_sensitive_v1.prepare phase_a
python -m scripts.research.pallet_clean19_pose_sensitive_v1.prepare geometry
python -m scripts.research.pallet_clean19_pose_sensitive_v1.test_contracts metadata
python -m scripts.research.pallet_clean19_pose_sensitive_v1.test_contracts calibrate
# STOP — no training or student evaluation
python -m scripts.research.pallet_clean19_pose_sensitive_v1.report
```

Original criterion/selector/checkpoints/cache never modified. No output namespace overwrite. Geometry sources are existing renderer pose_transform and GEOMETRY_SIDETABLE, not reconstructed GT. Finite-difference H is relative sensitivity, not a full Linear-Covariance reproduction.
