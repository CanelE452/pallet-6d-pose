# Final ML contribution test v1

This directory implements the attached final bounded protocol without changing old experiments. Read the frozen master and B design locks before running anything. GPU commands require the existing working NVIDIA userspace library path, not driver/system changes.

Entry points (all outputs confined to the matching new experiment roots):

1. `common.py` — initial source preservation and master lock (already frozen; do not rerun initialization).
2. `task_risk_audit.py qa|scores|catastrophe` — nontraining A audit.
3. `preflight.py` — source, train data, geometry, parameter/evidence/order locks.
4. `test_contracts.py`, `test_pipeline.py`, `test_canonical_yaw.py` — pytest; initial35 contracts passed before training, then39 pipeline checks and40 after the explicitly disclosed post-training canonical-yaw fixture correction.
5. `train_point.py` — exactly seeds1/2/3 ×6000 updates; final checkpoint only; never silently retry incomplete runs.
6. `select_point.py cache|select` — validation logits, synthetic-only temperature/shared-rule selection, then heldout.
7. `evaluate_point.py infer|score|runtime` — 3008 actual image forwards per seed, fixed canonical scoring and deployment timing.
8. `statistics_and_mechanism.py` — exact same-seed paired contrasts. Its original frozen mechanism path encounters missing optional domain metadata/empty subgroups; `mechanism_report_fix.py` completes that report without changing predictions, gates or the frozen statistic source. `mechanism_completion.py` and its receipt preserve the first metadata-only completion attempt.
9. `audit.py` — separate source preservation, training and evaluation integrity audit.
10. `report.py` — emits an apply_patch payload for final reports; does not directly edit Markdown.

The functional source-data logic lives in `preflight.py` and reuses old read-only `FeatureDataset`; there is deliberately no generic `select.py` filename (which could shadow Python's standard-library `select`). Evaluation, mechanism and runtime are grouped instead of creating duplicate wrappers. Deployment is `point_inference.PointInference` with GT-free `predict(bgr)`.

Typical environment:

```sh
LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/pallet-final-ml-mpl PYTHONDONTWRITEBYTECODE=1 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B scripts/research/pallet_final_ml_contribution_test_v1/train_point.py
```

Do not launch this if completed artifacts already exist. No architecture/budget/seed/real-selection retries are authorized. Large features/checkpoints/raw predictions remain ignored; source bindings, checksums and compact findings are committed in docs.

CPU canonical pose scoring additionally needs `PYTHONPATH=/home/minjae/Documents/github/pallet-pose/scripts/paper/pose_metric_closure_v1` because the legacy evaluator dynamically imports its companion modules. See `SCORING_ENVIRONMENT.json`. For a fresh replay of existing outputs, invoke the statistic functions and the corrected mechanism entry separately; do not rerun training or overwrite immutable artifacts.

Measured runtime conditions are disclosed in `RUNTIME_CONDITIONS_NOTE.json`: CPU scoring overlapped the timing start. Paired wall times are observations, not an isolated causal speed comparison. Runtime is not used by the scientific verdict gates.
