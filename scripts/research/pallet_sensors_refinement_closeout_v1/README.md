# Fixed point refinement closeout

Use the existing `pallet-yolo26` Python environment, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `MPLCONFIGDIR=/tmp/pallet-sensors-mpl`. GPU commands require host GPU access. No system/driver/environment mutation is part of the run.

Run entry points from the repository root with `python -B scripts/research/pallet_sensors_refinement_closeout_v1/<file>`:

1. `preflight.py` — freezes source/method/population; on resume verifies bindings.
2. `analysis_panel.py` — existing P/R0/L full-precision analysis.
3. `test_actual.py` — actual cache descriptor/coordinate gate,1 disposable update; does not repeat if receipt exists.
4. `train_direct.py` — only D3×6000, durable optimizer/RNG/order-position state; completed fits skipped.
5. `select_direct.py cache`, then `select_direct.py select` — source validation predictions, exact synthetic selection, heldout.
6. `evaluate_direct.py infer`, then `evaluate_direct.py score` — actual images followed by canonical CPU scoring.
7. `analysis_panel.py --include-d`, then `secondary_statistics.py`.
8. `runtime_panel.py` — after all training/scoring exits;10 models,26×5 actual image runs each. No parallel scoring.
9. `posefix_adapter.py`, `receipts.py`, `confirmation.py` — prior/readiness and future-data preparation. Official raw checkouts must be at the bound commits; these entry points do not train an external network.
10. `report.py`, `figures.py`, `final_audit.py` — bounded conclusions and generated manuscript.

Raw weights/predictions/history remain ignored under `data/pallet/results/pallet_sensors_refinement_closeout_v1`. Stage only this script directory, the corresponding new experiment docs and `_docs/paper/sensors_refinement_closeout_v1`. Existing dirty user files and historical experiment outputs are not part of this work.

Checkpoint resume reuses stored optimizer, CUDA/CPU RNG and exact order prefix. CUDA `grid_sample` backward itself is not guaranteed bit deterministic. Performance-driven restarts or hyperparameter changes are not allowed. Recorded seed-mean results are not an ensemble. Final scientific evidence, execution completeness and missing paper comparisons are separate statuses.
