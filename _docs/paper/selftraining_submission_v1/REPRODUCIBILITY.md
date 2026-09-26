# Reproducibility

All commands run from repository root. Python: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`.

```bash
python -m scripts.research.pallet_selftraining_paper_closure_v1.prepare
python -m scripts.research.pallet_selftraining_paper_closure_v1.freeze_predictions
python -m scripts.research.pallet_selftraining_paper_closure_v1.evaluate
python -m scripts.research.pallet_selftraining_paper_closure_v1.report
python -m scripts.research.pallet_selftraining_paper_closure_v1.package prepare
python -m scripts.research.pallet_selftraining_paper_closure_v1.package build
python -m unittest scripts.research.pallet_selftraining_paper_closure_v1.test_closure
python -m scripts.research.pallet_selftraining_paper_closure_v1.audit
```

Completed expensive stages verify existing locks and do not rerun fitting. The closure includes NO training entry point. A fresh clone requires the private hash-bound datasets/checkpoints and frozen result caches, or regeneration from the historical training scripts. It is not an independently downloadable public benchmark. Do not overwrite historical artifacts to satisfy a changed hash.

Historical fit implementation: `scripts/research/pallet_type_selftrain_v1/recovery_pose.py` and `recovery_repeat.py`. Arguments and final checkpoint SHA256 for every arm: `CORE_COMPARABILITY_AUDIT.json`. Exact input paths, masks, source data, images, teacher, split, and evaluator dependencies: `INPUT_BINDINGS.json`, `PREDICTIONS_LOCK.json`, `PAPER_AUDIT.json`. The paper's 2D and6D values are generated, not copied from incompatible194/300-image historical reports.

CPU: existing environment, four evaluation workers. GPU is needed only for128 frozen Replay inference when no saved output exists; RTX3080 checked, thermal stop80C, observed48–54C. No reboot/driver changes or unrelated process termination.

Tectonic is reused from the existing local compiler. Its cache is COPIED to the new namespace; old paper and compiler cache are not modified. PDF build logs and rendered-page checks are retained in the new result namespace.

Numbers: generated_tables/NUMBER_PROVENANCE.json binds each table to source path/hash; MANUSCRIPT_NUMBER_PROVENANCE.json binds result macros to JSON paths. Counts that describe the protocol are in the comparability audit; transitions and per-frame categories are in PAIRED_ANALYSIS.json. MAIN128 corner denominator985, matched120frames/931corners, anchor16frames/66points remain distinct.
