# Reproducibility

## Material extension addendum

The original Plastic320 artifacts and tables below remain unchanged. Wood uses the same frozen Replay9/38 and two new matched320-update students, with361 sampled unique images and a provenance-locked45-image panel. See `../../experiments/pallet_material_selftrain_closure_v1/REPRODUCE.md` for the separate executable workflow, paired checks, prediction-before-scoring lock, material tables, 30 contract tests, and the updated13-page build. The bounded640-update Plastic test remains supplemental. Raw images/coordinates/checkpoints are not a public dataset release. Trusted Wood pseudo-quality and independent physical6D remain unresolved. Only externally known material routing is evaluated.

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

## Subsequent bounded diagnosis

The commands above reproduce the original closure at `9238735e`; rerunning `package prepare` on the revised manuscript would regenerate original package prose and is unnecessary. The post-hoc budget experiment has separate commands, immutable inputs/results, and build/audit logs in `_docs/experiments/pallet_visible_transfer_closure_v1/REPRODUCE.md`. It adds only two students, 640 updates each. Original first320 prefixes are bit-exact. The original tables and core results are preserved; `visible_transfer_appendix.tex` adds the mixed results without promoting a new main checkpoint.
