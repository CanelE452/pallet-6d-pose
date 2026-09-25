# Selector recovery + frozen expert routing pilot

Run from repository root using the `pallet-yolo26` environment. Existing snapshots are intentionally immutable: rerunning a completed freeze/fit raises rather than overwriting results. Reproduction requires a separate copy/new namespace with the hash-bound private datasets/checkpoints. No base model training is performed.

Sequence (GPU is required for extraction, small fits and benchmark):

```bash
python -m scripts.research.pallet_selector_recovery_v1.preflight
python -m scripts.research.pallet_selector_recovery_v1.stage1_diagnose
python -m scripts.research.pallet_selector_recovery_v1.render 1
python -m scripts.research.pallet_selector_recovery_v1.report 1
# Stage1 commit/push; git_record 1
python -m scripts.research.pallet_selector_recovery_v1.synth_split
python -m scripts.research.pallet_selector_recovery_v1.extract_features synth
python -m scripts.research.pallet_selector_recovery_v1.extract_features real
python -m scripts.research.pallet_selector_recovery_v1.synth_labels
python -m scripts.research.pallet_selector_recovery_v1.router_features
python -m scripts.research.pallet_selector_recovery_v1.train_scorer
python -m scripts.research.pallet_selector_recovery_v1.render 2
python -m scripts.research.pallet_selector_recovery_v1.report 2
# Stage2 commit/push; git_record 2
python -m scripts.research.pallet_selector_recovery_v1.lock_real_scorer
python -m scripts.research.pallet_selector_recovery_v1.eval_real_scorer
python -m scripts.research.pallet_selector_recovery_v1.render 3
python -m scripts.research.pallet_selector_recovery_v1.report 3
# Stage3 commit/push; git_record 3
python -m scripts.research.pallet_selector_recovery_v1.build_router_dataset base_lock
python -m scripts.research.pallet_selector_recovery_v1.build_router_dataset plans
python -m scripts.research.pallet_selector_recovery_v1.build_router_dataset extract
python -m scripts.research.pallet_selector_recovery_v1.build_router_dataset assemble
python -m scripts.research.pallet_selector_recovery_v1.train_router
python -m scripts.research.pallet_selector_recovery_v1.lock_real_router
python -m scripts.research.pallet_selector_recovery_v1.eval_real_router
python -m scripts.research.pallet_selector_recovery_v1.benchmark
python -m scripts.research.pallet_selector_recovery_v1.render 4
python -m scripts.research.pallet_selector_recovery_v1.render 5
python -m scripts.research.pallet_selector_recovery_v1.report 4
python -m scripts.research.pallet_selector_recovery_v1.audit
# Stage4 commit/push; git_record 4
python -m scripts.research.pallet_selector_recovery_v1.audit --require-git
python -m scripts.research.pallet_selector_recovery_v1.report 0
# Integrated commit/push; verify live remote SHA, then cli
python -m scripts.research.pallet_selector_recovery_v1.cli
```

The production scorer and router decisions are frozen before reference scoring in separate processes. `train_scorer` and `train_router` install a runtime real-reference read guard before importing training dependencies. Exact synthetic labels are linked only after their predictions freeze. The router occlusion plan may use synthetic geometry for the original coverage constraints, never model errors.

Source contracts and hashes: `_docs/experiments/pallet_selector_recovery_v1/INPUT_BINDINGS.json`, `SELECTOR_FEATURE_CONTRACT.json`, per-stage locks. Large predictions, exact geometry and trained small checkpoints stay under `data/pallet/results/pallet_selector_recovery_v1/` and are not published. Public MD includes relative links to small, privacy-cropped real examples and charts; force-add those image paths because of repository ignore rules.

Interpretation limits: reused real DEV, geometry-derived real pose reference, plastic only, single seeds, synthetic group/source distribution shift, original base model exposure to the broader synthetic population. AUC gains are diagnostic pilot results, not independent final-test evidence or a deployment promotion. No pointwise remapping or new threshold sweep.
