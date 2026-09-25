# S1 + frozen GEO_LINEAR zero-init adapter pilot

One architecture / one320-step fit / last checkpoint only. Run from repository root with `pallet-yolo26` Python. Existing freezes are immutable; completed phases cannot be rerun in place. Raw tensors, model checkpoints, exact predictions and labels stay private under `data/pallet/results/`. Public MD includes small ROI examples and figures.

```bash
python -m scripts.research.pallet_single_model_preserve_v1.preflight
python -m scripts.research.pallet_single_model_preserve_v1.prepare
python -m scripts.research.pallet_single_model_preserve_v1.baseline
python -m scripts.research.pallet_single_model_preserve_v1.parity    # GPU
python -m scripts.research.pallet_single_model_preserve_v1.train     # GPU
python -m scripts.research.pallet_single_model_preserve_v1.infer     # GPU
python -m scripts.research.pallet_single_model_preserve_v1.evaluate
# Only if decision != PRESERVATION_SUPPORTED:
python -m scripts.research.pallet_single_model_preserve_v1.supervision_gap
python -m scripts.research.pallet_single_model_preserve_v1.render
python -m scripts.research.pallet_single_model_preserve_v1.audit
python -m scripts.research.pallet_single_model_preserve_v1.report
# Commit/push only this experiment namespace and force-add its report figures.
# Only if gap == MIN_HARD_LABELING_JUSTIFIED:
python -m scripts.research.pallet_min_hard_labels_v1.inventory
python -m scripts.research.pallet_min_hard_labels_v1.expand_pool_audit
python -m scripts.research.pallet_min_hard_labels_v1.report_audit
# Metadata insufficient => stop; do NOT create fake annotation task or retrain.
```

Training uses3 external per-scale adapters on frozen Pose26 one-to-one final keypoint projection. The functional training decoder is bit-exact to the stock in-place inference decoder, but supports backward. Only x/y residual channels are applied. Original model parameters/BN stats and GEO_LINEAR are hash-checked unchanged. No clipping cap, auxiliary pose loss, confidence training, architecture search or real DEV checkpoint selection.

Learning guards deny real-evaluation/anchor reference reads. The first32 source sanity images are used only for parity, not training. Original Clean10 augmented tensors and source512 immutable caches drive the locked4/4/8 batch composition. Source256 is the same pre-existing evaluation probe.

For this execution, conditional label preparation stops at metadata inventory: eligible known-hard general-plastic frames span only2 recordings, not the required3. No `label_hard` command is available because no valid queue exists. User labels are never fabricated. A future run requires additional model-independent hard metadata and a new selection lock before opening predictions or presenting a dedicated blind GUI.
