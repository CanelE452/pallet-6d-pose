# Model-conditioned selector v1

Only the user's prescribed phases 1–3. Never retrain keypoint networks or fit on real GT.

Run with the existing `pallet-yolo26` environment from repository root. GPU inference/training requires host CUDA, not the restricted sandbox. Commands are separate processes deliberately: runtime file-access guards prohibit real references during fitting and real decision generation.

```bash
python -m scripts.research.pallet_model_conditioned_selector_v1.prepare
python -m scripts.research.pallet_model_conditioned_selector_v1.stage1
python -m scripts.research.pallet_model_conditioned_selector_v1.examples
# Publish stage1 report and exact figures (image files are globally gitignored).
python -m scripts.research.pallet_model_conditioned_selector_v1.synth_infer
python -m scripts.research.pallet_model_conditioned_selector_v1.train_scorers
python -m scripts.research.pallet_model_conditioned_selector_v1.synth_test
python -m scripts.research.pallet_model_conditioned_selector_v1.audit
# Publish stage2, before real comparison.
python -m scripts.research.pallet_model_conditioned_selector_v1.lock_real
python -m scripts.research.pallet_model_conditioned_selector_v1.evaluate_real
python -m scripts.research.pallet_model_conditioned_selector_v1.final_examples
python -m scripts.research.pallet_model_conditioned_selector_v1.audit --final
python -m scripts.research.pallet_model_conditioned_selector_v1.report
python -m scripts.research.pallet_model_conditioned_selector_v1.status
```

Completed fits, TEST and real decisions reject repeated execution. Partial fits are not silently restarted: inspect the existing immutable checkpoint and VAL artifacts first. `stage1` and `synth_infer` can verify their completed locks without recomputation. Figure/report rendering is not new inference or training. Existing upstream artifacts are read-only and all writes are confined to this namespace.

`FRAME_DECOMPOSITION_PRIVATE.json` in the public docs is a binding to the detailed local file, not the coordinates themselves. Public per-group reports and deterministic explanatory examples cover improvements and damage. Selector weights are tiny (~3.4 KB each) and are explicitly versioned; raw predictions and image pools remain local.

Reports contain 19 required charts and 12 actual-image examples. Stage1/2/3 Git receipts are after-push snapshots; subsequent receipt commits have a different HEAD. Never use broad `git add -A`, force push, or include unrelated untracked experiments.

Research limits: real128 and syntheticTEST are previously viewed development-heldouts; camera-facing convention remains; oracle is GT-dependent/nondeployable; no independent confirmation; no severity-based routing. Model-specific fitting uses one model output per image, unlike the old pooled two-model scorer. A successful predeclared DEV decision is a frozen candidate, not a production deployment.
