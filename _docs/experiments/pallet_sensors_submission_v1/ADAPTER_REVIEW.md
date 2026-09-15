# Changed-interface review before prior DEV opening

No changes to locked `prior_model.py`, `prior_data.py`, `train_prior.py`, losses, seeds, order or budget were made during training.

Read-only review found two adapter bookkeeping cases before any prior DEV inference:

- Canonical `replace_points` requires the replacement artifact's `arm` to equal its output arm. Raw prior replacements therefore carry `PRIORn_raw`, while capped output carries `PRIORn`; otherwise raw canonical scoring would stop rather than produce comparable results.
- Missing input poses return unchanged without entering the added network. Raw prior output preserves every invalid input landmark and center8; it cannot fill a missing baseline point with the prior expectation. The selected capped wrapper already preserves those inputs.

The CPU interface regressions include absent pose, empty detections, zero lambda/cap, one application of lambda, original-diagonal cap, invalid points/boxes and bilinear target boundary behavior. Raw/capped prior predictions will also pass canonical box/score/order/center and fixed-denominator checks during actual evaluation. Adapter inspection alone is not actual model evaluation.

Before the first expanded runtime measurement, the one-model memory transition was also corrected to release the temporary `head` variable used for parameter counting. Otherwise it would retain the last large prior backbone after the panel dictionary was cleared. The new transition requires zero live PyTorch CUDA allocation before loading each isolated-memory panel's first model. This affects measurement bookkeeping, not any training, model output, or historical timing file.

Inspection of the installed Torch2.1.1 `torch.backends.cudnn.flags` signature found that its default `enabled` is **False**. Supplying only `allow_tf32=False` in the new inference adapter would therefore disable cuDNN. Before any source selection, DEV inference or timing, both new inference contexts were made explicit: `enabled=True, benchmark=False, deterministic=False, allow_tf32=False`. This matches the cuDNN-enabled float32 network path used in the completed GPU parity check and ongoing main training. The adapter context restores prior backend settings before R0's next call. No training code, optimizer or initialization was changed; no prior DEV result was available when this interface correction was made. A CPU flag/restoration regression records the intended context without an extra network fit.

Earlier `IMPLEMENTATION_BLOCKED.json` is retained as the historical failed parity attempt. Its resolution is the executed-operator epsilon amendment and subsequent hashed `IMPLEMENTATION_COMPLETE.json`, `NETWORK_PARITY.json`, `TRAIN_OPERATOR_PARITY.json`, `GPU_PARITY.json`. It is not an unresolved terminal verdict and was not repaired by broadening the declared tolerance.
