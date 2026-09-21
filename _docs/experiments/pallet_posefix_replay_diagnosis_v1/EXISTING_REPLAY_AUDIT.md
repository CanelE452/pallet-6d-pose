# Existing experiment audit (read-only)

## The word “replay” has three different meanings here

`pallet_posefix_replay_v1` means **synthetic training-data rehearsal during real fine-tuning**. It is neither iterative inference (A) nor a numerical equality check (B). Its accuracy failure cannot be attributed to feeding its outputs back into itself: that experiment used one forward pass. The present diagnosis deliberately uses the committed **synthetic-only PRIOR1/2/3 last6000**, not the real+synthetic `last300.pt`.

| Existing experiment | Model / purpose | Completed result | Numerical / runtime issue and stopping reason |
|---|---|---|---|
| `pallet_sensors_submission_v1` | PoseFix-derived RGB+9 Gaussian maps, ResNet152, 68,661,641 parameters; three synthetic-only last6000 checkpoints | All three completed. Original source selection fixed λ=1 and 1%-original-image-diagonal cap. Historical DEV tables used a legacy metric; this diagnosis recomputes current corner8 whole-object symmetry metrics. | Original implementation validation stopped on stem parity before the executed TF BatchNorm epsilon was matched. Subsequent implementation/training completed. Later fixed-input cuDNN repeats first differed at up1; a recorded 3e-4 crop-coordinate component tolerance separated numerical replay from accuracy. No tolerance increase here. |
| `pallet_posefix_large_error_v1` | Start PRIOR1; 300-step real-only fine-tune on 9 images / 38 verified manual corners, raw-R0 / GT+radial-noise inputs | Training fit succeeded. DEV72 raw PCK10 45.23%; GREEN150 manual 68.14%; good5→bad10 damage 35/138 and 76/344. | Accuracy/generalization failure, not a failure to run. The model was not promoted. GT training perturbations are supervision, not inference inputs. |
| `pallet_posefix_replay_v1` | Start original PRIOR1; same real order, 300 steps, real8 + synthetic8 each step; synthetic GT replay | Completed. DEV72 raw PCK10 60.18%, GREEN150 manual 77.39%; damage 1/138 and 14/344. DEV hard recovery 4/156; green matched hard 1/4. Synthetic clean PCK10 90.95%, below original 94.02%. | Six of ten preregistered screening gates failed; stopped before that experiment's conditional selective self-training. No iterative feedback. Separate later user-authorized self-training exists; this is not a claim that self-training was never run. |
| `pallet_posefix_corner_gate_v1` | Frozen N2 and frozen real+synthetic Replay; hand-designed pair selection; no new learned refiner | GPU inference completed on 222 images. GREEN PCK10 79.736%, below N2 81.351%; DEV72 58.919%, below N2 59.459%. | CPU plan was explicitly amended to GPU at user request; stored-vs-recomputed Replay maximum difference 0.0001763 px, within that experiment's 0.01px threshold. Accuracy gate did not establish a replacement. That threshold is not imported into this diagnosis. |
| `pallet_posefix_utility_selector_v1` | Frozen N2/Replay endpoints, new 53,233-parameter selector, 1,500 steps | Training/evaluation completed. DEV PCK10 60.000%, GREEN 78.267%. Accepted the highlighted gross P0 miscorrection; not promoted. | First N2 source parity failed before training because cuDNN TF32 / historical batch16 / CPU decode differed. v2 runtime restored 1,616 clean predictions bit-exact. These were N2 compatibility corrections, not evidence of iterative PoseFix failure. Augmentation stability also initially stopped on R0 TF32 mismatch (0.0061798px), then used a separately recorded runtime correction. |

Historical controls and their filenames remain untouched and are hash-bound in `SOURCE_BINDING.json`. The source code confirms that Replay adds a synthetic loss term and exposures, rather than calling PoseFix recursively. Historical thresholds, support, metric definitions and populations differ; these numbers describe those experiments, not interchangeable current main-table measurements.

## Scope of this diagnosis

- All existing calibration 1,004, selection 1,031, heldout 1,985, reused DEV319, and separate square C4 DEV155 frames.
- Three original checkpoints, no training, no new seed, no detector inference, no model promotion.
- Two explicit feedback paths: RAW→RAW and CAPPED→CAPPED, max three passes. Each also stores raw and capped candidates from the **same input** for a clean cap comparison.
- Fixed R0 bbox, crop, instance and center; only native corner input maps change. Dimensions and GT are evaluation metadata, not neural input.
- Square prepared images have an existing 100px border; it is used once, with stored raw coordinates translated to prepared pixels for cropping and back for evaluation. This does not imply square supervised training.
- The three-seed study addresses canonical synthetic-only PoseFix. It does **not** establish the same causal decomposition for the later last300 Replay checkpoint.

## Git protection

At entry, local main is `0edd38ee40d3d73a9af3ed5c64b34ef668186107`; fetched origin/main is `a8a7c73ce50ac522502be61a47e082e3933d3cb8`. Two pre-existing commits are ahead, including a large prior research commit. New staging is restricted to this diagnosis. A push would also transmit those ancestor commits; an earlier transfer-approval denial must not be bypassed. Report the precise push status separately from completion of the scientific work.

During this execution, the user answered the explicit scope question: **include both existing commits in the push**. This newly granted authority covers that transfer; unrelated untracked directories remain excluded.
