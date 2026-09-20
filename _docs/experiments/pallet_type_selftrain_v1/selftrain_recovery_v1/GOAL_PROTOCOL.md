# Persistent self-training recovery goal

2026-09-20 KST. User objective: “selftraining을 했을때 개선이 될때까지 방법을 찾아볼수있어?”

## Scope and success audit

First pursue the currently requested ordinary-plastic problem. Keep the full objective active; a diagnostic, completed run, isolated favourable metric, or known-label training fit is not success. Other pallet types are not silently assumed improved.

Primary recovery evidence is R0 superiority on the same complete194 population: improved full-denominator symmetry PCK20 and improved original paper fixed-index matched-keypoint median, accompanied by AP50–95 and MAIN IoU3D that do not decline. Report AP50, AUROC/FPR95, matching coverage, F0 errors, rotation/yaw/translation and ADDsym as guardrails; no adverse metric is hidden. Stable repeat runs and a same-protocol source-only control are required before declaring a robust self-training improvement. Results from this repeatedly used DEV population are exploratory, not independent final confirmation. New evaluation labels must not be used for training or selection. An unqualified broad improvement claim is not justified by satisfying only a subset of these checks.

No evaluation-image filtering, oracle axis input, best-on-real checkpoint selection, test-data BN adaptation, redefinition of metric, or undocumented sweep. Preserve all attempts and report the search history. The existing final model is not replaced automatically. No restart/driver updates or unrelated process termination. GPU guard: <80°C and no foreign compute other than the pre-existing RustDesk process.

## What motivates the next intervention

R0 plastic PCK20 77.86%; additional synthetic-only training76.14%; Replay pseudo student77.26%; IoU90 student76.60%. Thus pseudo-label noise is not established as the sole explanation; additional learning itself can hurt. The current checkpoint comparison shows252 BN running-mean/variance tensors (plus126 counters) differ along with trainable weights.

Stage A: stored-weight/BN-buffer 2x2 diagnostic. Existing R0 and student are the controls; export student weights+R0 BN and R0 weights+student BN. Only running buffers change. Score all194 and fixedNEG2689 with unchanged metrics. Do not confuse an inference intervention with corrected training.

Stage B is to be separately locked before execution based on Stage A evidence: compare preserved-statistics training and/or restricted pose-head adaptation against the same synthetic-only control and raw-pseudo control. Keep data exposure and training budgets explicit. Raw pseudo labels would be a control to determine whether Replay targets help, not a silent replacement of the user's corrected-label objective.

Human truth-audit of a small training-only sample remains useful if a verified label-quality decision becomes necessary. No fabricated annotations or evaluation-label reuse. No claim that stable/IoU-pass labels are ground truth.

## Primary literature used to motivate, not guarantee, interventions

- Li et al., Revisiting Batch Normalization for Practical Domain Adaptation: https://arxiv.org/abs/1603.04779 — BN statistics can affect cross-domain behaviour; this does not establish the direction of benefit in this repository.
- Tarvainen & Valpola, Mean teachers are better role models: https://arxiv.org/abs/1703.01780 — weight-averaged teachers and consistency supervision. Not yet implemented as a new method here; the existing YOLO trainer already maintains a checkpoint EMA, which must not be mislabeled as a new online Mean Teacher pipeline.
- Liu et al., Unbiased Teacher: https://arxiv.org/abs/2102.09480 — pseudo-label bias and teacher/student learning motivate examining both label targets and learning stability. No paper result is treated as evidence that our experiment improved.

Goal status remains active until the full completion audit is satisfied. The immediately preceding advisory turn did not run an experiment; this turn revalidates current artifacts and begins an actual diagnostic intervention.
