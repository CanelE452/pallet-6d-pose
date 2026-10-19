# B/I-conditioned local observation experiment

This namespace contains the fixed six-fit research experiment and its audits. It does not change inference defaults, historical models, or the paper. The result and exact replay commands are in [the public reproduction guide](../../../_docs/experiments/pallet_bi_conditioned_observation_v1/REPRODUCE.md).

Run modules from the repository root with `python -B -m scripts.research.pallet_bi_conditioned_observation_v1.<module>`. Use the recorded environment and private input archive described in that guide. Public scalar results and saved output poses support offline inspection; the private RGB, masks, synthetic GT, historical caches, six checkpoints, and original seals are required for a full replay.

| Stage | Modules | Boundary |
|---|---|---|
| Input ancestry | `input_contract` | Read-only same-N0/RGB/mask/state/source audit; real references hashed only |
| Common observations | `observer`, `features`, `bank`, `audit_bank` | Same cached N0 and automatic states; common cue union; no GT or pose solve |
| Synthetic supervision | `supervision` | Separate whole-object C1/C2 virtual-corner targets; raw renderer matrix preserved |
| Six fixed fits | `models`, `training` | TRAIN-only normalization, paired initialization and fixed seeds; no new real labels |
| Blind real prediction | `solver`, `predict`, `registry` | Frozen common solver and explicit input allowlist; all automatic outputs sealed before scoring |
| Postseal analysis | `evaluate`, `diagnostics`, `candidate_diagnostics` | References become readable only after the complete prediction seal; diagnostic arms stay separate |
| Independent verification | `independent_audit` | NumPy score replay and independent raw-pose metric/statistics; no optimizer or new solve |
| Public derivatives | `artifacts`, `figures`, `gallery`, `publication_audit`, `verify_remote_publication` | Sanitized evidence, derived images, byte/hash audit and ordinary remote-main verification |

The scorer consumes 181 numeric values, 181 validity values, and two condition values. COMMON uses `(0.5, 0.5)`; BI uses the frozen automatic B/I one-hot condition. Fixed partial-line replacements are excluded from neural loss and scoring. Their normal-only likelihood remains in the common solver. The target is `log1p(norm(actual_applied_xy - aligned_virtual_corner) / 2px)`, with one whole-object symmetry selected against all finite original N0 first-eight coordinates. These virtual corners are not certified physical material edges.

`test_models --artifact-root <new-directory>` runs nine constructed tests with MockAdamW and zero real optimizer constructions. `test_observer`, `test_solver`, `test_supervision`, and `test_independent_audit` provide separate analytic checks. Tests are not additional scientific fits. See the public guide for the actual six-fit counts, technical attempts, immutable archive requirements, and reference-read order.
