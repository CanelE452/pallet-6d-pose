# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `challenge/yolo_pose_one_model/paper_selftrain_v3/V3A_TRUE_IGNORE__FULL/weights/last.pt`
- checkpoint SHA-256: `74eb3806abc4d639d428b1270ee30293e8cb89baeba2fda40e5118761e3a8338`
- populations: `PAPER_EVAL_ALL_POS` (N=319) + `DEV_NEG2689` (N=2689)
- per-frame rows: `3008`

## Development result

- Box AP50:95: `0.755378`
- Box AP50: `0.9413136558331038`
- 2D keypoint diagnostic: median `6.968986084278795` px; p90 `41.99534514090237` px; supervised N `2818`.
- DAY/NIGHT positive N: `168/106`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `plastic_standard_110x130x11:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;plastic_standard_110x130x11:FINAL_MANIFEST_NOT_FROZEN;wood_small_80x59x14:CANONICAL_MIGRATION_NOT_PASS;wood_small_80x59x14:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;wood_small_80x59x14:SYMMETRY_NOT_FROZEN;wood_small_80x59x14:FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
