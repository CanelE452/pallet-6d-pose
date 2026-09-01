# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `challenge/yolo_pose_one_model/legacy_v1v2_ft/runs/LV1V2_FT_15EP_SEED42/weights/best.pt`
- checkpoint SHA-256: `71986e0564fb5231238a6360c0be2205d99a843b2d0692d7a323fa9ea9a84475`
- populations: `PAPER_EVAL_ALL_POS` (N=319) + `DEV_NEG2689` (N=2689)
- per-frame rows: `3008`

## Development result

- Box AP50:95: `0.717222`
- Box AP50: `0.922316330514048`
- 2D keypoint diagnostic: median `5.5473040406557566` px; p90 `31.033523454895047` px; supervised N `1314`.
- DAY/NIGHT positive N: `168/106`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `plastic_standard_110x130x11:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;plastic_standard_110x130x11:FINAL_MANIFEST_NOT_FROZEN;wood_small_80x59x14:CANONICAL_MIGRATION_NOT_PASS;wood_small_80x59x14:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;wood_small_80x59x14:SYMMETRY_NOT_FROZEN;wood_small_80x59x14:FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
