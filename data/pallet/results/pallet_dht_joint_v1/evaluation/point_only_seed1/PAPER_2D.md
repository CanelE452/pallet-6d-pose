# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `data/pallet/results/pallet_dht_joint_v1/runs/point_only_seed1/weights/final.pt`
- checkpoint SHA-256: `184941f0a6dad49b878a4f9f727b25606eea21122cb038d9ef54835cc2508f56`
- populations: `PAPER_EVAL_ALL_POS` (N=319) + `DEV_NEG2689` (N=2689)
- per-frame rows: `3008`

## Development result

- Box AP50:95: `0.763344`
- Box AP50: `0.9296351448377037`
- 2D keypoint diagnostic: median `6.995097656884383` px; p90 `38.19186490658345` px; supervised N `2818`.
- DAY/NIGHT positive N: `168/106`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `plastic_standard_110x130x11:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;plastic_standard_110x130x11:FINAL_MANIFEST_NOT_FROZEN;wood_small_80x59x14:CANONICAL_MIGRATION_NOT_PASS;wood_small_80x59x14:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;wood_small_80x59x14:SYMMETRY_NOT_FROZEN;wood_small_80x59x14:FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
