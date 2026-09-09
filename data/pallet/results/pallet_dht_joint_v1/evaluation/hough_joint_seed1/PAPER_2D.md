# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `data/pallet/results/pallet_dht_joint_v1/runs/hough_joint_seed1/weights/final.pt`
- checkpoint SHA-256: `0960fb32fd99fd0837a588792e07727298fc6f88f1e4ec3464efd69bd5574d37`
- populations: `PAPER_EVAL_ALL_POS` (N=319) + `DEV_NEG2689` (N=2689)
- per-frame rows: `3008`

## Development result

- Box AP50:95: `0.732589`
- Box AP50: `0.893562905940215`
- 2D keypoint diagnostic: median `6.896977995515455` px; p90 `41.48732863482036` px; supervised N `2818`.
- DAY/NIGHT positive N: `168/106`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `plastic_standard_110x130x11:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;plastic_standard_110x130x11:FINAL_MANIFEST_NOT_FROZEN;wood_small_80x59x14:CANONICAL_MIGRATION_NOT_PASS;wood_small_80x59x14:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;wood_small_80x59x14:SYMMETRY_NOT_FROZEN;wood_small_80x59x14:FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
