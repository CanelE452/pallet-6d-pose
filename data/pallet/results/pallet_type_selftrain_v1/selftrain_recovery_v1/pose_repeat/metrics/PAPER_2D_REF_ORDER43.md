# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `data/pallet/results/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_repeat/runs/REF_ORDER43/weights/last.pt`
- checkpoint SHA-256: `a8d77a0dfef83f1a9d9c19b04962825a5af5d35f698563473174413d702a1fbb`
- populations: `PAPER_EVAL_PLASTIC_POS` (N=194) + `DEV_NEG2689` (N=2689)
- per-frame rows: `2883`

## Development result

- Box AP50:95: `0.719432`
- Box AP50: `0.894897534538637`
- 2D keypoint diagnostic: median `6.572973359762938` px; p90 `40.194848373428854` px; supervised N `1707`.
- DAY/NIGHT positive N: `144/50`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
