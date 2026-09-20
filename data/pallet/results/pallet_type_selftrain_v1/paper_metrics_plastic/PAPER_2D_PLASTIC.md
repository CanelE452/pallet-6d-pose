# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `data/pallet/results/pallet_type_selftrain_v1/runs/PLASTIC/weights/last.pt`
- checkpoint SHA-256: `9821ed665359bad8184e067d7d798a64571a1f35d7f8f01fa1db2db6d63f6b0b`
- populations: `PAPER_EVAL_PLASTIC_POS` (N=194) + `DEV_NEG2689` (N=2689)
- per-frame rows: `2883`

## Development result

- Box AP50:95: `0.700035`
- Box AP50: `0.8794945551258575`
- 2D keypoint diagnostic: median `7.545770945472884` px; p90 `40.089974086498486` px; supervised N `1707`.
- DAY/NIGHT positive N: `144/50`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
