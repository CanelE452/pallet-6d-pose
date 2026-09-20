# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `data/pallet/results/pallet_type_selftrain_v1/selftrain_recovery_v1/bn_probe/checkpoints/R0_STUDENTBN.pt`
- checkpoint SHA-256: `a78f941f165243ead1956bd1089368e8d797f79ca2b86c0985d39d376907a5bf`
- populations: `PAPER_EVAL_PLASTIC_POS` (N=194) + `DEV_NEG2689` (N=2689)
- per-frame rows: `2883`

## Development result

- Box AP50:95: `0.705018`
- Box AP50: `0.8523972598364014`
- 2D keypoint diagnostic: median `6.722445023059364` px; p90 `35.621199880388076` px; supervised N `1707`.
- DAY/NIGHT positive N: `144/50`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
