# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `challenge/yolo_pose_one_model/runs_ft/ft_a_real157_neg259_synth12k/weights/best.pt`
- checkpoint SHA-256: `e14c59e963607d72e5c63fd59c312674a276be73fdbc4ce194720c8d01c3db22`
- populations: `PAPER_EVAL_ALL_POS` (N=319) + `DEV_NEG2689` (N=2689)
- per-frame rows: `3008`

## Development result

- Box AP50:95: `0.839944`
- Box AP50: `0.9849316844214525`
- 2D keypoint diagnostic: median `4.553476585990938` px; p90 `16.865578696642093` px; supervised N `1314`.
- DAY/NIGHT positive N: `168/106`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `plastic_standard_110x130x11:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;plastic_standard_110x130x11:FINAL_MANIFEST_NOT_FROZEN;wood_small_80x59x14:CANONICAL_MIGRATION_NOT_PASS;wood_small_80x59x14:POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;wood_small_80x59x14:SYMMETRY_NOT_FROZEN;wood_small_80x59x14:FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
