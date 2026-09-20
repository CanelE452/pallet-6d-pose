# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt`
- checkpoint SHA-256: `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`
- populations: `PAPER_EVAL_PLASTIC_POS` (N=194) + `DEV_NEG2689` (N=2689)
- per-frame rows: `2883`

## Development result

- Box AP50:95: `0.719432`
- Box AP50: `0.894897534538637`
- 2D keypoint diagnostic: median `7.275210225300859` px; p90 `39.362924818786816` px; supervised N `1707`.
- DAY/NIGHT positive N: `144/50`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
