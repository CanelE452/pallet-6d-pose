# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt`
- checkpoint SHA-256: `1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c`
- populations: `COMMON_DEV_POS128` (N=128) + `DEV_NEG2689` (N=2689)
- per-frame rows: `2817`

## Development result

- Box AP50:95: `0.574189`
- Box AP50: `0.7196850046668395`
- 2D keypoint diagnostic: `UNAVAILABLE` because current GT-v2 visibility is unreviewed and supervised keypoint N is 0.
- DAY/NIGHT positive N: `100/28`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
