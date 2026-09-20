# DEV Evaluation Report

- role: `DEV` (development only; never a FINAL table value)
- checkpoint: `data/pallet/results/pallet_type_selftrain_v1/iou_selftrain/runs/IOU90/weights/last.pt`
- checkpoint SHA-256: `d439617a968280f56c9510e8f2505812b95632a30cb2586a4cbfb26846079452`
- populations: `PAPER_EVAL_PLASTIC_POS` (N=194) + `DEV_NEG2689` (N=2689)
- per-frame rows: `2883`

## Development result

- Box AP50:95: `0.699804`
- Box AP50: `0.8869668530679563`
- 2D keypoint diagnostic: median `7.715391807565386` px; p90 `42.69852330070315` px; supervised N `1707`.
- DAY/NIGHT positive N: `144/50`
- session-cluster bootstrap 95% CI: `UNAVAILABLE` because DEV_NEG2689 lacks capture_session_id metadata.

## Pose fields

- status: `BLOCKED`
- blocked reasons: `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;FINAL_MANIFEST_NOT_FROZEN`
- Restricted ADD-S AUC / rotation / translation / yaw: `null`
- These DEV values must not be copied into paper-final tables.
