# Independent confirmation: collect only after method freeze

No existing population has been established as never consulted during method
development. A directory named FINAL is not sufficient provenance. The sensor
validation population is explicitly development-only and cannot be reused as
independent confirmation.

Collect 12 independent acquisition sessions, 15 pre-specified frames each (180):
three indoor-day, indoor-night, outdoor-day and outdoor-night sessions each.
Balance plastic/wood, distance and elevation where feasible; retain some
occlusion. No new rendering substitutes for these observations.

Before opening annotations, freeze the proposed method, checkpoint SHA, every
threshold, evaluator revision, physical dimension contract and input recipe.
Compare only R0, the closest applicable existing baseline and at most one
proposed method. Do not use the population for calibration or selection.

Record raw RGB SHA256, acquisition session, 9 keypoints with visibility, known
dimensions, intrinsics and reference provenance. Independently duplicate at
least 30 annotations. Require zero image-hash overlap with adaptation/training,
DEV319 and all earlier diagnostic populations; verify session provenance too.
Resolve annotation disagreements without seeing method predictions.

Primary: pooled supervised keypoint median in raw pixels. Required safety:
pooled P90 and matched-detection coverage nonworse. Report raw-diagonal-normalized
frame mean, gross20, MAIN rotation/yaw/translation/IoU3D/ADDsym and pose coverage
separately. Run paired frame and session-cluster bootstrap (10,000 resamples,
seed20260912). A positive primary difference must have a session-cluster 95%
interval excluding zero to support confirmation. A performance failure does
not authorize another threshold/checkpoint search on this population.
