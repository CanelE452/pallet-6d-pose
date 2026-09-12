# Track C — incomplete execution, no C2 performance verdict

Stage0 PASS: actual autograd dependencies identify241,566 detection-only trainable parameters. Frozen tensor and dense raw-keypoint invariance passed a one-update probe. Final selection may still change.

| Available seed1 arm | AP50-95 | kp median px | kp P90 px | translation cm | yaw deg |
|---|---:|---:|---:|---:|---:|
| C0_seed1 | 0.738032 | 6.8596 | 42.1534 | 7.9888 | 1.2792 |
| C1_seed1 | 0.760135 | 7.4944 | 44.3198 | 9.1888 | 1.3751 |

Both fits consumed900 actual SGD updates, had identical initial tensor SHA and all900 actual synthetic batch hashes. Night subgroup N106. Detection/ranking, paired common-frame errors, gross20, normalized frame means, full MAIN pose and subgroup records are in PER_SEED.json.
C2 seed1 consumed900 updates but agent-written BN audit code raised KeyError before checkpoint save. Its weights were lost. This is an execution failure, not C_GEOMETRY_PRESERVING_DA_FAIL. C2 vs C0/C1 and three-seed inference are unavailable.
Prefix construction is corrected and regression-tested; final unverified weights now persist before audits. No extra900-update refit was authorized or performed.
A separate tsfm-peft-method-screen GPU process appeared at2026-09-12T14:58:33Z. No wait, kill or modification; remaining C fits NOT_RUN per resource rule.
900+900+900=2700 actual student updates consumed; only two final checkpoints/evaluations available. The requested8100-update comparison is NOT complete.
Other documented limitations: optional Albumentations skipped by installed API mismatch; PyTorch warned that cuBLAS strict determinism was not enabled. Actual input parity was checked, but bit-exact retraining is not claimed.
The C0 first detection-only evaluation is INVALID_FOR_POSE_DUE_TO_TASK_METADATA and preserved. A separate corrected metadata checkpoint has identical tensor weights; valid evaluations live in evaluation_pose/.
The earlier EXECUTION_BLOCKER/PAUSE_AUDIT files are historical event receipts. CPU D/B/E continuation and final Git disposition are recorded by FINAL_AUDIT.json; C itself remains unresolved.
