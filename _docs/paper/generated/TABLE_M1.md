# Table M1 — Main method comparison

population `PAPER_EVAL`.  N 은 manifest 에서 읽는다.

```text
Method                             Population    N_pos  N_neg  corner↓   det↑  AP50-95↑   AUROC↑   FPR95↓   R med↓    yaw↓
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
YOLO26n-Pose (synthetic-only)      PAPER_EVAL      319   2689    4.420  0.975    0.7688   0.9921   0.0417        —       —
Proposed                           PAPER_EVAL      319   2689    4.180  0.984    0.7585   0.9953   0.0283        —       —
DOPE (same-data control)           PAPER_EVAL      319   2689 — — — — — — —
```

`R med` 와 `yaw` 는 `POSE_METRICS_STATUS = BLOCKED` 이라 비워 둔다.
2D 개선을 6D pose 개선이라고 쓰지 않는다.

SingleShotPose / PVNet 은 아직 평가하지 않았다 — 상태는 APPENDIX_TABLES.md 참조.
