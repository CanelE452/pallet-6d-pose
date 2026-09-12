# Updated track decision after complete C comparison

Current recommendation: `NO_NEW_METHOD_FREEZE_EXISTING_STORY`. This supersedes the C-incomplete decision in the initial TRACK_DECISION_MATRIX.md; initial evidence is retained.

| Track | Research question | Stage0 status | Stage1 executed? | Main comparison | Primary result | P90 safety | Pose safety | Runtime/deployment cost | Real labels in training? | Extra inference sensor? | Novelty risk | Closest prior art | Evidence level | Paper candidate? | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Independent local-line confirmation? | PASS freeze | No new training; confirmation NOT_RUN | R0 vs frozen local-line | Existing development gain only | Not independently confirmed | Existing partial gain | Existing added8.378ms seed1 | No | No | Medium/high | Deep Hough line priors | DEVELOPMENT | No main candidate | Needs new independent population |
| C | Detection-only DA preserves geometry? | PASS | Yes9x900 | C2 vs C0/C1 | C2−C0 AP50-95 -0.004000 | PASS | PASS | Stock RGB graph; no extra parameters | No manual real GT; frozen pseudo labels | No | High: freezing/ST established | Soft Teacher; task-specific freezing | NEGATIVE_DEVELOPMENT | No | C_GEOMETRY_PRESERVING_DA_FAIL |
| D | Predict useful teacher normal components? | FAIL | Trust3x1500; students0 | Selected realized normal gain | All calibration thresholds unsafe | Students NOT_RUN | Students NOT_RUN | Training-only proposal | No | No planned | Medium/high | Learning to Reweight Examples | MECHANISM_ONLY | No | Harm safety fails |
| B | Reliable alignment Jacobian? | FAIL | No | Local derivative vs±1px PnP |6/256 catastrophic=2.34% | Training NOT_RUN | W/D branch switches | Loss-only proposal | No | No | High: LC-derived | Linear-Covariance Loss | MECHANISM_ONLY | No | Exceeds1% tail gate |
| E | RGB-D boundaries ready? | FAIL | No teacher/student | Manual non-ground hull vs depth | Clean coverage23.43% | Teacher NOT_RUN | Teacher NOT_RUN | RGB-D offline planned | No training; manual GT diagnostic | No planned RGB-student depth | High | Learning Using Privileged Information | MECHANISM_ONLY | No | Coverage below80% |

Primary-source boundaries remain in PRIOR_ART_BOUNDARY.md. Reused DEV is not independent confirmation; no automatic novelty or state-of-the-art claim.
