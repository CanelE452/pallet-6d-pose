# Paper contribution screening decision

No main proposed method is frozen. This is a resource/implementation-limited
screen, **not** five completed negative training experiments. In particular,
C2 is unresolved, not disproven. All comparisons remain development/mechanism
evidence; none qualifies as independent confirmation.

| Track | Research question | Stage0 status | Stage1 executed? | Main comparison | Primary result | P90 safety | Pose safety | Runtime/deployment cost | Uses real labels in training? | Extra sensor at inference? | Novelty risk | Closest prior art | Evidence level | Candidate for paper? | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Does frozen local refinement generalize independently? | PASS: candidate audit/freeze | No new training; confirmation NOT_RUN | R0 vs image_line_only seed1 | Existing3-seed mean kp6.070 vs R0 6.616px; original overall gate not passed | Existing development gain, not confirmed | Existing partial development gains | Existing seed1 added median8.378ms | No | No | Medium/high: learned line priors established | Deep Hough Transform; Deep Hough-Transform Line Priors | DEVELOPMENT | Existing study only; not confirmed main method | No established untouched population |
| C | Can detection-only DA preserve geometry? | PASS: real dependency/one-update freeze probe | Partial: C0/C1 seed1 evaluated; C2 seed1 checkpoint lost after900 updates; remaining NOT_RUN | C2 vs C0/C1 | C2 comparison unavailable; C1 seed1 AP50-95 improves0.02210 over C0 | C2 unknown; C1 seed1 worse | C2 unknown; C1 seed1 worse | Stock RGB-only graph; no new latency benchmark | No manual real labels; existing pseudo labels | No | High: branch freezing/ST established | Soft Teacher; task-specific freezing | DEVELOPMENT, incomplete execution | No, unresolved | Agent audit bug; then foreign GPU occupancy. No extra-budget refit |
| D | Can useful teacher normal components be predicted safely? | FAIL | Trust3x1500 done; student0 | Predicted-gain selection vs realized normal gain | No calibration-safe threshold in any seed; selected test coverage0 | Student NOT_RUN | Student NOT_RUN | Trust/teacher intended training-only; student not built | No; synthetic GT for trust only | No planned | Medium/high: reliability reweighting established | Learning to Reweight Examples | MECHANISM_ONLY | No | Calibration harm exceeds25% even when mean gain positive |
| B | Is local x/z/yaw sensitivity reliable for loss design? | FAIL | No; gradient/student NOT_RUN | 0.01px local derivative vs canonical PnP at±1px |6/256 catastrophic samples=2.34%, limit1% | Training NOT_RUN | Selector branch discontinuities invalidate tail gate | Loss-only proposed; deployment not changed | No | No | High; LC projection prior art | Linear-Covariance Loss | MECHANISM_ONLY | No | Good median Jacobian error does not excuse failures in the tail |
| E | Are RGB-D boundaries ready for privileged geometry teaching? | FAIL | No teacher/student | Manual non-ground silhouette vs depth discontinuities | Clean-reference coverage23.43%, required80%; conditional median2px | Teacher NOT_RUN | Teacher NOT_RUN | RGB-D offline teacher planned; no fitted teacher/student | No training; manual real GT only for sensor diagnostic | No planned RGB-student depth input | High: LUPI/distillation established | Learning Using Privileged Information | MECHANISM_ONLY | No | Sparse boundary support; no scale/offset fitting to rescue |

See [PRIOR_ART_BOUNDARY.md](PRIOR_ART_BOUNDARY.md) for primary-source links and
the limited scope of the novelty check. No numerical score selects a winner.

`RECOMMENDED_PAPER_PATH = NO_NEW_METHOD_FREEZE_EXISTING_STORY`

The current defensible story is the strong synthetic baseline, existing
controlled adaptation studies, local-line development result and measured
failure mechanisms. C requires a repaired, fully accounted bounded comparison
before any C2 conclusion. A requires new independent acquisition for a stronger
generalization claim. Neither missing experiment is replaced by speculative
novelty. Paper-facing final files are unchanged.
