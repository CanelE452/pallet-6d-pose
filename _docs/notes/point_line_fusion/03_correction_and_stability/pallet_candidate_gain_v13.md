# V13: frozen candidate gain evaluation

User authorized adding a module that evaluates keep/half/full recovery candidates, and asked whether that changes the architecture. It changes the complete predictor by adding a learned head; YOLO topology/weights and the V10/V12 parent models stay frozen. No Hough branch in this isolation experiment, no self-training, real training, or production replacement.

Implementation: `scripts/research/pallet_candidate_gain_v13/`.
Artifacts: `data/pallet/results/pallet_candidate_gain_v13/index.html`, `REPORT_KO.md`, `PROTOCOL.json`, `SUMMARY.json`, `AUDIT.json`, `FAILURE_DIAGNOSIS.json`.

- Base = V12 learned_gate; full = V12 gated_amodal, same seed. Candidates = base + [0, .5, 1] * (full - base).
- Constant, geometry, image scoring arms; 3 seeds each, 2,000 updates, batch16, AdamW3e-4, synthetic3609 observations. Image head47,300 total parameters. Final checkpoints only, no real selection/tuning.
- Target = original-ID synthetic GT error(base) - error(candidate). SmoothL1 + .25 soft preference CE. Baseline score exactly0, argmax selection, ties keep. No GT/masks/condition names at inference.
- Same reused real297/9sessions/manual1251clicks and synthetic505 evaluation, 6 artificial stress conditions; same parent detections and reflect100 padding. Not fresh natural occlusion or 6D validation.

All three seeds failed the preregistered stage1 gate. Clean noninferiority and retention of >=80% of hard-stratum parent gains passed; crop20 noninferiority and beating full/geometry failed. Do not run joint refinement on this result: `JOINT_FOLLOWUP_DECISION.json` explicitly records no followup.

Three-seed mean point errors in pixels (clean is frame average, others named point strata):

| Condition / stratum | base | full | image scorer |
|---|---:|---:|---:|
| clean / frame mean | 5.537562 | 5.542445 | 5.539674 |
| corner20 / hidden280 | 16.539330 | 13.554489 | 13.637115 |
| crop20 / outside558 | 12.372119 | 16.496415 | 16.748662 |
| crop40 / outside571 | 27.770936 | 19.921341 | 20.222629 |

Candidate oracle crop20 outside mean11.565661px: privileged GT selection, not achievable-performance evidence. Scorer applied harmful corrections on66.3–78.0% of its applied crop20 outside points. Constant chose full on all91,200 valid evaluation point/seed rows; geometry chose full91,199 and half1. Image input did not establish additional gain.

Posthoc calibration evidence: crop20 outside true full gain in synthetic training +2.810..+3.140px, synthetic validation +0.374..+0.578px, real -4.554..-3.613px. Image scorer predicted real full gain +0.651..+1.087px. Supports mismatch in learned improvement prediction; does not isolate domain shift as the sole cause. Parent models already saw scorer training images: in-sample parent errors are an explicit limitation.

Audit PASS: 88,160 prediction frame rows and41,760 selection frame rows; exact argmax replay, candidate reconstruction max6.1035e-5px, metric reconstruction max1.0658e-14. Parent checkpoints and training sources unchanged. Manual GT was not reassigned; its annotation noise has not been newly measured.

Possible next experiment, not run: train score head on out-of-fold parent predictions and include helpful/harmful synthetic residual cases. If adding lines, propose independently evidenced alternative directions near points rather than only scaling the existing displacement. Current failure does not prove point/line fusion impossible and does not show a Hough benefit.

Delivery: HTML has condition/domain/seed/metric selectors and12 galleries comparing actual candidates, predicted gain and GT posthoc gain. Delivery receipts record browser rendering/opening and previously authorized completion-only Discord notification.
