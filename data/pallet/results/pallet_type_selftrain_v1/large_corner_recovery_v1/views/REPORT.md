# Large-corner recovery: multi-view R0 candidates

All194 ordinary-plastic DEV images, official C2 unchanged. Only first8corner predictions can change; original boxes/scores/confidences/center preserved. No evaluation-image rejection. Primary CONSENSUS selected before GT by whole-pose medoid, not oracle.

| Arm | PCK20 % | median8 px | P90_8 px | matched hard recovered / hard | good corners damaged / good | IoU3D |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 77.859 | 7.509 | 43.637 | 0/281 | 0/511 | 0.58572 |
| FULL960 | 70.919 | 7.830 | 64.181 | 34/281 | 43/511 | 0.47373 |
| FULL1280 | 68.539 | 8.746 | 89.399 | 31/281 | 64/511 | 0.45011 |
| CROP125 | 78.586 | 7.599 | 34.671 | 29/281 | 18/511 | 0.57130 |
| CROP175 | 77.264 | 7.514 | 43.637 | 26/281 | 22/511 | 0.57377 |
| FLIP640 | 74.686 | 8.012 | 66.337 | 22/281 | 42/511 | 0.57520 |
| CONSENSUS | 78.057 | 7.378 | 46.709 | 23/281 | 24/511 | 0.56764 |

Follow-up gate (not goal completion): {"at_least5_large_corner_recoveries": true, "at_least3_recovered_frames": true, "damage_rate_at_most1percent": false, "full194_PCK20_preserved": true}

GT best-whole-view diagnostic only: {"frames": 186, "corners": 1459, "hard": 281, "recovered": 72, "recovery_rate": 0.25622775800711745, "good": 511, "damaged": 4, "damage_rate": 0.007827788649706457, "matched_only": true, "canonical_GT_identity_aligned": true}

GT best view is never used for model output, pseudo-label generation, training, selection-rule tuning, or a performance claim. C4 is never treated as valid ordinary-plastic symmetry. Existing repeated DEV is not an independent test.
