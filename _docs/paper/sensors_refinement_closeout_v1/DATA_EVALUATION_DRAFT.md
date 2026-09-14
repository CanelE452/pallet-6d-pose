# Data and evaluation draft

Source partitions: {'calibration': 1004, 'heldout': 1985, 'selection': 1031, 'train': 55980}. Matched usable training rows:55915. The new D fits use exactly the frozen P seed-specific order,3×6,000 actual updates,batch16 and96,000 nominal exposures per seed. AdamW LR0.001,betas(0.9,0.999),WD0.0001,100-step warmup,cosine final fraction0.1,clip5. No real image/feature/label or negative data enter the optimizer. The frozen detector itself and prior research have their own training/selection histories; no claim of no real annotation anywhere is made.

D selection uses the old synthetic1031-frame partition and all10 lambda/cap rules, averaging the old normalized capped8-corner full-GT frame score over seeds; ties prefer smaller lambda then stricter cap. Calibration is only a diagnostic probe for D; there is no fake temperature sweep. The1985-frame synthetic heldout is evaluated after selection. These are historically reused synthetic datasets, not universally untouched data.

Development: 319 positives in13 sessions,2689 negatives. Precision has311 matched frames and2756 supervised points; all-GT PCK has2818 points. No keypoint confidence threshold alters the existing supervision contract. Negative session metadata is unavailable (0 populated records). Actual D negative inference checks candidate preservation, while negative points are not supervised localization endpoints.

Pose coverage follows the canonical pose evaluator, not the2D IoU matching gate: it uses the top prediction with sufficient finite corners and successful selector/solvers. Thus319 available poses do not imply319 correctly matched2D detections. Missing/nonfinite/unsolved pose IDs and matched2D denominators are reported separately. The primary conditional residual cannot replace complete-failure-aware PCK.

Full-precision predictions and canonical targets are used for residuals. CSV rounding differences are separately checked. R0 is a single baseline in each shared bootstrap draw; each seed's pooled statistic is computed before averaging. Session clusters are the primary uncertainty description and frame bootstrap is secondary.10,000 draws,seed20260914. Session/seed count does not guarantee out-of-session performance. No noninferiority margin or operational tolerance was invented.

Formal prior comparison and independent confirmation remain unfinished. No present DEV result is renamed as final test evidence. See NUMBER_SOURCES.json for every numerical table binding.
