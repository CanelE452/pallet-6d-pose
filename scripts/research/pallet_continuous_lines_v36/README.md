# V36 continuous Hough lines and point fusion

Requires completed immutable V32/V34/V35 artifacts and the GPU environment:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_continuous_lines_v36/run36.py
```

The driver resumes missing stages in this run. Use a new protocol/result directory
for a fresh experiment; never overwrite completed prior results.

Primary: four arms (grid / refine / rank / both), three seeds, 2000 updates each.
Same fixed V34 image/query features, diverse eight candidates per role, same point
head initialization and V32 batch plans. Train bounded continuous theta/rho and
quality with physical GT; point gradients are stopped at the line predictor.
Line appearance stays cached; refined coordinates do not resample image features.

Post-primary: final inference resets/donors and 20 fixed-batch gradient probes per
seed. Since geometry influence stayed small with frozen line encoders, three arms
(grid / refine / both) additionally adapt line token/relation encoders and point
heads from identical V34 initial states, with trained line predictors frozen.
Nine additional models / 18k updates. Total: 21 models / 42,000 optimizer updates.
Primary and exploratory protocols and outcomes remain separate.

Scripts: cache36, train36, eval36, lines36, probes36, adapt36, audit36, report36,
deliver36. Each accepts the stage arguments documented in run36.py. Final reports
include per-seed accuracy, source-group/session CIs, geometry counterfactuals,
actual synthetic/real images, input/GT/weight audits and visible Chrome delivery.

The render check is reused and shares the same two CAD assets with training.
Real297 is the existing user-authorized reused DEV lineage, distinct from
canonical140/paper/sealed evaluation. No real GT fitting or manual annotation edits.
Best-of8 and GT role association describe candidate geometry, not deployment.
Whole-training repeated-run equality is not claimed for all21runs; actual fixed-batch
forward/backward parity and V34's corrected sampling path were verified.

Completion Discord notification is sent separately after final artifact checks.
