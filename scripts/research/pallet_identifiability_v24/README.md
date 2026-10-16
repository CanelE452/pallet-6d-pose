# V24 learnability and direction-information diagnosis

Frozen YOLO/V10/V11/V12/V18; new gates initialized from the same seed, never from V23 retention checkpoints. Existing V23 observations and original manual GT evaluation are reused.

Use `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python` and run these scripts from the repository:

```text
register24.py
train24.py --phase sign
train24.py --phase tiny
eval24.py --phase inference
eval24.py --phase score
classify24.py
tiny_audit24.py
audit24.py
network24.py
visual24.py
report24.py
delivery24.py
```

CUDA is required for training, inference, tiny audit and integration. Completed training protocols are protected from overwrite; a fresh reproduction needs an isolated result context. Reports incorporate a manually written result-specific `INTERPRETATION_KO.md`, which must be revised for new results.

Tiny: 12 synthetic originals, six image conditions, paired line/local gates, 3 seeds, 6,000 updates; 48 other originals for tiny-only generalization diagnosis. Free per-point logits are a privileged positive control, not a deployable model. Sign: 3 arms × 3 seeds × 2,000 updates; same expanded V23 batch plans. Total 15 neural models / 54,000 neural updates plus 6,000 privileged lookup updates.

The positive direction label identifies infinitesimal squared-error improvement, not a safe full step. Applying its score to the unchanged vector is a separate exploratory geometric test. Synthetic class balancing means these scores should not be interpreted as calibrated probabilities. No real fitting, thresholds, self-training, inference filtering or operating-model replacement.

All real/synthetic evaluation sets were seen in earlier work. Tiny check originals are excluded only from tiny gate fitting, not from the full sign experiment or historical V23 fitting. Parent-control removes current feature map and64D context, but retains image-derived parent geometry/probabilities/line coordinates/content mask. Failure of a probe does not prove absence of information or impossibility of fusion.
