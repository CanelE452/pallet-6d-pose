# Reproduce the action-learning branch

Run from the repository root with the existing Python environment containing
NumPy1.26.4 and PyTorch2.1.1. The private synthetic RGB/features, original renderer
references and old checkpoints are required; they are not redistributed here.
These commands create fresh output directories. They do not replace operating
defaults or old artifacts.

Set paths to your actual authorized cache locations:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P_PY=python
P_OLD=/path/to/original/statewise-end-to-end-recovery-v1
P_RUN=/path/to/new/boundary-action-repair-run
```

The original inputs are the1003-row signed-log feature file
`synthetic_features_v1/HEAD_FEATURES.jsonl.gz`, renderer target file
`synthetic_v1/SYNTH_GT_LABELS.jsonl.gz`, original768/235 split lock and three
`head_training_v1/L_HEAD_seed20261018...20/best.pt` checkpoints. The actual
hashes and frozen source are in the learning plan and source bindings. D0/H0
checkpoint reuse is zero new training runs:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.train_actions reuse-legacy \
  --legacy-dir "$P_OLD/head_training_v1" \
  --output "$P_RUN/action_training_v1"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.action_data \
  --features "$P_OLD/synthetic_features_v1/HEAD_FEATURES.jsonl.gz" \
  --feature-seal "$P_OLD/synthetic_features_v1/FEATURE_SEAL.json" \
  --targets "$P_OLD/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz" \
  --target-seal "$P_OLD/synthetic_v1/PROVISION_SEAL.json" \
  --split-lock "$P_OLD/synthetic_v1/DATA_SPLIT_LOCK.json" \
  --output "$P_RUN/d0_action_labels_v1" --domain D0

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.train_actions train \
  --features "$P_OLD/synthetic_features_v1/HEAD_FEATURES.jsonl.gz" \
  --feature-seal "$P_OLD/synthetic_features_v1/FEATURE_SEAL.json" \
  --labels-dir "$P_RUN/d0_action_labels_v1" \
  --output "$P_RUN/action_training_v1" --arm D0_H1_MLP --training-GO
```

The last command runs all three fixed seeds. Every optimizer attempt is reserved
in the global training ledger; do not create a new ledger to bypass its18-run cap.
Technical failures and their folders must be preserved and reviewed before a
retry. The current CLI deliberately refuses to silently overwrite an attempt.

After the separately documented D1 producer has actually completed both domains,
and its immutable execution seals certify independently executed E0, prepare D1
labels. Substitute the actual producer output directories, recorded in the final
input bindings:

```bash
P_D1_SYNTH="$P_RUN/d1_synthetic_candidates_v1"
P_D1_REAL="$P_RUN/d1_real_candidates_v1"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.action_data \
  --features "$P_D1_SYNTH/HEAD_FEATURES.jsonl.gz" \
  --feature-seal "$P_D1_SYNTH/FEATURE_SEAL.json" \
  --targets "$P_OLD/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz" \
  --target-seal "$P_OLD/synthetic_v1/PROVISION_SEAL.json" \
  --split-lock "$P_OLD/synthetic_v1/DATA_SPLIT_LOCK.json" \
  --output "$P_RUN/d1_action_labels_v1" --domain D1

for P_ARM in D1_H0_MLP D1_H1_MLP D1_H1_ROLE_AFFINE D1_H1_PAIR_MLP D1_H1_PAIR_ROLE_AFFINE
do
  "$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.train_actions train \
    --features "$P_D1_SYNTH/HEAD_FEATURES.jsonl.gz" \
    --feature-seal "$P_D1_SYNTH/FEATURE_SEAL.json" \
    --labels-dir "$P_RUN/d1_action_labels_v1" \
    --output "$P_RUN/action_training_v1" --arm "$P_ARM" --training-GO
done
```

This is15 additional fits, not15 independent datasets. Run only when the declared
worker budget has a free CPU slot. Use an external process timer for shared input
loading/preparation as well as the per-run timers. The trained manifest remains
incomplete until all21 reused/new checkpoints are available; blocked groups must
be reported as such rather than counted as zero successes.

All21 selectors use the same real feature bytes. The metadata join is the old
reference-free canonical input238 file; prediction does not load real references:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.predict_actions \
  --features "$P_D1_REAL/HEAD_FEATURES.jsonl.gz" \
  --feature-seal "$P_D1_REAL/FEATURE_SEAL.json" \
  --training-dir "$P_RUN/action_training_v1" \
  --metadata "$P_OLD/inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz" \
  --output "$P_RUN/action_predictions_v1"
```

Run unit contracts and the independent action/split audit. The auditor does not
import production action construction, model-input construction, loss, labels or
model classes. Its actual synthetic labels use original per-row C1/C2 renderer
references; it never reads real/CAL/PRIMARY references. It independently builds
the188 input whitelist and applies saved matrices to verify real logits and
actual output-source membership.

```bash
"$P_PY" -B -m unittest scripts.research.pallet_boundary_action_repair_v1.test_action_contract -v

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.independent_action_audit \
  --targets "$P_OLD/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz" \
  --target-seal "$P_OLD/synthetic_v1/PROVISION_SEAL.json" \
  --split-lock "$P_OLD/synthetic_v1/DATA_SPLIT_LOCK.json" \
  --d0-features "$P_OLD/synthetic_features_v1/HEAD_FEATURES.jsonl.gz" \
  --d0-feature-seal "$P_OLD/synthetic_features_v1/FEATURE_SEAL.json" \
  --d0-labels-dir "$P_RUN/d0_action_labels_v1" \
  --d1-features "$P_D1_SYNTH/HEAD_FEATURES.jsonl.gz" \
  --d1-feature-seal "$P_D1_SYNTH/FEATURE_SEAL.json" \
  --d1-labels-dir "$P_RUN/d1_action_labels_v1" \
  --training-dir "$P_RUN/action_training_v1" \
  --common-real-features "$P_D1_REAL/HEAD_FEATURES.jsonl.gz" \
  --common-real-seal "$P_D1_REAL/FEATURE_SEAL.json" \
  --predictions-dir "$P_RUN/action_predictions_v1" \
  --output "$P_RUN/independent_action_full_v1"
```

The initial D0-only independent audit's absolute1e-8cm comparison failed34
candidate/action records because their preexisting extreme translation errors
have float64 spacings larger than that absolute bound. That failure is retained.
The amended audit keeps the bound and enumerates the same stable-fsum
representation exceptions (maximum2ULP) while requiring exact reproduction of
every saved translation using the original dot-product expression. No label,
training target, success threshold or old extreme pose was changed.
