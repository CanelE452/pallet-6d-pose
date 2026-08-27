# Paper real evaluation v2

The paper evaluator uses explicit repo-local manifests and GT v2 fields only.
Historical evaluators are unchanged.

Modules:

```text
real_dataset_contract.py  membership/count/hash/pair validation
pnp_selector.py           prediction-only short/long-face PnP selector
pose_metrics.py           four-gate canonical pose metrics
paper_real_eval.py        CLI, 2D evaluation, dry-run contract report
```

Allowed DEV paper comparison pair:

```text
COMMON_DEV_POS128 + DEV_NEG2689
```

`DEV_POS140` is diagnostic-only. FINAL manifests currently have zero members
because untouched membership is unavailable; they are not a valid empty test.

Example contract dry-run:

```bash
python challenge/evaluation_v2/paper_real_eval.py \
  --positive-manifest challenge/real_gt_v2/manifests/COMMON_DEV_POS128.json \
  --negative-manifest challenge/real_gt_v2/manifests/DEV_NEG2689.json \
  --population-role DEV \
  --weights /path/to/model.pt \
  --migration-gate challenge/real_gt_v2/MIGRATION_GATE.json \
  --symmetry-contract challenge/real_gt_v2/SYMMETRY_CONTRACT.json \
  --out /new/output/report.json \
  --dry-run
```

Dry-run performs no model import or inference. Canonical migration and symmetry
now pass through the exact yaw-180 equivalence-class contract. Pose fields remain
structured nulls because the DEV140 W/D-parity selector diagnostic has not run
and untouched FINAL membership has not been frozen.
