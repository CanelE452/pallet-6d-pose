# Paper real evaluation v2

The paper evaluator uses explicit repository manifests and GT-v2 fields only.
Historical evaluators remain unchanged. Geometry is selected exclusively by

```text
item.object_type -> OBJECT_GEOMETRY_REGISTRY.json -> named X/Y/Z dimensions
```

It never selects dimensions from a frame name, session, legacy `dimensions_m`,
or GT pose. A label's v2 dimensions are checked against the registry but are
not used as the dispatch source.

Modules:

```text
real_dataset_contract.py  membership/count/hash/role/pair validation
pnp_selector.py           prediction-only object-specific W/D PnP hypotheses
pose_metrics.py           per-object metrics and fail-closed ALL aggregation
paper_real_eval.py        registry dispatch, 2D subgroups, pose gates, CLI
```

## Populations and allowed DEV pairs

```text
PLASTIC  COMMON_DEV_PLASTIC_POS128 + DEV_NEG2689  (role DEV)
WOOD     DEV_WOOD_POS45            + DEV_NEG2689  (role CROSS_SHAPE_DEV)
ALL      COMMON_DEV_MULTISHAPE_POS  + DEV_NEG2689  (role DEV)
```

Legacy `DEV_POS140` and `COMMON_DEV_POS128` remain registered unchanged;
`DEV_PLASTIC_POS140` and `COMMON_DEV_PLASTIC_POS128` are object-explicit aliases
with identical ordered memberships. Wood45 was previously evaluated, so it is
development-only. Every FINAL positive/negative placeholder is unavailable,
not a valid empty test.

All inference reports contain `ALL`, `PLASTIC`, and `WOOD` 2D/pose rows. Box AP
uses the shared pallet class and common negative population. The additive
`ALL_ANNOTATED_UNKNOWN_VISIBILITY` keypoint diagnostic may use migrated points,
but it is not a visible/occluded subgroup claim. If any required object pose
gate is blocked, every ALL pose field is JSON `null`; a passed object subgroup
may never be relabeled as ALL.

## Current gate state

```text
PLASTIC migration PASS; symmetry FROZEN; selector FAIL (83/140, NIGHT 13/28)
WOOD    mechanical migration PASS; signed pose/symmetry UNREVIEWED; selector NOT_RUN
ALL     Restricted ADD-S / rotation / translation / yaw BLOCKED and null
```

Formal selector evidence is in `selector_diagnostic/PLASTIC_*`; wood's explicit
`NOT_RUN` record is `selector_diagnostic/WOOD_SELECTOR_STATUS.json`. Thresholds
were not changed.

## Multi-shape contract dry-run

```bash
python challenge/evaluation_v2/paper_real_eval.py \
  --positive-manifest challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json \
  --negative-manifest challenge/real_gt_v2/manifests/DEV_NEG2689.json \
  --population-role DEV \
  --weights /path/to/model.pt \
  --migration-gate challenge/real_gt_v2/MIGRATION_GATE.json \
  --selector-diagnostic challenge/evaluation_v2/selector_diagnostic/PLASTIC_SELECTOR_DIAGNOSTIC.json \
  --symmetry-contract challenge/real_gt_v2/SYMMETRY_CONTRACT.json \
  --object-migration-gate wood_small_80x59x14=challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json \
  --out /new/output/report.json \
  --dry-run
```

The latest checked contract result
`dev_results/YOLO26_G38_MULTISHAPE_DEV_CONTRACT_V3.json` proves the 173-positive
dispatch, manifest-level plastic/wood object scope, exact object-bound selector
populations (`DEV_POS140` and wood `25+20`), and null pose aggregation without
importing a model. The earlier dry-runs are preserved as generated history.
Dry-run is not a Box/2D performance result.
