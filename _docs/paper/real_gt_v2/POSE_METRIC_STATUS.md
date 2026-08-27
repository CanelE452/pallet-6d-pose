# Paper pose metric status

Status: **BLOCKED**

Paper pose fields are enabled only when all gates below are true:

| Gate | Current status |
|---|---|
| canonical migration | `PASS`; 140 exact yaw-180 equivalence classes, no fabricated signed pose |
| GT-independent W/D selector | implementation complete; DEV140 model-output diagnostic `NOT_RUN` |
| benchmark symmetry specification | `FROZEN`; restricted ADD-S over `{I, Ry(180°)}` |
| untouched FINAL manifests | membership unavailable |

Therefore ADD(-S) AUC, rotation median, translation median, and yaw median must be
serialized as `null` with a non-empty `blocked_reason`. Zero and NaN are not
substitutes for a blocked measurement. Box AP50:95 and 2D keypoint diagnostics are
separate and may be computed on an explicitly named DEV population.

Historical ALL161 pose values used per-frame GT dimensions and cannot be copied
into the v2 paper table.

The evaluator was dry-run against both allowed population roles. The DEV pair
`COMMON_DEV_POS128 + DEV_NEG2689` passed its membership contract; FINAL returned
`FINAL_MEMBERSHIP_UNAVAILABLE`. In both reports all four pose values are `null`
with the two remaining reasons:

```text
POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR
FINAL_MANIFEST_NOT_FROZEN
```

Evidence: `challenge/evaluation_v2/dry_runs/DEV_CONTRACT_DRY_RUN.json` and
`FINAL_CONTRACT_DRY_RUN.json`.
