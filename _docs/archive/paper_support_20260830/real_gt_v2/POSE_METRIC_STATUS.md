# Paper pose metric status

Status: **BLOCKED**

## Multi-shape status

```
(1/2)
Object subset   Migration/geometry                                                     Symmetry
───────────────────────────────────────────────────────────────────────────────────────────────
PLASTIC         PASS                                                                   frozen {I,Ry(180°)}
WOOD            45/45 materialized; mechanical checks PASS; signed-axis gate BLOCKED   UNREVIEWED
ALL             both required                                                          every object required

(2/2)
Object subset   Selector                Pose fields
───────────────────────────────────────────────────
PLASTIC         FAIL                    null; BLOCKED_SELECTOR
WOOD            NOT_RUN                 null; BLOCKED_MIGRATION_SYMMETRY_AND_SELECTOR
ALL             every object required   null; BLOCKED_BY_CONSTITUENT_OBJECT
```

The plastic contract below remains byte-for-byte conceptually separate from
wood. Wood uses its own geometry and
[`WOOD_SYMMETRY_SPEC.md`](WOOD_SYMMETRY_SPEC.md); it never inherits the plastic
symmetry merely to fill a metric. An ALL pose value cannot be computed from the
plastic subset while wood is blocked.

## Plastic gate detail

Paper pose fields are enabled only when all gates below are true:

```
Gate                               Current status
─────────────────────────────────────────────────
canonical migration                PASS; 140 exact yaw-180 equivalence classes, no fabricated signed pose
GT-independent W/D selector        actual DEV140 diagnostic FAIL: 83/140 overall, 13/28 NIGHT, minimum session 4/12, all four worst tails 13/14 selector failures
benchmark symmetry specification   FROZEN; restricted ADD-S over {I, Ry(180°)}
untouched FINAL manifests          membership unavailable
```

Therefore Restricted ADD-S AUC, rotation median, translation median, and yaw median must be
serialized as `null` with a non-empty `blocked_reason`. Zero and NaN are not
substitutes for a blocked measurement. Box AP50:95 and 2D keypoint diagnostics are
separate and may be computed on an explicitly named DEV population.

Historical ALL161 pose values used per-frame GT dimensions and cannot be copied
into the v2 paper table.

The evaluator was dry-run against both roles and was then executed on the DEV
pair `COMMON_DEV_POS128 + DEV_NEG2689` using the selector-bound checkpoint.
FINAL returned `FINAL_MEMBERSHIP_UNAVAILABLE`. The actual DEV result keeps all
four pose values `null` with the two remaining reasons:

```text
POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR
FINAL_MANIFEST_NOT_FROZEN
```

Evidence: `challenge/evaluation_v2/selector_diagnostic/PLASTIC_SELECTOR_DIAGNOSTIC.json`,
`challenge/evaluation_v2/selector_diagnostic/CAMERA_ONLY_REPLAY_EQUIVALENCE.json`,
and `challenge/evaluation_v2/dev_results/YOLO26_G38_DEV.json`. The two JSON files
under `challenge/evaluation_v2/dry_runs/` are pre-diagnostic contract snapshots,
not current selector evidence.
