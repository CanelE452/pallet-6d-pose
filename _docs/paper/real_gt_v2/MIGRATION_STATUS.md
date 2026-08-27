# Real GT v2 migration status

Status: **PASS — YAW-180 CANONICAL POSE EQUIVALENCE CLASS**

GT-QA note (2026-08-27): 23 confirmed-invalid legacy JSON files are in the
recoverable quarantine; 21 were members of the raw 161 eval scan and two were
stale duplicate paths. The reviewed 140 migration sources have zero forbidden
SHA overlap, so no migrated v2 label was deleted or rewritten. See
`GT_QA_STATUS.md` and `challenge/real_gt_v2/INVALID_GT_QUARANTINE.json`.

The migration is intentionally non-destructive. It wrote 140 new JSON files
under `challenge/real_gt_v2/migrated_gt/` (the authoritative v2 tree) while
preserving the legacy directory structure. It verified all source-label SHA-256
values, sizes, and mtimes against the Phase A baseline before and after
conversion.

`challenge/data/01_real/gt_v2_canonical` is a relative symlink to that directory,
and `.gitignore` explicitly unignores that one link. The dataset-facing path is
therefore reproducible from Git while both names resolve to one set of files
rather than two drifting mutable copies.

Migration reruns are fail-closed. The default `skip-identical` policy leaves an
exact existing mechanical output untouched and records `SKIPPED_IDENTICAL`. If
an existing output differs—especially after human visibility or signed-axis
review—the run blocks with `EXISTING_OUTPUT_PROTECTED` and does not overwrite any
v2 JSON. `--existing-output-policy error` rejects even identical files; no force
overwrite option exists.

## Verified result

```text
source labels                         140
migrated v2 copies                    140
schema-valid copies                   140
fixed physical dimensions             140
source SHA/size/mtime changes            0
legacy-field mismatches                  0
reflection transforms                   0
resolved singular canonical poses        0
resolved yaw-180 pose classes           140
axis manual-review queue                  0
visibility manual-review queue          140
last rerun identical outputs skipped    140
last rerun v2 JSON state changes          0
```

Candidate rotation maxima:

```text
R orthogonality error       3.3306690738754696e-16  (limit 1e-6)
abs(det(R)-1)               3.3306690738754696e-16  (limit 1e-6)
projection parity error px  1.1368683772161603e-13  (limit 1e-4)
```

The 81 short-face-front labels retain `YAW_0/YAW_180`; the 59 long-face-front
labels retain `YAW_90/YAW_270`. All 1,260 migrated keypoint visibility values are
zero because legacy point-level provenance is unavailable.

Legacy dimensions recover yaw parity, not signed physical direction. The frozen
benchmark contract defines the two signs in each parity pair as one pose class.
All 140 labels were revalidated as exact `R2 = R1 · Ry(180°)` classes with
identical translation. No label was rewritten and no signed direction was
fabricated: every file still carries `canonical_pose=null` and both candidates.
The aggregate gate is now:

```text
PASS: YAW_180_EQUIVALENCE_CLASS
```

The PASS is bound to the exact path and file-byte SHA-256 of
`challenge/real_gt_v2/SYMMETRY_CONTRACT.json`. `MANUAL_REVIEW_QUEUE.csv` is
empty; `VISIBILITY_REVIEW_QUEUE.csv` still contains 140 frames because point
visibility/provenance is an independent review task. Generated evidence is in
`MIGRATION_GATE.json`, `MIGRATION_REPORT.csv`, and the two review queues.
