# Real pallet GT v2 artifacts

This namespace is additive. The reviewed legacy 140 labels remain untouched.

```text
audit/                    Phase-A source and consumer audit
manifests/                explicit DEV/COMMON/NEG/FINAL memberships
migrated_gt/              140 additive v2 JSON copies
INVALID_GT_QUARANTINE.json confirmed-invalid source-label denylist (23 exact identities)
SYMMETRY_CONTRACT.json     frozen yaw-180 benchmark equivalence
MIGRATION_REPORT.csv      per-frame conversion checks
MIGRATION_GATE.json       aggregate machine-readable gate
MANUAL_REVIEW_QUEUE.csv   signed physical-axis review queue
VISIBILITY_REVIEW_QUEUE.csv point visibility/provenance review queue
```

`challenge/real_gt_v2/migrated_gt/` is the single authoritative v2 label tree.
The dataset-facing path `challenge/data/01_real/gt_v2_canonical` is a tracked
relative symlink to it; `.gitignore` contains an exact exception for that link.
Do not create a second copy of the JSON tree.

The aggregate migration status is `PASS` in
`YAW_180_EQUIVALENCE_CLASS` mode. Every migrated label still keeps
`canonical_pose=null` and both signed candidates; the gate proves that the pair
is exactly related by the frozen 180-degree yaw symmetry. No candidate is
promoted by reprojection score or presented as signed physical GT.

Run the migration from the repository root:

```bash
python scripts/annotate/migrate_real_gt_v2.py
```

The CLI validates `SYMMETRY_CONTRACT.json` by default. The library API defaults
to no symmetry contract and therefore remains blocked unless one is supplied
explicitly. The default `--existing-output-policy skip-identical` never rewrites
an existing JSON. An exact deterministic migration result is skipped byte-for-byte; a
different file (including a human-reviewed label) fails closed with
`EXISTING_OUTPUT_PROTECTED`. Use `--existing-output-policy error` when even an
identical pre-existing output should abort. There is intentionally no force or
overwrite mode. Point visibility review remains separate and currently has 140
queued frames.

Confirmed-invalid legacy source labels are handled separately from migration.
On 2026-08-27, 21 invalid eval labels and two stale duplicate labels were moved
to the recoverable local archive `_archive/real_gt_invalid_20260827/`; source
images were retained. The two stale duplicates share frame IDs with good
canonical labels, so they are denied by exact path/SHA rather than frame ID.
See `_docs/paper/real_gt_v2/GT_QA_STATUS.md` for the clean 140 population and
clickable image index.
