# Multi-shape FINAL test requirements

```text
FINAL_PLASTIC_POS = UNAVAILABLE
FINAL_WOOD_POS    = UNAVAILABLE
FINAL_ALL_POS     = UNAVAILABLE
FINAL_NEG         = UNAVAILABLE
```

Current plastic 140/128, wood45, and negative DEV were used for development or
diagnosis. None is untouched FINAL. In particular, the two existing wood
sessions were previously evaluated and cannot be promoted.

## Required design

- new independent plastic and wood sessions, at least six total and multiple
  sessions per object;
- approximately 300 positives as a planning target, illustratively plastic ~200
  and wood ~100, without a token wood subgroup;
- at least 1,000 negatives, preferably 1,500–2,000;
- about 80–100 NIGHT positives across actually captured conditions;
- per-frame object type, session, timestamp, camera serial, native intrinsics,
  intrinsics quality, lighting, physical pallet ID, point visibility/reason,
  occlusion, and truncation;
- exact geometry-registry and object-specific symmetry-contract hashes.

Negative frames must be audited as absent of both registered pallet types.

## Freeze gate

Use the exact ten-step order in
[`../final_dataset_capture/FREEZE_CHECKLIST.md`](../final_dataset_capture/FREEZE_CHECKLIST.md):
capture, blind annotation, QA reasons, training/DEV overlap audit, exact
duplicates, perceptual duplicates, coverage, four-population membership freeze,
contract/checkpoint freeze, then one-shot evaluation.

No method result is inspected before membership freeze. A material later fault
creates a versioned replacement rather than a silent membership edit.
