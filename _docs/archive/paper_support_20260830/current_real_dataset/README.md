# Current real pallet pose dataset

Status date: 2026-08-28. This is a two-object **development** dataset, not an
untouched final test.

## Population summary

```
(1/2)
Population                                      Object    Count   Sessions/source sets
──────────────────────────────────────────────────────────────────────────────────────
DEV_PLASTIC_POS140 (DEV_POS140)                 plastic     140   7
COMMON_DEV_PLASTIC_POS128 (COMMON_DEV_POS128)   plastic     128   provenance-limited
DEV_WOOD_POS45                                  wood         45   2
COMMON_DEV_MULTISHAPE_POS                       both        173   composite
DEV_NEG2689                                     none      2,689   incomplete session metadata

(2/2)
Population                                      Role
────────────────────────────────────────────────────
DEV_PLASTIC_POS140 (DEV_POS140)                 migration/selector DEV
COMMON_DEV_PLASTIC_POS128 (COMMON_DEV_POS128)   controlled DEV
DEV_WOOD_POS45                                  CROSS_SHAPE_DEV
COMMON_DEV_MULTISHAPE_POS                       controlled multishape DEV
DEV_NEG2689                                     pallet-absent DEV
```

The new plastic names do not change existing 140/128 memberships. Wood consists
of `wood_183705` 25 + `wood_184309` 20. All 45 wood frames were previously
evaluated, so “Development cross-shape evaluation” is the strongest permitted
role description.

## Object geometry and pose state

```
(1/2)
Object    Registered (X,Y,Z) m   Intrinsics/resolution             Symmetry                Selector
───────────────────────────────────────────────────────────────────────────────────────────────────
Plastic   (1.10,0.11,1.30)       640×480 legacy setup              frozen {I,Ry(180°)}     FAIL
Wood      (0.80,0.14,0.59)       1280×720, SENSOR_PROFILE_SCALED   UNREVIEWED              NOT_RUN
ALL       object-specific        mixed                             every object required   every object required

(2/2)
Object    Pose status
─────────────────────
Plastic   blocked
Wood      blocked
ALL       null
```

Legacy camera-facing W/D tuples remain provenance only. The evaluator resolves
geometry through manifest `object_type` and the object registry. Plastic and
wood dimensions/symmetry are never interchanged.

Plastic selector diagnostic: 83/140 overall, 13/28 NIGHT, minimum session 4/12,
with 13/14 selector failures in every frozen pose tail. It is not retuned on
DEV. Wood integration does not change that result.

## Data quality

- Plastic migration: 140/140 PASS, source SHA/size/mtime mutation 0.
- Wood membership/GT QA: 45/45 GREEN, exact image/hash overlap with plastic DEV
  and DEV negatives 0, source JSON modified 0.
- Wood intrinsics: one RealSense sensor profile scaled to 1280×720, not
  per-session calibration.
- Plastic visibility: 140/140 frames unreviewed.
- Wood visibility: 405/405 point states unknown and queued.
- Annotation reliability: `PREPARED_NOT_MEASURED`; active plan is 28 plastic +
  12 wood.

## FINAL status

```text
FINAL_PLASTIC_POS = UNAVAILABLE
FINAL_WOOD_POS    = UNAVAILABLE
FINAL_ALL_POS     = UNAVAILABLE
FINAL_NEG         = UNAVAILABLE
```

Existing DEV data are never copied to FINAL. See
[`../final_dataset_capture/FINAL_CAPTURE_PROTOCOL.md`](../final_dataset_capture/FINAL_CAPTURE_PROTOCOL.md).

## Paper metrics

Main metrics remain Box AP50:95, Restricted ADD-S AUC, symmetry-aware rotation
median, translation median, and symmetry-aware yaw median. Report
ALL/PLASTIC/WOOD. `5cm5deg` and `10cm10deg` are historical/diagnostic only.
