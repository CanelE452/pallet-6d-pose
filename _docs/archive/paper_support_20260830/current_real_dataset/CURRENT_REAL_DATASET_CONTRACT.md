# Current Real Dataset Contract

## Canonical populations

```text
DEV_PLASTIC_POS140          140   alias-compatible with DEV_POS140
COMMON_DEV_PLASTIC_POS128   128   alias-compatible with COMMON_DEV_POS128
DEV_WOOD_POS45               45   25 wood_183705 + 20 wood_184309
COMMON_DEV_MULTISHAPE_POS   173   exact union of plastic 128 and wood 45
DEV_NEG2689                2689   unchanged common negative DEV
```

Every composite-positive row records frame ID, object type, session ID, image
path, GT-v2 path, population role, and source population. Session-qualified wood
IDs are mandatory because bare six-digit wood stems collide with negative stems.

## Object dispatch contract

```text
plastic_standard_110x130x11 -> (X,Y,Z) = (1.10,0.11,1.30) m
wood_small_80x59x14         -> (X,Y,Z) = (0.80,0.14,0.59) m
```

The positive manifest supplies `object_type`; the geometry registry supplies
physical dimensions. Legacy W/D, GT pose/parity, filename, and session name are
forbidden geometry selectors. Unknown types fail closed.

Plastic symmetry is frozen `{I,Ry(180°)}`. Wood symmetry is independently
`UNREVIEWED`; it does not inherit the plastic set. Plastic selector is FAIL and
wood selector is NOT_RUN. Therefore ALL Restricted ADD-S/rotation/translation/
yaw fields are null.

## Audited membership facts

- Existing plastic 140/128 and negative 2,689 membership is unchanged.
- Wood images/labels are present 45/45 with exact internal duplicates 0.
- Wood exact image/decoded-image overlap with plastic DEV and negatives is 0.
- Wood GT QA is 45 GREEN / 0 AMBER / 0 RED; no frame was silently removed.
- Wood is 1280×720 with a common `SENSOR_PROFILE_SCALED` intrinsic profile.
- Wood45 was used by historical Stage-B/wood diagnostics and is permanently DEV.

## Evaluation use

```text
plastic controlled DEV   COMMON_DEV_PLASTIC_POS128 + DEV_NEG2689
wood controlled DEV      DEV_WOOD_POS45 + DEV_NEG2689
all controlled DEV       COMMON_DEV_MULTISHAPE_POS + DEV_NEG2689
plastic selector DEV     DEV_PLASTIC_POS140
```

Box/detection and declared 2D diagnostics may run while pose is blocked. A
partial passing subgroup is never reported as ALL.

## FINAL

All four FINAL populations are `UNAVAILABLE`, with null membership rather than
a valid zero count. New plastic and wood captures are required; DEV membership
is not promoted or copied.

## Legacy plastic provenance

The raw plastic scan contained 161 labels; 21 invalid eval labels were excluded
to obtain 140 and 12 FT-overlap frames were excluded to obtain the common 128.
Legacy labels contained 81/59 camera-facing W/D variants and no per-keypoint
visibility. Those facts remain provenance, not current evaluator inputs.
