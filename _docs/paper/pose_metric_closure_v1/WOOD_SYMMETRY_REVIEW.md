# Wood symmetry review

```text
W1 = FAIL
```

But not for the reason the blocker name suggests. The geometry supports a symmetry
set. What blocks it is a repository policy conflict.

## What the measurements support

Proposed equivalence set:

```text
accepted proper rotations   { I ,  Ry(180 deg) = diag(-1, +1, -1) }
excluded                    Ry(90 deg), Ry(270 deg)
```

Evidence, all measured:

```text
item                                          value                        n
─────────────────────────────────────────────────────────────────────────────
footprint                                     0.80 m x 0.59 m, rectangular   —
  -> 90 and 270 are not symmetries at all
cuboid point set invariance under Ry(180)     canonical<->CF roundtrip
                                              2.3e-13 px                    125
deck slat pattern, mirror NCC on the
  0.80 m axis profile                         0.957                          45
slat centre mirror residual                   <= 6 mm over 0.80 m (0.8%)      45
fork-entry axis under Ry(180)                 entry face maps to the
                                              opposite entry face of the
                                              same axis; axis preserved       —
```

The last row is the one that is **not** fully verified. The opposite face is never
observed in any frame — the same gap that leaves the canonical sign undetermined.
One block also carries an IPPC stamp, so the object is visually asymmetric under
180 degrees even though it is geometrically symmetric.

Consequence for honesty: if wood were frozen, it would have to be frozen at the same
evidentiary grade as plastic —

```text
kind                          DECLARED_BENCHMARK_ASSUMPTION
physical_inspection_claimed   false
```

Writing "verified by physical inspection" would be false. Four photographs of the
real pallet's four faces would remove this caveat entirely.

## What actually blocks it

```text
challenge/evaluation_v2/paper_real_eval.py:1358
    if spec.symmetry_status != "FROZEN" and symmetry_path is not None:
        raise ContractError("UNREVIEWED_OBJECT_CANNOT_ACCEPT_SYMMETRY_CONTRACT")

scripts/annotate/object_geometry_registry.py:168
    enforces  (status == "FROZEN") != (contract is None)
```

So `symmetry_status` and `symmetry_contract` must change together, and both live in
`challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json`.

```text
registry sha256   0c7a1072...f0627
pinned in         62 files, 198 places
```

That hash is a load-bearing part of the paper track's provenance. Changing it is not
a code edit; it is a re-issue that invalidates every lock that pins it.

**This is a user decision, not an engineering one.** Two options:

```text
A   re-issue the registry with wood frozen, then update all 198 pins in one
    controlled pass, and record the re-issue in the provenance chain

B   exclude wood from the pose table. Report wood for detection and 2D keypoints
    only, which is what the current paper already does.
```

Option B costs nothing and is consistent with the existing paper. Option A is only
worth it if wood pose is actually wanted in the table, and that depends on the wood
selector, which has never been run.

## The symmetry schema is more constrained than it looks

```text
scripts/annotate/pallet_symmetry.py:39-43
    EQUIVALENT_YAW_DEGREES = (0, 180)
    METRIC_VARIANT         = "ADD-S"
    CANONICAL_AXIS         = "+Y"        <- module constants, hardcoded
```

No symmetry set other than {0, 180} about +Y can be expressed. So the question is
binary: freeze with {I, Ry(180)}, or do not freeze. There is no third option to
design.

A draft contract JSON was written to scratchpad and passed `load_symmetry_contract()`
schema validation, confirming the mechanism works. It was **not** added to the
repository.

## Why the 90-degree exclusion matters more than it appears

The unit tests make this concrete:

```text
test_unrestricted_adds_forgives_the_square_swap
    unrestricted ADD-S returns exactly 0.0 for a 90-degree swap on a square
    footprint
```

If 90 degrees were admitted into the symmetry set out of convention, the selector
failure would vanish from the metric — the evaluator would score a wrongly-oriented
pallet as perfect. For a forklift that is the difference between entering the
pockets and hitting the deck. The exclusion is not pedantry.
