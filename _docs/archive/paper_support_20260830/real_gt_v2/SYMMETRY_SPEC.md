# Plastic-standard pallet symmetry specification

Status: **FROZEN**

Scope: **`plastic_standard_110x130x11` only**. Wood does not inherit this
contract; its current status and review boundary are defined separately in
[`WOOD_SYMMETRY_SPEC.md`](WOOD_SYMMETRY_SPEC.md).

Machine-readable contract:
`challenge/real_gt_v2/SYMMETRY_CONTRACT.json`

## Decision and evidence boundary

For the registered plastic-standard object, poses that differ only by a 180-degree
rotation about canonical `+Y` are defined as equivalent. This is a benchmark
assumption fixed by the evaluation-contract owner on 2026-08-27. It is not a
claim that every physical pallet instance was independently inspected for
visual or operational symmetry, and the contract explicitly records
`physical_inspection_claimed=false`.

The rule was fixed without using real GT v2 DEV or FINAL pose results. Existing
blocked dry runs contain no Restricted ADD-S, rotation, translation, or yaw values.

## Accepted proper rotations

The canonical frame has `+Y` pointing down through the pallet height. The only
accepted object-frame symmetry rotations, in this exact order, are

```text
S0 = [[ 1, 0,  0],
      [ 0, 1,  0],
      [ 0, 0,  1]]

S180 = [[-1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0, -1]]
```

Both are proper rotations with determinant `+1`, and `S180 @ S180 = S0`.
The equivalence set is therefore `{0 degrees, 180 degrees}`. A 90-degree yaw is
not equivalent because it exchanges the unequal canonical X and Z extents
(`1.10 m` and `1.30 m`). Reflections and pitch/roll rotations are not accepted.

## Metric definition

The paper metric variant is `ADD-S`, implemented as finite-set symmetry-aware
ADD over corresponding canonical points:

```text
min over S in {S0, S180}
  mean_i ||(R_pred X_i + t_pred) - (R_gt S X_i + t_gt)||
```

Rotation and yaw errors take the same minimum over `{S0, S180}`. Translation is
unchanged. This is not unrestricted nearest-neighbour ADD-S, which could
silently grant unreviewed pitch/roll cuboid symmetries.

## Scope

- The equivalence applies to the pallet body's canonical geometric pose.
- Only yaw angles separated by 180 degrees are identified.
- Cargo, straps, fixtures, and directional attachments are outside the
  symmetric pallet-body definition and do not expand the rotation set.
- The same frozen rule must be used for DEV diagnostics and untouched FINAL;
  it may not be changed after observing pose results.

Freezing symmetry clears only the symmetry prerequisite. Canonical migration,
the GT-independent W/D selector, and untouched FINAL membership remain separate
pose-metric gates.
