# Real pallet GT v2 frame convention

Status: **CONTRACT DEFINED; LEGACY SIGN REPRESENTED AS A FROZEN EQUIVALENCE CLASS**

This is the paper-facing geometry contract. Legacy `dimensions_m` and
`pose_transform` are compatibility data, not physical-pose ground truth.

## Canonical physical frame

- Origin: geometric centroid of the nine-point cuboid; keypoint 8 is the origin.
- `+X`: designated positive direction of the physical 1.10 m axis.
- `+Y`: downward through the pallet height, 0.11 m.
- `+Z`: designated positive direction of the physical 1.30 m axis.
- `(X,Y,Z)` is right-handed. Thus the canonical top has `Y=-0.055` and bottom has
  `Y=+0.055`.
- Physical dimensions are always exactly `(x,y,z)=(1.10,0.11,1.30)` metres.

Axis lengths alone do not establish the two horizontal signs. For this paper,
the frozen benchmark contract identifies poses that differ only by a 180-degree
yaw, so the two signs are one pose class rather than two labels that must be
guessed from legacy data.

Canonical keypoint order is:

```text
0 (-X,-Y,-Z)    4 (-X,-Y,+Z)
1 (+X,-Y,-Z)    5 (+X,-Y,+Z)
2 (+X,+Y,-Z)    6 (+X,+Y,+Z)
3 (-X,+Y,-Z)    7 (-X,+Y,+Z)
8 centroid
```

Each coordinate above uses the corresponding half extent.

## Camera-facing PnP frame

The model keeps the established `camera_dynamic_0123_v4` order. Its local axes
are `+X=right`, `+Y=down`, `+Z=forward`; indices 0--3 are the near face. Its
width/depth therefore depend on which physical face points toward the camera.
This frame is useful for stable keypoint prediction but is not a fixed physical
pose frame.

Let `A` be the proper rotation from canonical coordinates to camera-facing
coordinates and let `perm[cf_index]` be the matching canonical keypoint index:

```text
P_cf[i] = A @ P_can[perm[i]]
R_can   = R_cf @ A
t_can   = t_cf
```

The unchanged translation is valid because both frames use the same centroid
origin.

| assignment | canonical-to-CF rotation `A` | CF `(width,height,depth)` | `perm[cf]=canonical` |
|---|---|---|---|
| `YAW_0` | `[[1,0,0],[0,1,0],[0,0,1]]` | `(1.10,.11,1.30)` | `[0,1,2,3,4,5,6,7,8]` |
| `YAW_90` | `[[0,0,1],[0,1,0],[-1,0,0]]` | `(1.30,.11,1.10)` | `[1,5,6,2,0,4,7,3,8]` |
| `YAW_180` | `[[-1,0,0],[0,1,0],[0,0,-1]]` | `(1.10,.11,1.30)` | `[5,4,7,6,1,0,3,2,8]` |
| `YAW_270` | `[[0,0,-1],[0,1,0],[1,0,0]]` | `(1.30,.11,1.10)` | `[4,0,3,7,5,1,2,6,8]` |

The implementation derives every permutation by coordinate-set matching; the
table is explanatory, not a second source of truth. Every accepted transform must
satisfy `A.T @ A = I` and `det(A)=+1`.

## Legacy identifiability limit

Legacy dimensions determine only yaw parity:

```text
legacy W/D = 1.10/1.30  -> YAW_0 or YAW_180
legacy W/D = 1.30/1.10  -> YAW_90 or YAW_270
```

Neither reprojection nor cheirality distinguishes the two members of each pair.
Migration therefore records both candidates and never picks the smaller
reprojection error as a physical sign. The frozen
`challenge/real_gt_v2/SYMMETRY_CONTRACT.json` declares each pair to be one
`YAW_180_EQUIVALENCE_CLASS`; the migration gate verifies that relation for every
frame while leaving `canonical_pose=null` and the signed assignment unset.

LR-only and front/back-only permutations are reflections (`det=-1`) and are
forbidden. A combined LR+front/back operation happens to be a proper 180-degree
yaw, but is accepted only when generated and checked as a 3D rotation.
