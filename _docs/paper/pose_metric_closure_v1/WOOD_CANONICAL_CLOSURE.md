# Wood canonical migration closure

```text
W2 = FAIL
```

The geometry is clean. The failure is that the signed pose is undetermined, and its
only missing input is the symmetry contract from W1.

## Gate state

```text
challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json

status                                     BLOCKED
blocked_reason                             UNCONFIRMED_SIGNED_CANONICAL_AXIS
pose_resolution_mode                       UNRESOLVED_SIGNED_AXIS
manual_review_required_count               45        (PASS requires 0)
canonical_pose_equivalence_resolved_count  0
symmetry_contract_path / sha256            null
```

## What was checked, and what it found

```text
check                                       result                            n
──────────────────────────────────────────────────────────────────────────────
0<->1 edge                                  camera-facing W axis (kp0 = -w)   —
0<->4 edge                                  camera-facing D axis              —
0<->3 edge                                  camera-facing H axis (-h = up)    —
W/D/H against registry                      every frame matches one of the
                                            two parities of (0.80, 0.14,
                                            0.59); no third value exists    125
axis candidates                             always exactly a 180-degree pair 125
canonical <-> camera-facing roundtrip       max 2.3e-13 px                   125
model reprojection vs stored annotation
  DEV45                                     median-of-medians 1.94 px         45
                                            worst single keypoint 11.07 px
                                            frames with max > 8 px: 5 / 45
  day / night                               median 0.00-0.02 px               80
                                            worst 3.7 / 4.5 px
plastic convention conflict                 none. Same keypoint_frame
                                            "camera_dynamic_0123_v4",
                                            {0,1,4,5} = up, 8 = centroid,
                                            shared diagram points; only the
                                            dimensions differ                125
signed canonical pose                       canonical_pose = null            125
                                            pose_status
                                            UNCONFIRMED_SIGNED_AXIS           80
```

## Reading the residuals

There is **no geometric inconsistency**. The coordinate transforms are exact to
1e-13 px. The residual between the fitted model and the stored clicks is 1.94 px
median on DEV45 and essentially zero on the newer 80 frames — that is annotation
click noise, not a convention error. Two frames of 45 exceed 4 px median.

So W2 is not a mismatch to repair. It is an **undetermined sign**, and the
undetermined degree of freedom is always exactly one: 180 degrees about Y.

## What would close it

Supplying the wood symmetry contract and re-running the migration would flip the 45
frames from `MANUAL_REVIEW_REQUIRED` to `CANONICAL_POSE_EQUIVALENCE_RESOLVED`
(`scripts/annotate/migrate_real_gt_v2.py:960-999`). The other 14 checks already pass
and their maxima (4.4e-16, 1.3e-13 px) sit far below their thresholds (1e-6, 1e-4).
This is the same path the 140 plastic frames took.

`[추정]` — the re-run was not performed. The audit was read-only by instruction, so
even a dry run was not executed. The claim is that the inputs are in place, not that
the output was observed.

## Two further blockers beyond W1 and W2

Reported because they change the final verdict even if both close.

### Wood selector has never been run

```text
WOOD_SELECTOR_STATUS.json   NOT_RUN
gate                        overall >= 0.95, night >= 0.90, min session >= 0.85
```

This is the real risk. Plastic already failed the same gate at 0.5929.

One reason for cautious optimism, and it is a prediction rather than a result:

```text
plastic footprint aspect   1.10 / 1.30   ratio 1.182
wood    footprint aspect   0.80 / 0.59   ratio 1.356
```

The unit test `test_set_distance_shrinks_as_the_footprint_approaches_square` shows
the cost of a wrong hypothesis growing with the aspect difference. Wood is further
from square than plastic, so its two hypotheses should be more separable
geometrically. Whether that is enough to cross 0.95 is unknown and untested, and
wood's 56 nighttime frames carry their own 0.90 gate.

### No held-out wood population

```text
FINAL_WOOD_POS   membership_status UNAVAILABLE, items 0
current 125      all CROSS_SHAPE_DEV, held_out_final false
```

### Intrinsics quality caveat

```text
DEV45          SENSOR_PROFILE_SCALED   1920x1080 profile scaled by 2/3;
                                       no distortion model, no camera serial
day / night 80 UNKNOWN
```

Any absolute translation or ADD value for wood must carry this caveat. Rotation and
yaw are far less sensitive to it than translation is.

## Verdict

```text
WOOD_POSE_ELIGIBLE = NO
```

Running the actual gate function with the current artifact values fails all four
conditions:

```text
canonical_migration = BLOCKED     -> CANONICAL_MIGRATION_NOT_PASS
selector            = NOT_RUN     -> POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR
symmetry            = NOT_FROZEN  -> SYMMETRY_NOT_FROZEN
final_manifest      = NOT_FROZEN  -> FINAL_MANIFEST_NOT_FROZEN
```

Closing W1 and W2 removes two of the four. The selector and the held-out population
remain, and the selector is the same problem that plastic has already failed.
