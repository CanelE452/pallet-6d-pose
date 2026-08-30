# Real pallet GT v2 schema

Status: **DEFINED**

Scope note: the schema is object-aware, but the worked values below document
the historical **`plastic_standard_110x130x11`** migration. Wood v2 labels must
declare `wood_small_80x59x14` and use registry dimensions
`(x,y,z)=(0.80,0.14,0.59)`; they must not copy the example's plastic dimensions
or symmetry state.

`real_pallet_gt_v2` extends the existing label rather than rewriting its legacy
contract. A migrated JSON is a new file. The source JSON is never edited.

## Required additions

`objects` contains exactly one object (a pallet in this dataset). The following
is a minimal schema-valid unresolved example (the identity camera-facing pose is
illustrative; real files retain their measured transform). In particular, an unresolved
`canonical_pose` does **not** permit an empty candidate list: W/D parity always
produces exactly one of the two signed pairs `YAW_0/YAW_180` or
`YAW_90/YAW_270`.

```json
{
  "schema_version": "real_pallet_gt_v2",
  "objects": [{
    "keypoint_frame": "camera_dynamic_0123_v4",
    "physical_dimensions_m": {"x": 1.1, "y": 0.11, "z": 1.3},
    "camera_facing_pnp": {
      "axis_assignment": null,
      "axis_assignment_candidates": ["YAW_0", "YAW_180"],
      "dimensions_m": {"width": 1.1, "height": 0.11, "depth": 1.3},
      "pose_transform": [
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1]
      ]
    },
    "canonical_pose": null,
    "canonical_pose_candidates": [{
      "axis_assignment": "YAW_0",
      "pose_transform": [
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1]
      ],
      "canonical_to_camera_facing_rotation": [
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
      ],
      "canonical_to_camera_facing_keypoint_permutation": [0,1,2,3,4,5,6,7,8]
    }, {
      "axis_assignment": "YAW_180",
      "pose_transform": [
        [-1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, -1, 0], [0, 0, 0, 1]
      ],
      "canonical_to_camera_facing_rotation": [
        [-1, 0, 0], [0, 1, 0], [0, 0, -1]
      ],
      "canonical_to_camera_facing_keypoint_permutation": [5,4,7,6,1,0,3,2,8]
    }],
    "keypoint_annotations": [
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"},
      {"xy": [0,0], "visibility": 0, "in_frame": true, "source": "unknown", "reason": "unknown"}
    ],
    "occlusion_level": "unknown",
    "truncation": {
      "is_truncated": false,
      "outside_keypoints": [],
      "bbox_outside_fraction": 0.0
    },
    "legacy": {
      "dimensions_m": {"width": 1.1, "height": 0.11, "depth": 1.3},
      "pose_transform": [
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1]
      ],
      "fix_swap": null
    },
    "migration_status": "MANUAL_REVIEW_REQUIRED"
  }]
}
```

`population_role`, `capture_session_id`, `camera_serial`, `capture_timestamp`,
and `lighting_condition` are optional for legacy migration and omitted when
unknown. The annotation CLI accepts them explicitly for new captures. A
paper-final capture must populate the capture metadata through its protocol
rather than guessing it from directory names.

For a signed, human-confirmed assignment, `canonical_pose` contains
`pose_transform`, `canonical_to_camera_facing_rotation`, and
`canonical_to_camera_facing_keypoint_permutation`. For unresolved legacy data it
is `null`; candidate poses remain diagnostics and are not paper GT. The same
deterministic parity pair remains present after confirmation; the selected
`camera_facing_pnp.axis_assignment` and `canonical_pose.axis_assignment` must
agree and must be one member of that pair.

There is one explicit paper-evaluation exception to the singular-pose rule. If
the migration gate is `PASS` in `YAW_180_EQUIVALENCE_CLASS` mode and is bound by
path and SHA-256 to the frozen symmetry contract, the two candidates together
are the canonical quotient-pose GT. In that mode `canonical_pose` and the signed
`axis_assignment` deliberately remain `null`; neither candidate is promoted as
a physical direction. The evaluator revalidates the exact two-member class and
permits only restricted ADD-S and modulo-180 rotation/yaw calculations.

The original top-level object fields `dimensions_m`, `pose_transform`,
`manual_kps`, projected points, and other legacy values remain unchanged in the
new copy. Their originals are also captured under `legacy` so consumers cannot
mistake them for canonical physical fields. The paper evaluator selects an
object type from the manifest, resolves dimensions from the geometry registry,
and only then checks that the v2 label agrees. Neither legacy nor v2 label
dimensions may choose the evaluator geometry.

## Annotation workflow

`scripts/annotate/annotate.py` treats
`challenge/data/01_real/{manual_gt,eval_canonical}` as read-only. With no
`--out_dir`, it reads an existing legacy frame only as a fallback and saves the
additive copy under `challenge/data/01_real/gt_v2_canonical/<legacy-layout>/`.
An explicit output inside either legacy tree is rejected, including paths that
reach one through a symlink. `--population-role DEV|FINAL` is always required;
directory names never infer the role.

The review keys are intentionally two-stage:

- `w`: switch W/D parity (`short_face_front` / `long_face_front`), clearing any
  signed-axis confirmation from the previous parity;
- `y`: confirm or cycle the two signed candidates allowed by the selected
  parity;
- `b`: cycle the active keypoint visibility/reason state.

The current annotation editor writes singular poses, so FINAL saves through that
editor require known visibility for kp0–7 and a confirmed signed axis. This
stricter editor rule does not alter mechanically migrated equivalence-class GT,
whose signed direction remains unset by design.
Resetting every point and pressing save cannot delete a FINAL annotation or
bypass those gates.

## Per-keypoint annotation

`keypoint_annotations` has exactly nine entries:

```json
{
  "xy": [123.0, 45.0],
  "visibility": 0,
  "in_frame": true,
  "source": "unknown",
  "reason": "unknown"
}
```

- visibility `0`: no reliable location/visibility label, or unknown provenance;
- visibility `1`: not directly visible, but its location was deliberately
  estimated and labelled;
- visibility `2`: directly visible and human-confirmed.

Allowed sources are `manual_click`, `extrapolated`, `pnp_projected`,
`centroid_auto`, and `unknown`. Allowed reasons are `visible`, `occluded`,
`truncated`, and `unknown`.

Legacy migration does not infer human visibility from `manual_kps`, object-level
visibility, or an in-frame projection. When a valid nine-boolean legacy
`extrapolated_mask` accompanies nine `manual_kps`, the migrator preserves the
manual coordinate and transfers only its known source: mask false becomes
`manual_click`, and mask true becomes `extrapolated`. This still writes
`visibility=0` and `reason=unknown`; provenance does not prove visibility or
occlusion. Without that complete mask contract it keeps the stored projected
coordinate with `source=unknown`. Geometric `in_frame` and truncation are derived
from the selected coordinate and image bounds; `bbox_outside_fraction` remains a
legacy-projection diagnostic. Occlusion stays `unknown` unless reviewed from
image content.

## Local YOLO conversion

The audited Ultralytics 8.4.60 pose loss masks only visibility zero. Values 1 and
2 are both supervised valid points. The converter mapping is:

```text
v0 or missing xy -> 0 0 0
v1               -> x y 1
v2               -> x y 2
```

This is a local implementation fact, not an assumption that values 1 and 2 have
different loss weights. The older repository auditor that permits only 0/2 is not
the v2 schema validator. The executable mapping is
`real_gt_v2_schema.keypoint_annotations_to_ultralytics`; coordinate normalization
is deliberately left to the surrounding dataset converter.

## Validation rules

- exactly one object (the population contract supplies a pallet) and nine
  keypoint annotations;
- canonical physical dimensions exactly `1.10/.11/1.30`;
- visibility/source/reason values from the enums above;
- `xy=null` whenever no location label exists;
- resolved canonical rotations orthonormal with determinant +1;
- a non-promoted signed axis means `canonical_pose=null` and explicit candidates;
- unresolved candidates exactly match the deterministic two-element W/D parity
  pair; neither an empty list nor a single promoted candidate is valid;
- a paper evaluator may treat that pair as resolved only through a validated
  `YAW_180_EQUIVALENCE_CLASS` migration PASS bound to the frozen symmetry file;
- no paper evaluator fallback to legacy pose or dimensions.
