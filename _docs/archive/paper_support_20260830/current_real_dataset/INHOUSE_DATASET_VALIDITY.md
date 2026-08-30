# In-house dataset validity

```text
DATASET_QUALITY_AUDIT = PARTIAL
MULTISHAPE_DEV_MEMBERSHIP = AUDITED
ANNOTATION_RELIABILITY = PREPARED_NOT_MEASURED
FINAL_VALIDITY = NOT_ESTABLISHED
```

## Plastic DEV

- 140 reviewed labels: DAY 112 / NIGHT 28, 640×480.
- Common comparison population: 128, DAY 100 / NIGHT 28.
- GT-v2 migration/fixed geometry passed without modifying sources.
- Per-keypoint visibility, occlusion, and truncation remain unreviewed.
- Prediction-only axis selector failed, blocking pose metrics.

## Wood DEV

- 45 labels in two sessions: 25 + 20, all 1280×720.
- Canonical geometry `(0.80,0.14,0.59)` m is separate from plastic.
- Membership and GT QA passed; no exact overlap with plastic DEV or negatives.
- Nine geometrically truncated frames are retained and disclosed, not silently
  removed.
- All 405 point-visibility states remain unknown.
- Intrinsics are `SENSOR_PROFILE_SCALED`, not per-session calibrated.
- Symmetry is unreviewed and selector is not run, so pose remains blocked.
- Prior Stage-B/wood use makes this `CROSS_SHAPE_DEV`, not FINAL.

## Split integrity

Plastic positive/negative exact-hash overlap is zero. Wood exact image overlap
with plastic DEV and negative DEV is zero. Bare stem equality is not a valid
overlap test because all 45 six-digit wood stems also occur among negative stem
names; session-qualified IDs and image hashes are required.

The existing negative population has one duplicate image pair and lacks full
capture-session metadata. Frame-wise CI is not substituted for a missing
session-cluster CI.

## Claims supported and unsupported

Supported: two registered physical object geometries, manual development GT,
zero exact cross-population image overlap, and development cross-shape coverage.

Unsupported: untouched FINAL performance, generic-pallet generalization,
unseen-topology claims, annotation accuracy/noise floor, occlusion robustness,
or reliable pose under the current selectors/symmetry state.
