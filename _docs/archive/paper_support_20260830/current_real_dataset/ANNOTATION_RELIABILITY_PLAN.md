# Multi-shape annotation reliability plan

```text
RELIABILITY_MEASURED = false
RELIABILITY_STATUS = PREPARED_NOT_MEASURED
RECOMMENDED_SAMPLE_N = 40
PLASTIC_SAMPLE_N = 28
WOOD_SAMPLE_N = 12
```

This study measures annotation noise, not model performance. No reliability
value exists until two completed blinded annotation records are locked.

## Sampling

Select 28 plastic and 12 wood DEV frames before annotation. Stratify across
object type, actual capture provenance/session, observed DAY/NIGHT, projected
size, and elevation. Plastic NIGHT must not be under-sampled. If the wood source
does not contain a verified NIGHT condition, do not invent one.

Wood membership spans `wood_183705` and `wood_184309`; both sessions must be
represented. Plastic `eval_outside` is a source-set stratum but not evidence of
an independent capture session.

Every sample record binds `object_type`, `source_population`, source-label SHA,
the relevant source-audit SHA, geometry-registry SHA, and projected-cuboid bbox
diagonal used for NME.

## Blinded procedure

Use two independent annotators, or one annotator performing a delayed blind
repeat. Both are blind to existing GT, the other annotation, and all model
predictions. Record nine keypoints, point visibility/reason, frame occlusion,
truncation, and pose under that sample's registered object geometry.

## Endpoints

Report each endpoint for `ALL`, `PLASTIC`, and `WOOD`:

- frame NME median/p90: mean eight-corner A/B disagreement divided by the
  frozen existing-GT projected-cuboid bbox diagonal;
- raw corner-pixel median/p90 and centroid disagreement median/p90;
- rotation, translation, and yaw disagreement median/p90.

Pose disagreement uses the object-specific frozen symmetry contract. Plastic
uses `{I, Ry(180°)}`. Wood pose endpoints remain structured `null` while wood
symmetry is `UNREVIEWED`; plastic symmetry is never copied merely to complete
the table. The ALL pose endpoint is likewise null when wood is blocked. The 2D
endpoints may still be reported for all three subsets.

## Machine-artifact state

The active machine-readable contract is the coordinated multi-shape v2 set:
`SAMPLING.json/csv`, A/B templates, the noise-floor template, and calculator
implement one 40-frame membership split into plastic 28 and wood 12, with
ALL/PLASTIC/WOOD subgroup output. Its status is
`PREPARED_NOT_MEASURED`: preparation is complete, but the two blinded human
annotation records do not yet exist. Do not mix historical v1 sampling files
with v2 annotations.

The final status remains `PREPARED_NOT_MEASURED`; it is never promoted to PASS
from blank templates or PnP reprojection self-consistency.
