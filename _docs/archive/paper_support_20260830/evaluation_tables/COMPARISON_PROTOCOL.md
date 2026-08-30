# Method comparison protocol

## Controlled method comparison

Every method uses the exact G38 38,002/1,998 membership and split, zero real
supervision, zero evaluation-target training exposure, the same fixed evaluation
memberships, the same object-geometry registry, and the same evaluator. Native
optimizer, learning rate, schedule, loss, augmentation, and a reasonable
convergence rule are allowed when disclosed.

This is a controlled **method** comparison, not architecture-only. Each method
may derive a native target from the same source: boxes/keypoints, belief maps,
vector fields, or native cuboid targets. It may not add target-specific real
labels or private rendered examples.

The evaluator uses manifest `object_type` to dispatch plastic or wood geometry.
Every method is evaluated on exactly the same `DEV_WOOD_POS45`; membership is
not changed after any result is inspected.

Development pairs are:

```text
PLASTIC  COMMON_DEV_PLASTIC_POS128 + DEV_NEG2689
WOOD     DEV_WOOD_POS45 + DEV_NEG2689
ALL      COMMON_DEV_MULTISHAPE_POS + DEV_NEG2689
```

Wood45 is **Development cross-shape evaluation**, not untouched FINAL. Main
paper rows require new frozen FINAL plastic, wood, all-positive, and negative
populations.

## Native-setting reference

Native CAD, supplied bbox, target-specific synthetic data, real labels, and
external pretraining remain disclosed in a separate Table 1b. Published-dataset
numbers are not copied into the in-house controlled table.

## Architecture-only definition

Architecture-only means identical trainer, loss, output representation, data,
augmentation, and budget with only a backbone/head change. The controlled
method comparison does not meet that definition.

## Guardrails

- Unknown object type or missing geometry/symmetry fails closed.
- Plastic selector thresholds are not changed after its DEV FAIL.
- Wood symmetry is reviewed independently; plastic symmetry is not inherited.
- Box/2D results may proceed while pose stays null.
- ALL pose is null if any constituent object is blocked.
- Unavailable is blank/structured null, never zero.
- Every row includes exact membership and provenance metadata.

