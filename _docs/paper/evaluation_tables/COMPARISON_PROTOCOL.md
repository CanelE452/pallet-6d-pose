# Method comparison protocol

## A. Controlled method comparison

This comparison fixes the following for every method:

- G38 image membership and train/validation split;
- target-specific object exposure and real-supervision budget;
- the same frozen final test membership;
- one evaluation implementation and one physical-object geometry contract.

Every method derives its native target from the same underlying GT: YOLO uses a
box and keypoints, DOPE uses belief/affinity fields, PVNet uses masks/vector
fields, and SingleShotPose uses projected cuboid points.

Native optimizer, learning rate, scheduler, loss, augmentation, and a reasonable
convergence epoch are allowed, but all are disclosed. This is a **controlled
method comparison**, not an architecture-only experiment. The adopted paper
method is YOLO; DOPE is retained only as a comparison implementation, not as the
paper pipeline.

Every DEV comparison uses exactly `COMMON_DEV_POS128` with `DEV_NEG2689` where
negative performance is applicable. `DEV_POS140` may be used only for migration
and selector diagnostics. The paper-final table uses frozen `FINAL_POS` and
`FINAL_NEG` only.

## B. Native-setting reference

Native-setting results preserve conditions from each original method, including
CAD at inference, required bounding-box input, target-specific synthetic data,
real labels, and external pretraining. These rows disclose those conditions and
are not mixed with controlled rows to claim a winner.

## Architecture-only definition

The term architecture-only is reserved for experiments with identical trainer,
loss, output representation, data, augmentation, and training budget, where only
the backbone or head changes. The controlled method comparison above does not
meet that definition.

## Reporting guardrails

- Do not compare numbers obtained on different in-house memberships.
- Do not transfer published-dataset numbers into the controlled in-house table.
- Do not fill pose columns until canonical pose, selector, symmetry, and FINAL
  gates all pass.
- Publish unavailable or blocked values as blank table cells and structured nulls,
  never as zero.

