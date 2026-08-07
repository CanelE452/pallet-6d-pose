# Addendum: what the 0.033px direct-PnP number does and does not establish

The R1C document says "the belief detour costs accuracy rather than adding any."
That sentence stays in the original file, and this corrects it.

## The overclaim

`GT edge -> CIGM -> solve_pose` returning a reprojection median of 0.033px is a
**geometry oracle**.  Its 2D correspondences are derived from ground truth, so it
measures whether the interface is well-posed, not whether it is more accurate
than a belief route that has to predict its own correspondences.  Comparing it
against C0 or F1 as if both were deployable is not a fair comparison, and I made
it.

## What is established

```
DIRECT_EDGE_TO_PNP_INTERFACE_VALID
```

Twelve edge roles and a fixed incidence carry enough information for the
canonical solver to recover pose, with no belief map and no decoder in the path.
The interface is not degenerate and not ill-conditioned.

Measured against `projected_cuboid` rather than against the solver's own input,
so it is not self-consistency:

```
solved                    507 / 512
GT-corner reprojection    median 0.0332 px   p90 0.0604 px
failures                  run1|001775, run1|008845, run1|008338,
                          run2|015514, run2|014556
```

## What is not established

That a predicted-edge route beats a belief route.  Nothing here predicts an edge.
Round-1 is the only measurement of that, and it produced 16-degree orientations
and 0.9% edge-only accuracy.  The gap between those two rows is the entire
remaining problem, which is why the next step measures how accurate an edge
predictor would have to be rather than assuming any accuracy is enough.

## Cause labels, narrowed

R1C recorded `JOINT_TRAINING_CREDIT_ASSIGNMENT_FAIL`.  That name asserts a
mechanism the experiment did not isolate.  Narrowed:

```
NOISY_PROPOSAL_COADAPTATION_FAIL   CONFIRMED
    EGCR trained against a noisy PEQ does not respond to clean edges at
    inference; EGCR trained against clean edges does.  Directly measured.

CREDIT_ASSIGNMENT_FAIL             HYPOTHESIS
    that the failure is specifically gradient credit assignment between PEQ and
    EGCR was never separated from the alternative that EGCR simply learned to
    down-weight an uninformative channel.
```

`EDGE_USE = false`, `EDGE_CONTRIBUTION_NON_POSITIVE` stands: ZERO reaches R4 254
against NORMAL 253 on both arms.
