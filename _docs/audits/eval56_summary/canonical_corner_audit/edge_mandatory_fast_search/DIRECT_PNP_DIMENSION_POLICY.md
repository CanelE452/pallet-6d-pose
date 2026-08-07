# HARD_BLOCKED_DIMENSION_ORACLE

Phase 0 of the SLQ-DPnP plan asks whether direct PnP is quietly consuming
per-frame ground-truth dimensions.  It is.

## Census

10,000 frames sampled, **10,000 unique dimension tuples**.  Size is randomised
per frame, and `pallet_type` does not determine it:

```
Pallet_0   2,394 unique   width 0.729-1.530   depth 0.703-1.579   height 0.107-0.200
Pallet_1   2,548 unique   width 0.733-1.572   depth 0.705-1.553   height 0.098-0.184
Pallet_2   2,570 unique   width 0.757-1.548   depth 0.727-1.542   height 0.127-0.243
Pallet_3   2,488 unique   width 0.610-1.660   depth 0.618-1.656   height 0.111-0.202
```

```
A  KNOWN_OBJECT_DIMS   no -- there is no one shared size
B  KNOWN_TYPE_DIMS     no -- type carries no size information
C  per-frame ground-truth dims required at inference  ->  HARD_BLOCKED_DIMENSION_ORACLE
```

This is not a surprise in the project's own terms.  `CLAUDE.md` records that the
paper track deliberately has no fixed ratio -- squash augmentation across many
ratios is the generalisation goal -- and that PnP belongs only where dimensions
are known.  This dataset is the paper track.

## What this retroactively narrows

Two earlier numbers were produced by calling
`solve_pose(points, K, dims=ground_truth_dimensions)`:

```
R1C direct PnP        507/512 solved, GT-corner reprojection median 0.033px
edge noise budget     ANGLE_BUDGET 1.0 deg, OFFSET_BUDGET 0.5 cell
```

Both already carried one oracle -- correspondences derived from ground truth --
and they carried a second I did not name: the dimensions.  So
`DIRECT_EDGE_TO_PNP_INTERFACE_VALID` holds **under known dimensions**, and the
budget is the accuracy an edge predictor would need *given* that the object's
size is supplied.  Neither statement was wrong; both were stated without their
second precondition, and that is corrected here rather than in the original
files.

## Consequence for SLQ-DPnP

The plan's own rule applies -- a dimension predictor is not to be smuggled in --
so the twelve-line to direct-PnP architecture cannot be built against this
dataset as specified.  Phases 1 through 12 are not executed and no new model was
written.

Three admissible directions, none started:

1. Restrict direct PnP to data where dimensions are genuinely known.  The
   canonical sets are that case -- eval56 and wood45 are the user's own pallets
   with measured dimensions -- but they stay sealed, so this is a later decision
   rather than a move to make now.
2. Treat dimension estimation as an explicit, separately screened component with
   its own gate, instead of an implicit input.
3. Adopt a pose formulation that does not need metric size, accept the scale
   ambiguity, and re-derive the budget under it.

Choosing among these is a research decision, not a defect to patch.

## Standing

Round-1 (`NO_EDGE_ARCHITECTURE_PASS`), R1C (`FUSION_CAPACITY_VALID`) and the
noise budget are unchanged; this document narrows what they claim.  untouched,
eval56, wood45 and final-test remain unopened.
