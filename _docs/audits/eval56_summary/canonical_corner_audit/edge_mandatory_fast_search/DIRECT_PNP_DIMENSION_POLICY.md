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

---

# Resolved by problem setting (2026-08-07)

The block above was correct for the setting I assumed, and that assumption was
mine rather than the project's.  The paper problem setting is declared as:

```
PAPER_PROBLEM_SETTING          RGB + K + KNOWN_TARGET_DIMENSIONS
DIRECT_PNP_STATUS              ALLOWED_UNDER_KNOWN_DIMENSIONS
HARD_BLOCKED_DIMENSION_ORACLE  valid only under RGB_ONLY + UNKNOWN_METRIC_DIMENSIONS
```

Target dimensions are known object metadata at inference, so supplying them to
`solve_pose` is an input, not a leak.  SLQ-DPnP phases 1 through 12 are unblocked.

## What the census still says

The 10,000 unique tuples stand.  They describe the *dataset* -- size is randomised
per frame so a predictor cannot memorise one ratio -- not the deployment problem.
That randomisation is the paper track's generalisation goal, and it is also why
`pallet_type` cannot be used as a size lookup: at inference the dimensions come
from the caller, not from a type classifier.

Practical consequence for the model: it must work across width 0.61-1.66 m and
depth 0.62-1.66 m, so nothing may assume a fixed aspect ratio internally.

## What is still oracular

Only the dimension precondition is lifted.  The other one is not:

```
R1C direct PnP        507/512, 0.033px    correspondences derived from GT edges
edge noise budget     1.0 deg / 0.5 cell  perturbed GT lines
```

No problem setting supplies ground-truth 2D correspondences.  Those two results
remain capacity oracles, and the budget remains a target for a predictor rather
than a measured deployable accuracy.  `DIRECT_EDGE_TO_PNP_INTERFACE_VALID` now
reads: valid, given known dimensions and given edge lines accurate to roughly
1 degree and 0.5 cell -- and producing those lines is the open problem.

## Note against CLAUDE.md

`CLAUDE.md` currently reads that the paper track uses PnP only on
known-dimension data and lists the config's KS T-11 sizes as a v8 leftover
needing measurement.  Under this declaration the first clause is satisfied by the
problem setting rather than by restricting the data, and the two statements
should be reconciled there.  I have not edited `CLAUDE.md`; flagging it so the
inconsistency is not discovered later as a contradiction.
