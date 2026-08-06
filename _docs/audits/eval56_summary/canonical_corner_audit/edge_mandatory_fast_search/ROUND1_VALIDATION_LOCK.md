# Round-1 validation protocol lock

Written with **zero validation forwards executed**.  The manifest and the shuffle
below were built from metadata alone; no model output was read to produce them.

## Identity

```
R1B E1   weights/paper_s2_edge_fast/R1B/E1/round1.pth  sha 62b63c54a01f18a3...  {peq, egcr}
R1B E2   weights/paper_s2_edge_fast/R1B/E2/round1.pth  sha 984d189a3f507c08...  {peq, egcr, hcrm}
A1       weights/paper_s2_pdg/A1/epoch_003.pth         sha 00a0dcd8730e21d1...
split    9a755438...        calibration commit af3100d        CALIBRATION_V2 PASS

forbidden to load: weights/paper_s2_edge_fast/{E1,E2}/round1.pth  (R1A, selection_eligible false)
```

## validation512

Drawn from the 1,995 validation frames across all 39 appearance groups,
stratified on (diagnostic mode, pallet type, resolution, truncation) with a
per-group cap, ties broken by sha256 of the index.

```
                selected      pool
mode            29.9/28.1/21.5/20.5   30.0/28.4/21.2/20.4
pallet          29.1/26.4/24.8/19.7   28.9/27.0/24.5/19.6
resolution      46.5/27.5/15.0/10.9   46.7/27.4/14.8/11.1
run             48.6 / 51.4           50.0 / 50.0
truncation      34.0%                 34.5%
groups          39 / 39               max 19 frames per group
manifest sha    b236bd3d75c77397...
```

## EDGE_SHUFFLE

```
permutation  [8, 11, 4, 7, 5, 0, 1, 9, 2, 10, 6, 3]
derangement  yes, zero fixed points, not the identity, seed 1
```

Shuffling moves the whole role tuple -- centre, direction, half-length, support --
while CIGM keeps the original incidence, so only role semantics break.

## Evaluation rules fixed here

Validation never touches the A1 TRAIN loader.  The path is source RGB, an
anisotropic resize to 400x400, ImageNet normalisation, frozen A1.  No crop, no
rotation, no photometric change, no train-time truncation augmentation, no
re-projection through K and pose, no re-application of perm_v4.

Ground truth is `projected_cuboid` scaled to the 50 grid with the frame's own
width and height, and predictions return to source pixels the same way.  One A1
forward per batch, shared by C0, E1, E2 and every ablation.

`EDGE_ZERO` removes only the edge contribution to the final belief.  On E1 that
must reproduce C0 exactly; on E2 the HCRM residual stays, so E2 NORMAL minus E2
ZERO is the edge's own contribution.

## Query activity, defined before any result

Role k is active when its finite-prediction rate is at least 99%, its median
half-length exceeds 0.25 cell, its direction norm is finite, and its output is
not byte-identical to another role.

## Gates

Unchanged from the pre-registered set: 12/12 active, orientation median <= 15
degrees, perpendicular offset <= 5 cells, CIGM valid >= 50%, edge-only <= 20px on
at least 20%, ID1+2 +3pp or R4 +1, far >50px increase <= 5pp, and shuffle costing
at least 10pp of edge-only <= 20px.  Plus the edge-use requirement: EDGE_ZERO
must cost at least 3pp of near <= 20px or at least one R4 frame, otherwise
EDGE_PRESENT_BUT_NOT_USED regardless of headline performance.

Nothing here may change after a validation number is seen.
