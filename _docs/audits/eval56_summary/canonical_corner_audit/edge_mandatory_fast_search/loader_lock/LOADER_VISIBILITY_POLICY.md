# Loader visibility is not a source of truth for paper20k

`CleanVisiiDopeLoader._compute_visibility` applies `pose_transform` to the stored
cuboid.  For this dataset the projection lock established that `cuboid` holds
**world** coordinates and that `pose_transform` is not the matrix for them --
that composition is what produced the 272 px and 412 px residuals.  So
`sample["visibility"]` on paper20k is computed through an invalid frame
composition and cannot be trusted.

The loader is **not** patched.  A global change would alter A1's own behaviour on
the datasets it was fitted on, and nothing here needs that.  The isolation is
enforced in the PEQ pipeline instead.

```
status   LOADER_VISIBILITY_UNTRUSTED_FOR_PAPER20K
scope    paper20k only; A1's existing datasets are untouched
blocks   nothing -- belief and affinity loader compatibility is unaffected
```

## Forbidden in the PEQ pipeline

```
sample["visibility"] -> edge targets
sample["visibility"] -> visible/occluded labels
sample["visibility"] -> difficulty strata
sample["visibility"] -> eligibility
```

A test asserts the runner and the edge-target code reference it zero times.

## Used instead

```
edge_support           segment against the [0,50) x [0,50) grid frame, from
                       transformed refine_keypoints
occlusion diagnostics  mask_visible / mask_amodal / source metadata
difficulty strata      index.csv, usable_manifest, source label metadata
visible keypoint count source metadata, never the loader field
```

`edge_support` is geometric frame support, not visibility.  An edge hidden
behind cargo but geometrically inside the grid keeps `regression_mask = 1`, so
the amodal supervision the twelve roles need survives; only a fully off-grid
edge is dropped from direct regression.
