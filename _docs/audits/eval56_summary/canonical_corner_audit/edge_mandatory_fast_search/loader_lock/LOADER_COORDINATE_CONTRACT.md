# Loader coordinate contract

```
image space                400 x 400
refine_keypoints / beliefs  50 x 50      (dataset.output_size)
conversion                  image = grid * 8
```

Confirmed from code, not inferred: `dataset.output_size == 50`, `beliefs` is
`9x50x50`, and on interior channels the belief argmax agrees with
`refine_keypoints` to within a cell.

## The earlier audit was wrong and is kept, not deleted

A first pass treated `refine_keypoints` as 400 px image pixels.  Every derived
number was therefore computed against the wrong frame: the off-frame set, the
"104/256 retained" figure and the "34 mask mismatches" are all uninterpretable.
That file is preserved as
`loader_offframe_summary.INVALID_400PX_ASSUMPTION.json` with a correction JSON
beside it.

A second defect made it worse: gates and metrics shared one dict and the check
was `v is True or v == 0`.  Python evaluates `False == 0` as true, so a failing
gate printed ALL PASS.  Gates are now booleans only, aggregated with
`all(v is True for v in gates.values())`, and an empty dict cannot pass.

## Corrected audit

256 truncation frames x 4 seeds = 1,024 loads.

```
shape_bad 0 · non-finite 0 · sentinel 0
belief mask mismatch     0
affinity mask mismatch   0
off-grid coordinates retained   768 / 1,024 loads
peak consistency  n=5,608   median 0.400 cell   p99 0.656 cell
```

The semantics come out as intended: `refine_keypoints_valid = 1`,
`inside50 = false`, `belief_channel_mask = 0`.  The coordinate survives; only
the loss is masked.  A coordinate that disappeared with its mask set to zero
would not have been a pass.

Overlays draw `grid * 8` and never clamp an off-frame point to the border; they
print its id, grid coordinate, image coordinate and exit direction in the margin.
