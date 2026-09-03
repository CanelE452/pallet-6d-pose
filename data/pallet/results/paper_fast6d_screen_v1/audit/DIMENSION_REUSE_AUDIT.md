# Dimension conditioning reuse audit

No dimension-conditioned network was trained in this screen, and none will be on
the strength of it.

## What exists

`challenge/yolo_pose_one_model/dimension_conditioning_probe/` and the related
spatial-fusion and concat runs are present in the repository as untracked
experiment directories.

## What the recorded evidence says

The pre-registered finding kept in project memory is that feeding correct
dimensions, zeroed dimensions and shuffled dimensions produced outcomes that did
not separate: the network did not use the dimension channel to localise better.
The exact dimensions already enter the pose through PnP geometry, which is where
they demonstrably help.

## Verdict

```text
DIMENSION_HEADROOM = WEAK
```

Weak rather than NONE because the probe measured classifier and pose outcome
rather than keypoint localisation directly, so the absence is of demonstrated
benefit rather than of any possible benefit. Either way it does not justify
training, and no new threshold was invented to reach this label.
