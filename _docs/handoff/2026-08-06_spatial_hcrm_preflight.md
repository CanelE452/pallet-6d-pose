# Handoff — Spatial HCRM screen, preflight complete

Training has not started.  Zero optimizer steps, zero HCRM checkpoints, A1
untouched.  Everything below is measured, not assumed.

## Verified and frozen

```
A1 checkpoint     weights/paper_s2/paper_s2_pdg/A1/epoch_003.pth
                  sha256 00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657
                  run_state completed, its recorded sha matches the file

A1 canonical parity, reproduced by an actual forward, exact on all eight:
  eval56  centroid 55/56  R4 50/56  PnP 52/56  reproj 10.1608px
  wood    centroid 45/45  R4 42/45  PnP 44/45  reproj  8.9569px
  decoder MD.decode_all(...)['D0'], threshold 0.30, no padding, EvalFrame.solve

convention        NEAR_KP (0,1,2,3) / FAR_KP (4,5,6,7), from
                  paper_s2_mechanism_diagnostic
F50               PDGStage1Model.net.vgg(x) = vgg[26], B x 128 x 50 x 50
injection         beliefs[-1][:, :9], the tensor MD.decode_all reads
```

## Correction 1 -- the pointwise control was not pointwise

GroupNorm normalises over (C, H, W).  A single-cell perturbation moved all 2500
output cells in *both* arms, so H1 had access to a global spatial statistic and
could not have controlled for spatial context.  Replaced with
`ChannelLayerNorm2d`, a LayerNorm over the channel axis at each location.

```
                params    single-cell perturbation support
H1 pointwise    12,932    (1, 1) -- exactly its own cell
H2 spatial      14,852    (5, 5) -- 25 cells, the theoretical support
ratio 1.1485 (gate <= 1.5)      zero-init residual max abs 0.0 for both
```

## Correction 2 -- the split had to be built, not assumed

The A1 loader spans six roots.  Their group structure, established by reading
the data rather than guessing:

```
v4_split_base           4,000   camera_data hdri|background|floor      77 groups
paper_4pallet_mask_v1  10,000   same key                               77 groups
aug_{squash,trunc,scale} 6,308  filename {kind}_{parent}_v{n}
                                  b*  -> v4_split_base    2,239  key inherited
                                  m_* -> mixed_v8_train   4,069  no key
mixed_v8_train          9,000   no scene metadata at all   -> no key
```

So 16,239 frames carry an admissible group key and 13,069 do not.  The rule
forbids an index fallback, and none was used.  Holdout groups are drawn only
from the keyed population; every unkeyed frame stays in train.  That is only
safe if no unkeyed frame duplicates a holdout frame, so it was checked rather
than argued: sha256 over all 3,059 holdout PNGs against all 13,069 unkeyed PNGs.

```
frames    train 26,249   validation 1,603   untouched 1,456
groups    train     62   validation     8   untouched     7
holdout group overlap                    0
unkeyed-to-holdout image duplicates      0
split_sha256  49f6ef5123b89583...
```

Both holdouts clear the 256-frame recommendation.  `mixed_v8_train` being
train-only is a stated policy, not a silent drop, and is recorded in
`split_summary.json`.

## Not done

`EVAL_NO_AUG` dataset mode, untouched access guard, hard manifest, the nine
training runs, selection, evaluation, gates and the decision.  No runner exists
yet.  The name `DETERMINISTIC_AUGMENTATION` was retired and is not used
anywhere; holdouts are to be called `NO_AUG_SOURCE_HOLDOUT` once that mode
exists.

## Resuming

Read `input_lock_v2.json` and `synthetic_split_manifest.csv`; both are frozen.
`input_lock.json` is marked `SUPERSEDED_BEFORE_TRAINING` and kept for the trail.
Start at Phase D.
