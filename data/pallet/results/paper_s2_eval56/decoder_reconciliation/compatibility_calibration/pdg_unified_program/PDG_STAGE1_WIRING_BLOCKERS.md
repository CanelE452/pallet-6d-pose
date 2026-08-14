# Two blockers found wiring Stage 1 to the trainer

Both are design facts about the canonical pipeline, not bugs, and both change
how A1 and A2 have to be built.  Recorded before writing the trainer so the
plumbing is built once.

## 1. TACA cannot be a batch-level transform

```
canonical loader delivers      img 400x400 (squashed), beliefs 50x50,
                               refine_keypoints in belief-grid units
loader options                 imagesize 400, sigma 2.0
TACA output canvas             640 x 480
TACA acceptance envelope       border proximity -64 .. +16 px
                               bbox width ratio 0.65 .. 1.05
```

The envelope is stated in 640x480 source pixels.  The loader has already
squashed 640x480 frames to 400x400, which is anisotropic: x is scaled by
400/640 and y by 400/480.  Cropping that squashed image and squashing again
compounds the anisotropy, and a border measured in 400-space pixels is not the
quantity the envelope was derived from.

So TACA has to run **before** the squash, on the source frame, which means a
Dataset wrapper that re-reads the image and JSON rather than a transform applied
to a batch the loader has already produced.  The wrapper then has to regenerate
the targets, because the transformed keypoints no longer match the loader's.

## 2. A1 needs the same regeneration as A2

`CreateBeliefMap` takes one `sigma` for all nine channels and the canonical
loader is configured with `sigma = 2.0`.  Stage 1's target policy asks for
corners at 2.0 and the centroid at 2.5, so the centroid channel cannot come from
the loader as it stands.

A1 was specified as the control that changes only the target bandwidth.  That
still holds conceptually, but it is not "train the existing loader with a
different setting" -- A1 needs the same target-regeneration path as A2, with
TACA disabled.  `pdg_targets.build_targets` already does the role-specific part;
what is missing is the wrapper that feeds it.

## What the wrapper has to do

```
read source image and JSON at native resolution
A2 only: TACA in 640x480 space           pdg_taca.apply
project keypoints through the transform
regenerate belief targets                pdg_targets.build_targets  (sigma 2.0 / 2.5)
regenerate affinity targets              utils_belief.GenerateMapAffinity
apply the off-screen masks               belief mask 0, affinity 2i and 2i+1 at 0
build visibility labels                  three-state, from transformed coordinates
build the palletness target              pdg_targets.palletness_target
squash to 400x400 and normalise          the existing preprocessing
carry through the DiffPnP fields         gated to fully in-frame samples
```

None of this changes the modules already committed; it is the connective layer
between them and the trainer.

## State

```
optimizer steps      0
checkpoints written  0
E44 inference        0        SEALED
W45 inference        0        SEALED
```
