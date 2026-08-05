# Truncation-Aware Canvas Augmentation

## Why it exists, stated precisely

The training audit found 3.3% of samples with any corner off-screen and 0.0%
with the centroid off-screen.  The existing `utils_dataset.apply_truncation_aug`
looked like a counter-example, so it was read line by line:

```
utils_dataset.py:247  apply_truncation_aug
  crop the pallet at a frame edge                       <- produces truncation
utils_dataset.py:277  out_img, out_kps = _trunc_pad_back(crop_img, kps_c)
utils_dataset.py:233  _trunc_pad_back
  cv2.copyMakeBorder(..., cv2.BORDER_REFLECT_101)
  pad chosen by _trunc_required_pad so every one of the nine keypoints lands
  inside the [0.20, 0.80] band                          <- undoes it again
```

**The existing truncation augmentation pads its own truncation back inside the
frame.**  It yields frames that look cropped while nothing is actually
off-screen, which is exactly why the centroid has never been supervised outside
the frame and why the 3.3% figure is what it is.  It also uses reflect padding,
which the earlier audit measured as the worst of the three border modes.

TACA keeps the crop and drops the pad-back.

## Fixed sampling and the measured realisation

```
declared        legacy 50%   frame_edge_truncation 25%   constant_margin_scale 25%
measured (2000 draws, seed 1)
                legacy 60.0%  truncation 14.1%  scale 25.9%
                truncation attempts that fell back to legacy: 11.1%
```

The truncation branch is drawn 25% of the time and succeeds on about 56% of
those draws within its 10-attempt budget, so the realised share is 14.1% and the
remainder falls back to legacy rather than retrying without bound.  The declared
ratio is the sampling policy; this is what the envelope admits.  Neither number
was adjusted after being measured.

## What the truncated samples look like

```
off-screen corners (median)   4
border proximity (median)     -47.0 px      D13 population: -32.8 px
bbox width ratio (median)     0.807         D13 population: 540/640 = 0.84
centroid in frame             100%
```

The produced geometry sits inside the D13 failure envelope, which is what the
branch is for.  The centroid stays in frame in every accepted sample, so the
off-screen centroid case is still not represented -- worth recording, because
the audit showed the centroid is the channel that collapses first.

## Scale branch

Constant grey at 127, pad 100 px per side, the geometry of
`dope_predict_mp4_pad.pad_frame`, then the existing squash.  Reflect and
replicate are absent from the module and a test asserts it.  The choice follows
the padding audit, where constant127 >= replicate > reflect on every recovery
metric, so the useful part was canvas margin rather than context continuation.
