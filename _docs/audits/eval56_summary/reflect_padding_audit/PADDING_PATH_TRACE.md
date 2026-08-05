# Which padding, and why that constant

Two padding paths exist in the repository and only one is admissible here.

```
challenge/scripts/dope_predict_mp4_pad.py:207  pad_frame(img, pad, mode)
                                    :353  --pad default 100, --pad-mode reflect
   symmetric px pad on all four sides, cv2.BORDER_REFLECT_101,
   then cv2.resize back to (W, H).  No GT anywhere.        <- USED

challenge/scripts/pad_truncation_crops.py:45   MARGIN_FRAC = 0.20
                                     :63   required_pad(kps)
   derives the pad per frame so that the GT keypoints land inside a
   [0.20, 0.80] band.  This reads GT keypoints, which Phase 0 forbids at
   inference (rules 15 and 17).                            <- RECORDED, NOT USED
Deep_Object_Pose/common/utils_dataset.py:114   _TRUNC_MARGIN_FRAC = 0.20
   the same constant on the training-augmentation side.    <- NOT AN INFERENCE PATH
```

So the ratio this audit uses is the repository's own **inference** constant,
`pad = 100 px` per side, not the 0.20 fraction the plan expected.  The
difference is worth stating: on the 640x480 frames of D13 and C13, the
fractional form would give 128 px horizontally and 96 px vertically, while the
inference constant gives 100 px on all four sides.  The inference constant was
chosen because the fractional one is only ever applied through a GT-dependent
solver, and because this is an inference audit.

A1 and A2 call `pad_frame` directly.  A3 needs a 127 grey that `pad_frame` does
not expose (it offers black and white only), so its two lines are repeated with
the same pad, the same border call and the same resize-back; that repetition is
noted rather than presented as reuse.

No second ratio was tried, and no mode was added after the results were seen.
