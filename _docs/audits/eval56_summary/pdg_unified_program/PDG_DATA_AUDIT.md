# Training distribution audit (Phase D)

6,000 samples drawn from the canonical Stage-B loader (29,308 samples,
mixed_v8_train + paper_4pallet_mask_v1), scored on the belief-grid keypoint
targets the loader itself produces.  No model was run.

## Off-screen corners in training

```
sample class  count  share
──────────────────────────
0 off-screen   5800  96.7%
1 off-screen     24   0.4%
2 off-screen    133   2.2%
3 off-screen     37   0.6%
4 off-screen      6   0.1%
```

```
truncated (at least one corner off-screen)     200   3.3%
full view (all 8 corners in frame)            5800   96.7%
centroid off-screen                              0   0.0%
```

## Against the failure population

```
             metric  training median  D13 median
────────────────────────────────────────────────
      bbox width px            241.3       540.0
border proximity px            118.6       -32.8
   in-frame corners              8.0         6.0
    truncated share             3.3%       76.9%
```

```
samples inside the D13 envelope
(bbox width >= 416px AND border <= +16px AND 4-7 corners in frame)
    115 of 6000 = 1.92%
```

## Reading

The training distribution is close to the **complement** of the population that
fails.  96.7% of samples show the whole pallet; the failure set is 76.9%
truncated.  The training median puts the pallet 118.6px clear of the nearest
edge; the failure set has the bounding box 32.8px **past** the edge.  Fewer than
2% of training samples land anywhere near the failing regime.

Most decisive: **the centroid is off-screen in 0.0% of training samples.**  The
centroid channel has never once been asked to handle its own target leaving the
frame, which is consistent with it being the channel that collapses first on
truncated frames, and with the off-screen corners in the padding audit landing
~290px from GT.

This is the quantitative case for Truncation-Aware Canvas Augmentation: not that
padding is the fix -- the previous audit showed it is not -- but that the
supervision has essentially never seen the regime that fails.  The audit does
not change the pre-fixed TACA ratios (50/25/25); it only records why they are
the intervention under test.
