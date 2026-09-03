# FAST 6D SCREEN — bbox and structural cues for full 6D pose

No model was trained. Monocular RGB in, full six-degree pose out, no depth. The
question was whether any reformulation of how the existing signals are combined
deserves a short training run.

```text
FAST_6D_SCREEN = NO_PROMOTABLE_SIGNAL
PROMOTED_ARM   = None
```

The population is PAPER_EVAL positive, 319 frames, already used for development.
Nothing here may be called held-out, independent or final.

## Pose arms

```text
arm                                 PoseCov      R    Yaw    t cm    IoU3D   ADDsym
───────────────────────────────────────────────────────────────────────────────────
S0 YOLO current                       1.000   2.26   1.23    7.90   0.6032   0.4285
S1 Point translation control          1.000   2.26   1.23    7.90   0.6032   0.4285
S2 Point+BBox translation             1.000   2.26   1.23    8.05   0.5816   0.4187
D0 Raw DOPE                           0.715   3.92   2.14    9.72   0.5222   0.3724
S3 BBox-gated DOPE                    0.715   3.92   2.14    9.72   0.5222   0.3724
S4 Square-context DOPE                0.693   4.23   1.94   12.21   0.3883   0.3079
S5 Line-R + Point/BBox-t             BLOCKED_INCOMPATIBLE_PROVENANCE
```

## DOPE localisation

```text
arm                      DetCov  corners   kp med   kp p90   gross20
────────────────────────────────────────────────────────────────────
D0 Raw DOPE               0.850     2232    11.42    75.84     0.271
S3 BBox-gated             0.850     2232    11.42    75.84     0.271
S4 Square-context         0.796     2135    13.13   196.48     0.402
```

S3 fallback frames 19 (0.060). S4 inverse-map parity passed at under 1e-4 px.

## Paired uncertainty

```text
contrast    metric             diff           frame 95% CI         session 95% CI
─────────────────────────────────────────────────────────────────────────────────
S1-S0       iou3d           +0.0000     [-0.0000, +0.0000]     [-0.0000, +0.0000]
S1-S0       add_sym_auc     +0.0000     [+0.0000, +0.0000]     [+0.0000, +0.0000]
S2-S0       iou3d           -0.0216     [-0.0387, +0.0071]     [-0.0339, +0.0206]
S2-S0       add_sym_auc     -0.0098     [-0.0179, -0.0022]     [-0.0233, +0.0098]
S3-D0       iou3d           +0.0000     [+0.0000, +0.0000]     [+0.0000, +0.0000]
S3-D0       add_sym_auc     +0.0000     [+0.0000, +0.0000]     [+0.0000, +0.0000]
S4-D0       iou3d           -0.0917     [-0.1936, -0.0078]     [-0.2120, -0.0073]  excludes 0
S4-D0       add_sym_auc     -0.0477     [-0.0937, -0.0011]     [-0.1043, -0.0066]  excludes 0
```
clusters: 13 sessions (12 for S4, which loses one to coverage)

## What each arm showed

**S1 is a no-op, and that is the point.** The frozen selector already runs SQPnP
and RefineLM, so the point objective sits at its optimum and re-solving it moves
nothing. Any S2 gain could therefore not have been attributed to the mere act of
re-optimising — the control does its job by being empty.

**S2 makes everything worse.** Translation 7.90 to 8.05 cm, IoU3D −0.0216, ADDsym
−0.0098. This is not even the trap the lock warned about, where an objective falls
while truth worsens: the predicted reprojection also worsens, 3.55 to 3.75 px. The
YOLO box and the keypoints disagree, and forcing the pose to satisfy both lands
further from the truth than satisfying the keypoints alone. The session-clustered
intervals span zero, so the degradation is not resolved — but there is no direction
here to promote.

**S3 changed nothing whatsoever.** The gate ran on every frame and altered zero
corner selections across 305 comparable frames; the numbers are identical to raw
DOPE to the last digit. DOPE's highest peak already lies inside the 1.25x square
context. Its error is not that it looks in the wrong part of the image — it is that
the peak is off within the object region, or missing. A spatial prior has no
purchase on that, and this is the most informative negative in the screen.

**S4 is clearly worse** and both its 6D contrasts exclude zero under session
clustering. The square crop removes the aspect-ratio distortion that sank the
earlier strip crop, and it still loses detection coverage and localisation, with
p90 going 75.8 to 196.5 px. Re-inference on a crop hurts for reasons beyond the
aspect ratio.

**S5 was blocked before it ran.** Every line artifact belongs to the synthetic
multihead populations and no adapter maps that model onto these frames.

## Promotion gate

```text
arm   verdict      dIoU3D   dADDsym  reason
────────────────────────────────────────────────────────────
S1    STOP        +0.0000   +0.0000  no primary gain
S2    STOP        -0.0216   -0.0098  no primary gain
S3    STOP        +0.0000   +0.0000  no primary gain
S4    STOP        -0.1339   -0.0646  no primary gain
S5    STOP        +0.0000   +0.0000  BLOCKED_INCOMPATIBLE_PROVENANCE
```

Gate: delta IoU3D >= +0.020 or delta ADDsym >= +0.020, with the other primary at or
above baseline, coverage within 0.01, and rotation and translation no more than 5
percent worse. No arm comes near it.

## Interpretation

**1. Does the YOLO box carry translation information the keypoints do not?** No. It
carries information that disagrees with them, and letting it vote moves the pose
away from truth on every metric including the reprojection.

**2. Does a bbox spatial prior improve DOPE localisation?** No, and not because it
was applied badly — it had literally nothing to change. DOPE already peaks inside
the object region; the residual error lives inside that region.

**3. Does splitting rotation and translation between line and bbox cues reduce the
old trade-off?** Unanswered. The line ingredients exist only on synthetic
populations, so the arm was blocked rather than faked.

**4. What is worth a short training run?** Nothing in this screen. Every promotable
path is closed, and the one negative worth carrying forward is S3: the DOPE failure
is not spatial gating but the precision of the peak within a region the model has
already found.

```text
FAST_6D_SCREEN = NO_PROMOTABLE_SIGNAL
no further architecture training is opened on the strength of this screen
```

`NEXT_ACTION = USER_REVIEW_FAST_6D_SCREEN`

