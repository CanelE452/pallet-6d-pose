# Counterfactual upper bounds

Only channel 8 is ever replaced; the corner maps stay exactly as predicted, and
the deployment decoder runs at its own sigma = 3 with every threshold fixed.
13 frames.

```
                arm  objects built  assoc corners (med)  PnP solved  live gate  catastrophic
────────────────────────────────────────────────────────────────────────────────────────────
               BASE              0                  0.0           0          0             0
  U1_width_only_s25              0                  0.0           0          0             0
  U2_amplitude_only             11                  1.0           0          0             0
    U0_gt_ideal_s25             13                  1.0           0          0             0
U3_gt_ideal_full_p2             13                  1.0           0          0             0
```

```
U1  ideal Gaussian sigma 2.5 at the predicted argmax, predicted amplitude kept
U2  the predicted map rescaled to peak 1.0, width unchanged
U0  ideal Gaussian sigma 2.5 at the GT centroid, peak 1.0          (oracle)
U3  the same input as U0, read through the full deployment pipeline (oracle)
```

**Width alone does nothing.**  U1 builds 0 objects, because widening a blob
whose amplitude is 0.04 leaves it at 0.02 after the deployment blur.

**Amplitude alone builds objects but reaches no pose.**  U2 builds 11 of 13 and
solves 0.

**The oracle centroid builds all 13 and still solves 0.**  With a perfect
centroid at the true position, the deployment path constructs an object on every
frame and then finds a median of **1** corner to associate with it -- three short
of the four correspondences PnP needs.  Nothing downstream is reachable because
the corner candidates do not exist.

## What U1 does and does not bound

U1 keeps the predicted amplitude, so it is a **lower** bound on what retraining
at a wider target would give: a genuine wider-target model would produce both a
wider blob and a normal amplitude.  U2 brackets the other side by giving
amplitude without width.  The pair is the honest read: amplitude is what builds
the object, and neither amplitude nor width brings the corners back.
