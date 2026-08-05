# P1 -> P2: where the deployment path loses everything

```
   set  P1 coords  in a P2 object  dropped  objects built  centroid raw peak  after sigma=3  clears 0.30
────────────────────────────────────────────────────────────────────────────────────────────────────────
eval56        354               0      354              0             0.8561         0.2813         0/56
  wood        328               0      328              0             0.8917         0.2866         0/45
```

Every single P1 coordinate is dropped, and the loss is **not** spread across
the four stages the decomposition was meant to separate:

```
coordinate extraction loss   0   (P1 produced 354 / 328 coordinates)
affinity association loss    0   (no association was ever attempted)
object selection loss        0   (there was nothing to select between)
object construction loss     all (no centroid peak survives the smoothing)
PnP geometry loss            0   (PnP is never reached)
```

`find_objects` builds objects from peaks of the **Gaussian-smoothed** map that
exceed `thresh_map = 0.30` (`detector.py:684, 701`).  On ep57 the centroid
channel peaks at a raw 0.856 (eval56) and 0.892 (wood), but a sigma = 3 blur on
a 50x50 map spreads it down to a median 0.281 and 0.287 -- under the gate on
**0 of 56 and 0 of 45 frames**.  With no centroid peak, `all_peaks[-1]` is
empty, no object is constructed, corner association never runs, and
`find_object_poses` returns nothing.

## This is a model/config mismatch, not a wrapper bug

The same wrapper, same tensors path, same config, run on the challenge
checkpoints instead of ep57:

```
model                          centroid raw   after sigma=3   objects (6 frames)
ep57 paper stage B                   0.8520          0.2708             0
challenge0123 net_epoch_0060         0.9485          0.6093             6
challengenight net_epoch_0120        0.8378          0.5366             6
```

The challenge models -- the ones `challenge/config/task.yaml` actually points
at -- keep 0.54-0.61 after the same blur and decode normally.  ep57's belief
blobs are narrow enough that a sigma = 3 smoothing destroys them.  The
deployment sigma was tuned for a wide-target model; ep57 is not one.

Per the audit's own rules the sigma, the thresholds and the config are fixed
and were not adjusted to make this work.
