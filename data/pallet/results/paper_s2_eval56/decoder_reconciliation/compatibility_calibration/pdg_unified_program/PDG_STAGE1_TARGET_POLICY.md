# Stage-1 target policy

```
corner channels 0-7    Gaussian sigma 2.0 belief cells
centroid channel 8     Gaussian sigma 2.5 belief cells
```

The width split is the deployment bandwidth contract, not a truncation fix, and
is not claimed as one.

## Off-screen corners

```
projected centre outside the output frame
  -> no positive heatmap target
  -> belief loss mask 0
  -> affinity loss mask 0 on both of its channels (2i, 2i+1)
  -> visibility label off_screen
  -> no border clamp, no sentinel, no truncated Gaussian
```

A clamped target teaches a peak at the edge and a sentinel teaches a peak at a
fixed pixel; both are worse than teaching nothing, because the assembly step can
drop a channel it was told nothing about but cannot un-learn a false position.
The padding audit's off-screen corners landing ~290px from GT is what the
current unanchored channels produce.

## In-frame occluded corners

Occlusion does not remove the target.  A corner behind the cone still has a real
image position, so it keeps its amodal Gaussian, keeps its loss mask, and is
labelled `occluded` rather than `off_screen`.  Only the frame boundary removes a
target.

## Validity source

Validity comes from the transformed coordinate combined with the loader's own
channel mask, never from "did the Gaussian come out all-zero".  A channel the
renderer never produced gets no target **and no visibility label**; a channel
that exists but left the frame gets no target **and** an `off_screen` label.
Collapsing those two cases is what an all-zero test would do.

## Palletness

Projected cuboid convex hull clipped to the frame, at belief resolution, not the
fork-slot instance mask.  The slots put genuine cuboid edges outside the mask,
so masking would delete structure that is part of the pallet.
