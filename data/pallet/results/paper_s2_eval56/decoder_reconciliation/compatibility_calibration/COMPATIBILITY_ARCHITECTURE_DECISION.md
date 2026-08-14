# Architecture implication

Of the five cases the plan allows, this is **CASE 4**: no sigma passes the N87
gate, so the config-only rescue is rejected and the mismatch is between what the
model outputs and what the decoder's bandwidth assumes.

CASE 5 partially applies as well -- corner and centroid do have different
required widths (2.0 against 2.5) -- but it is not the reason the rescue failed.
The rescue failed because 13 of 87 frames carry essentially no centroid response
at any smoothing, and because ep57's centroid is 0.41 sigma too narrow even
where it does respond.

## Direction

**role-specific target width**, as the minimal change:

```
keep the 9-channel output and every decoder parameter
supervise corner channels 0-7 at their current narrow target   sigma ~ 2.0
supervise centroid channel 8 at a wider target                 sigma >= 2.5
```

This is the smallest edit that satisfies both measured minima without paying
corner localisation for the centroid's margin.  It needs no new head, no change
to the deployment code, and no change to any threshold.

**Dual-Bandwidth DOPE** -- a separate wider objectness head beside the narrow
corner head -- is the heavier variant.  It buys the freedom to give the
centroid a different receptive field and loss as well as a different width, and
it would be the right move only if a role-specific target alone leaves the
13 no-response frames unfixed.  Nothing measured here yet distinguishes the two,
because the no-response frames are a detection failure and this audit cannot
tell whether a wider target would fix them.

## What is not supported

- Lowering any threshold.  Already rejected, with wood corners landing 402px
  from GT at 0.25.
- Amplitude boosting the centroid to clear the gate.  That would move the peak
  over 0.30 without making the response any wider, and the 11x11 average would
  still read a narrow blob through a window sized for a wide one.
- Adopting the challenge checkpoints' configuration wholesale.  They clear the
  gate because they were trained at sigma 4; copying their config to ep57 is
  what this audit just measured failing.
