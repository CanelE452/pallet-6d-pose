# Decision

**The existing verdicts are decoder-invariant across every decoder that can be
evaluated, and the deployment decoder cannot run this checkpoint at all.**

## The three questions

**Q1 -- does D0's far-stage bias, stagewise trade-off and PFDR failure
reproduce under D2?**  Yes for everything measured at the corner, no for
everything measured by detection count.  E2 and N2 cut far error by 15-19% on
both decoders; the corner improvement fails to reach the pose on both; S1's
reprojection gain and C1's pose damage keep their sign.  But N3's PnP 50 -> 52
becomes 46 -> 46, C1's PnP 50 -> 55 becomes 46 -> 43 (sign reversed), and S1's
+30% near regression becomes a small improvement.  Every arm still REJECTs on
both decoders and both sets.

**Q2 -- do P1's coordinates convert to pose gains after affinity grouping?**
Unanswerable, and not because grouping destroyed them.  The deployment path
never builds an object: `find_objects` requires a centroid peak above 0.30 on
the sigma = 3 smoothed map, and ep57's centroid peaks fall from a raw 0.856 to
0.281 under that blur.  0 of 56 and 0 of 45 frames clear it.  Coordinate
extraction loss 0, association loss 0, selection loss 0 -- the entire loss is at
object construction.

**Q3 -- does any arm flip to PASS on the real deployment path?**  No.  There is
no flip candidate, and `figures/verdict_flip_candidates.txt` records NONE.

## Conclusion

Of the five permitted conclusions this is **(5) deployment path / config
mismatch -- BLOCKED for the P2 column**, combined with **(1) the existing
verdicts are decoder-invariant** on the two columns that could be evaluated.

The mismatch is specific and checkable: `challenge/config/task.yaml` and the
forklift FSM both use `sigma = 3` with `thresh_map = 0.30`, which is tuned for
the wide-belief challenge models.  Run through the identical wrapper,
`challenge0123` keeps 0.61 and `challengenight` 0.54 after the same blur and
decode normally; ep57 keeps 0.27 and decodes to nothing.  Per this audit's
rules the sigma and the thresholds were not adjusted to make ep57 pass.

## Per-arm

```
B0   baseline.  P0 and P1 reproduce the recorded numbers exactly; P2 empty.
E2   REJECT on P0 and P1, both sets.  INCONCLUSIVE on P2 (no pose).
     Far improvement is real and decoder-invariant; it does not reach the pose
     on either decoder.
S1   REJECT on P0 and P1, both sets.  INCONCLUSIVE on P2.
     Its original blocker (near +30%) is D0-specific, but it fails on other
     conditions under both decoders, so the verdict is unchanged.
C1   REJECT on P0 and P1, both sets.  REJECT on wood P2 (one garbage pose).
     Its PnP gain reverses sign under D2.
N2   REJECT on P0 and P1, both sets.  INCONCLUSIVE on P2.
N3   negative control.  Its PnP gain does not exist under D2, which agrees with
     the previous audit's finding that the gain was detection recall of
     mediocre corners rather than a pose effect.
```

**base ep57 architecture unchanged.**

## Next admissible experiment

The finding that matters is not about any arm.  It is that the paper checkpoint
and the project's deployment decoder are incompatible as configured, and that
no evaluation in this programme has ever tested the pose the forklift would
actually receive.  The admissible next step is a decoder-compatibility
experiment, run before any further architecture work:

1. measure the belief blob width ep57 produces against what `sigma = 3` and an
   11x11 average assume, on both sets;
2. decide, with the thresholds fixed in advance, whether the deployment config
   should be re-derived for a narrow-belief model or whether the paper track
   should train against the deployment target width;
3. only then re-run this reconciliation, because until P2 produces objects the
   deployment column of every past and future verdict stays unmeasured.

Nothing here licenses changing sigma or the thresholds to make the current
numbers look better; that would be tuning against the evaluation.
