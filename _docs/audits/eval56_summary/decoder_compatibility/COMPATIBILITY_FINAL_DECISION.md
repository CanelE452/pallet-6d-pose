# Decision

**CONFIG_ONLY_FAIL -- target/output bandwidth mismatch.**

No smoothing sigma makes ep57 usable by the deployment decoder while every
threshold stays fixed.  The gate needs centroid survival on 83 of 87 frames;
the best any sigma reaches is 74, and that is at sigma = 0, where there is no
smoothing left to remove.  Thirteen frames carry a raw centroid peak at or below
0.30 -- nine of them below 0.03 -- so the ceiling is the model's response, not
the decoder's blur.

## The four questions

**Q1 -- does tuning sigma alone make P2 work?**  No.  Centroid survival peaks at
74/87 (sigma 0-0.5), PnP candidates at 70/87, and the reprojection condition
fails at every sigma by 18-22%.  Sigma does move the needle -- survival goes
11/87 at the deployment sigma of 3 to 74/87 at 0 -- but not to the gate.

**Q2 -- does an N87-fixed sigma hold on eval56 and wood?**  Not answerable and
not attempted: no sigma was selected, so the holdouts were not spent.  Recorded
separately: N87 and eval56 share 12 frames, so eval56 would have been a
partially contaminated holdout.

**Q3 -- do any existing arm verdicts flip on a compatible P2?**  Not reached.
The P2 column stays unmeasured and every previous verdict stands.

**Q4 -- is the decoder-aware target/head hypothesis supported?**  Partly, and
the honest form is narrower than the question.  Measured: corner channels need
a target sigma >= 2.0 and the centroid needs >= 2.5, so the roles do have
different minima, and no model in the repository separates them (centroid/corner
width ratio 0.98-1.02 in ep57 and both challenge checkpoints).  Also measured:
a single shared target of 2.5 satisfies both, so separation is **not strictly
required** by these criteria.  What makes separation preferable is a cost this
audit does not measure -- corner localisation accuracy, which widening the
corner target is known to hurt.

## Architecture direction

**role-specific target width** (corner ~2.0, centroid >= 2.5), keeping the
9-channel output, every decoder parameter and every threshold unchanged.
Dual-Bandwidth DOPE stays a candidate for the case where a wider centroid target
still leaves the no-response frames unfixed.

## Next admissible experiment

1. Decide whether the 13 no-response frames are a width problem or a detection
   problem, without training: measure ep57's centroid channel on those frames
   against the corner channels on the same frames.  If the corners respond and
   the centroid does not, a wider target will not fix them and the objectness
   head is the real question.
2. Only then, a role-specific target-width run with the widths fixed in advance
   from `COMPATIBILITY_TARGET_WIDTH.md` (corner 2.0, centroid 2.5), scored on
   both the mechanism decoder and the deployment decoder so the two cannot
   diverge again.
3. Keep the deployment sigma and every threshold fixed throughout.  The lesson
   of this audit is that the model must meet the decoder's contract, not the
   other way round.
