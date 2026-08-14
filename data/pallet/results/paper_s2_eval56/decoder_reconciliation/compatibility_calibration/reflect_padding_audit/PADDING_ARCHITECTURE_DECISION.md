# Architecture implication

The audit answers its four questions as follows.

**Q1 -- does padding bring the response back?**  Yes.  From 0 of 13 on every
count to 11 of 13 centroids, 10 of 13 R4 and 10 of 13 D0 poses.  The global
collapse is an input-boundary effect, not an inability of this architecture to
represent a truncated pallet.

**Q2 -- context continuation or scale?**  Scale and canvas margin.  Constant
grey, which adds no context whatsoever, is the **best** arm (11 centroids, 10 R4,
8 R6, corner median 8/8), and reflect is the worst (9, 6, 6, 3/8).  If the
recovery came from continuing the scene past the edge, that order would be
reversed.  Mirrored texture appears to be actively misleading, putting a false
second pallet edge outside the true one.

**Q3 -- are the two failures independent?**  Yes.  With padding, 11 of 13 frames
move from "no raw response" to "centroid destroyed by the deployment sigma = 3
blur" -- the exact failure the compatibility audit measured on the other 74
frames.  Truncation collapse and belief-bandwidth mismatch are separate
conditions and a deployable model must satisfy both.

**Q4 -- is truncation-aware supervision needed?**  Yes, and the evidence is
specific.  Recovered corners whose GT is inside the frame land at a 25-30px
median; recovered corners whose GT is off screen land at roughly **290px**.  The
network can be told a pallet is present and where its visible part is; it has no
notion of where an unobserved corner belongs, because nothing has ever supervised
that.

## Direction: truncation-aware training

```
keep the 9-channel heatmap representation
supervise on real frame-edge truncation, not only on padded crops
give off-screen corners an explicit validity treatment rather than an
  unanchored target
add a pallet-presence objective so partial views keep a global response
```

## Explicitly not the direction

- **Inference-only padded second pass.**  Tempting, since it rescues 10 of 13,
  but it costs the healthy frames: C13 reprojection worsens 7-17%, worsened
  outnumbers improved on every arm, and 3-4 frames regress by 10px or more.  A
  conditional trigger would be needed and the condition cannot be GT.  It stays a
  fallback candidate, not a fix.
- **Centroid-only head or role-specific belief width.**  Ruled out for these 13
  frames by the previous audit and not revived here.
- **A global-context architecture** (Truncation-Aware Context DOPE) is not yet
  justified: padding shows the existing architecture *can* respond to these
  frames, so the missing piece is supervision, not receptive field.  It stays on
  the list only if truncation-aware supervision fails.
