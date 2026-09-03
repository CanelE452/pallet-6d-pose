# Figure plan

Four figures. Every panel is built from artifacts that already exist. **No new
inference, no new training, no new evaluation pass.** If a panel cannot be built
from stored artifacts, the panel is cut, not regenerated.

## Figure 1 — Pipeline

A single schematic, no numbers.

```text
synthetic rendering
   exact geometric labels
        |
        v
   source model R0  (no real image, no real pose label)
        |
        +--------------------------------+
        |                                |
   unlabeled target RGB             evaluation
        |                                |
   static teacher inference             three separate axes:
        |                                  detection coverage
   pseudo box + pseudo 9 keypoints         confidence ranking
        |                                  keypoint localisation
   selection
     confidence pre-filter
     projective consistency
     equivariant consistency
        |
        v
   exposure-matched student
```

The important visual property: the detection branch and the keypoint branch are
drawn **separately** all the way to evaluation. A reader should be able to see,
from the figure alone, that the paper measures them apart.

## Figure 2 — The main trade-off

The paper's central claim in one panel.

```text
x-axis   2D keypoint median error [px] (original-image, paired population)
         -> right is worse
y-axis   nighttime detection coverage
         -> up is better

points   R0            synthetic-only
         R0-CONT       synthetic-replay control
         R1 naive
         R2 confidence
         R3 + reprojection
         R4 + keypoint removal
         R5 full consistency filter
```

The expected shape is that adapted arms sit **up and to the right** of R0: better
detection, worse localisation. That is the finding, and it should be legible without
reading the caption.

Draw R0 with a distinct marker and a vertical reference line at its localisation
value, so "no arm is to the left of R0" is visible at a glance.

Caption must state that the horizontal axis is raw 2D pixel error on the paired
population, and that PAPER_EVAL is a development population, not a sealed test set.

Optional second panel with the same axes but ranking (FPR95, lower is better) on the
y-axis — this is where the full filter is the best arm, and the contrast with the
detection panel is informative.

Source: `data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json`.

## Figure 3 — Pseudo-label quality against retention

Built entirely from the existing separability analysis. **No new computation.**

```text
x-axis   retention (fraction of teacher predictions kept)
y-axis   pseudo-label 2D keypoint quality of what is kept
curves   confidence only
         + projective consistency
         + keypoint-removal consistency
         + equivariant consistency (full)
```

The point of the figure is that the curves **do** separate — the reliability signals
are not random — which makes the downstream null result more interesting, not less.

Caption must label this a post-hoc diagnostic computed against evaluation GT, and
state that it measures label quality, not student quality.

Source: `_docs/archive/paper_pre_final_20260903/diagnostics/FILTER_SEPARABILITY.md` and its backing JSON.

## Figure 4 — Failure modes

Four example frames, reusing stored contact sheets. If a needed panel has no stored
overlay, drop that panel rather than running inference.

```text
(a) correct prediction                    the normal case
(b) axis-permuted near-square view        corners in place, indices rotated 90 deg
(c) nighttime or occluded failure         the hardest acquisition condition
(d) high-confidence wrong pseudo-label    box right, corners wrong, confidence high
```

Panel (d) is the one that motivates the entire selection study: it is the label that
confidence filtering keeps and that training reinforces. It should be the visual
anchor of the figure.

Panel (b) needs an overlay that shows predicted index labels, not just points —
otherwise the failure is invisible, since the corner *positions* are correct and only
the *assignment* is wrong. Judge such frames by maximum 2D keypoint error, not the
median: the error distribution is bimodal and the median hides it.

Sources: existing contact sheets under the V1 and V2 diagnostic outputs.

## Rules for all figures

```text
no figure introduces a number that is not in a table
every caption names the population and its size
every caption naming a development-tier result says so
axis labels carry units; "px" means original-image pixels
colour is not the only channel distinguishing arms
```
