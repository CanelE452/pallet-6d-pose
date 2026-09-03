# Contributions

Three. No fourth is claimed, because no fourth is supported.

## C1 — An adaptation protocol that measures detection, ranking, and localisation separately

Most synthetic-to-real adaptation studies report an aggregate detection or pose
score. That aggregate cannot distinguish *finding more pallets* from *placing their
corners more precisely*, and in this study the two move in opposite directions.

We define a protocol in which target-domain adaptation is evaluated along three
axes that are never collapsed into one number:

```text
detection coverage       is the pallet found at all, per acquisition condition
confidence ranking       are true positives ranked above negatives
keypoint localisation    how far are the supervised corners, in original-image pixels
```

The separation is what makes the paper's main result statable at all.

## C2 — An exposure-matched comparison of pseudo-label selection rules

Five selection rules are compared under an identical training budget: the same
initialisation checkpoint, the same number of optimiser updates, the same number of
pseudo-label exposures and synthetic-replay exposures, the same learning rate, batch
size, augmentation, and seed. Only the selection rule changes.

```text
naive                   all teacher predictions above the detection floor
confidence              box confidence and keypoint validity
+ projective            reprojection consistency
+ removal               single-keypoint-removal reprojection consistency
+ equivariant           horizontal-flip keypoint consistency
```

A synthetic-replay-only control arm isolates how much of any change is caused by
additional optimisation alone rather than by adaptation to real images.

Exposure matching matters because unequal pseudo-label counts otherwise confound
selection quality with training quantity — a stricter filter yields fewer labels,
and fewer labels change the optimisation, not only the supervision.

## C3 — Evidence that better pseudo-label selection need not produce a better student

Improved selection improves the labels the student sees. It does not follow that the
student improves. We show this directly and then isolate candidate mechanisms:

```text
mechanism probed                       what it separates
────────────────────────────────────────────────────────────────────────────────
per-keypoint masking                   frame-level trust vs keypoint-level trust
true-ignore at the loss                masking semantics vs supervision semantics
geometry repair of weak keypoints      how much of the error is reconstructible
reliability weighting                  label purity vs downstream localisation
multi-view teacher consensus           observation count vs teacher quality
```

Each probe is a development experiment, not an independent confirmation, and is
labelled as such throughout the paper. Their collective value is that they rule
mechanisms *out*: the limitation is not one filter implementation, not the masking
semantics, and not the number of observations pooled.

## What is deliberately not claimed as a contribution

```text
not claimed   a method that improves keypoint localisation
not claimed   a method that improves 6D pose accuracy
not claimed   a geometry filter with a demonstrated downstream gain
not claimed   generalisation to pallets outside the studied category
not claimed   a new architecture
not claimed   a new dataset release
```

The synthetic generation pipeline and evaluation harness are described for
reproducibility, not offered as a contribution.
