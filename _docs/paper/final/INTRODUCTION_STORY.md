# Introduction story

Seven paragraphs. The argument must arrive at the detection/localisation
distinction, because that is the paper's result.

## P1 — Why pallet perception

Automated forklifts and AGVs must locate a pallet precisely enough to insert forks
into its pockets. This is a sensing problem with a tight tolerance: finding the
pallet is necessary but not sufficient, because insertion depends on the pallet's
orientation and the position of its structural edges, not merely on its presence.

## P2 — Why real supervision is expensive

Supervising this task on real warehouse imagery requires 6D pose or dense keypoint
annotation. Both are slow, need physical measurement or fiducial rigs, and must be
repeated whenever the pallet type, the camera, or the lighting changes. Annotation
cost, not model capacity, is the practical constraint.

## P3 — Why synthetic rendering, and what it leaves behind

Rendering supplies exact geometric labels at negligible marginal cost and removes
the annotation bottleneck entirely. What it does not remove is the synthetic-to-real
gap: appearance, sensor noise, illumination, and the distribution of viewpoints an
actual vehicle produces all differ from what was rendered.

## P4 — Why self-training, and the specific difficulty here

Self-training on unlabeled target images is the natural response, since target RGB
is free to collect. But pallet keypoint estimation has a difficulty that generic
detection self-training does not:

> a pseudo-label can be entirely good enough as a detection while several of its
> geometric keypoints are wrong.

The box is right, the confidence is high, and the corners are misplaced. Ordinary
confidence filtering keeps exactly these labels, and training on them reinforces the
error. This is why geometric reliability tests — reprojection consistency and
equivariance under horizontal flip — are the natural candidates for selecting
pseudo-labels here, and why we test them.

## P5 — The question this paper actually asks

The interesting question is therefore not

> does self-training work?

but

> **which parts of pallet perception improve under self-training, and can geometric
> consistency prevent pose-keypoint errors from being reinforced?**

Answering it requires that detection and localisation be measured separately. If
they are collapsed into a single score, an improvement in one can hide the absence
of improvement in the other.

## P6 — How we answer it

We compare selection rules under an exposure-matched protocol: identical
initialisation, identical optimiser budget, identical numbers of pseudo-label and
synthetic-replay exposures, identical augmentation and seed. Only the selection rule
changes. A synthetic-replay-only control isolates the effect of additional
optimisation. Every threshold and the checkpoint-selection rule are fixed before any
result on the evaluation population is observed.

## P7 — The main finding

Adaptation to unlabeled target images substantially improves detection coverage,
most visibly under nighttime acquisition, and improves confidence-based separation
of true positives from negatives. Under the same protocol it does **not** improve
fine keypoint localisation over the synthetic-only model, and adding projective and
equivariant consistency to the selection rule does not recover a localisation gain.

That asymmetry is the paper's central result. Post-hoc diagnostics — reported as
diagnostics — indicate the limitation is not attributable to a single filter
implementation: pseudo-label purity can be improved without moving the student, and
pooling more observations from the same teacher improves the median while worsening
the tail.

## Tone rules

```text
do        state the negative result in P7, in the introduction, unhedged
do        keep "detection improves / localisation does not" as one sentence
don't     promise a method that fixes it
don't     describe prior work as having failed where we succeeded
don't     use "label-free" — manual real annotations exist for evaluation
```

The honest phrasing for the training condition is **"without manual target-domain
pose labels during training."**
