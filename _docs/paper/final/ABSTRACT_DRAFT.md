# Abstract draft

Every number is Tier A and traceable through `generated/RESULT_SOURCE_MAP.json`.

## Draft

> Automated forklifts must localise a pallet precisely enough to insert forks into
> its pockets, but supervising 6D pose on real warehouse imagery is expensive and
> must be repeated whenever the pallet, camera, or lighting changes. Synthetic
> rendering removes that annotation cost while leaving a synthetic-to-real gap, and
> self-training on unlabeled target images is the natural way to close it. Pallet
> keypoint estimation, however, has a failure mode that generic detection
> self-training does not: a pseudo-label can be entirely adequate as a detection
> while several of its geometric keypoints are wrong, so confidence filtering
> retains exactly the labels that reinforce the error.
>
> We study this under a controlled, exposure-matched protocol in which every
> adaptation arm shares one initialisation, one optimiser budget, one number of
> pseudo-label and synthetic-replay exposures, and one seed, so that only the
> pseudo-label selection rule varies. We evaluate detection coverage, confidence
> ranking, and fine keypoint localisation separately rather than collapsing them
> into a single score, and we compare confidence filtering against projective
> consistency, single-keypoint-removal reprojection consistency, and horizontal-flip
> keypoint consistency.
>
> A source model trained only on rendered images, without any real pose label, is
> already a strong real-domain baseline: 0.975 detection and 0.9921 ranking AUROC on
> 319 real frames against 2,689 real negatives. Self-training on unlabeled target
> RGB raises nighttime detection from 0.840 to 0.960 — a gain that naive
> self-training achieves on its own — and adding the consistency filters reduces the
> false-positive rate at 95 percent recall from 0.0417 to 0.0283. Under the same
> protocol, none of the seven arms improves fine keypoint localisation over the
> synthetic-only model; median supervised keypoint error rises from 6.6 to 7.2
> pixels for the full filter, and the best adapted arm remains above the baseline.
>
> Post-hoc diagnostics indicate the limitation is not attributable to a single
> filtering implementation. A reliability score that measurably improves the quality
> of the labels the student sees leaves student localisation unchanged, and pooling
> additional views from the same teacher improves the median while worsening the
> tail. We conclude that under unlabeled single-frame adaptation, detection coverage
> and confidence ranking improve while fine keypoint localisation is bounded by the
> teacher, and we identify teacher quality and additional observational information,
> rather than better selection, as the directions that remain.

## Word budget

Roughly 330 words. If the venue requires less, cut in this order:

```text
1  the second half of paragraph 2 (the list of consistency signals)
2  the reliability-score sentence in paragraph 4
3  the "best adapted arm remains above the baseline" clause
```

Never cut: the localisation negative result, or the clause noting that naive
self-training already reaches 0.960.

## Number check

```text
0.975    R0 detection, ALL, n = 319
0.9921   R0 AUROC, 319 positive vs 2,689 negative
0.840    R0 nighttime detection, n = 50
0.960    naive self-training nighttime detection, n = 50
0.0417   R0 FPR95
0.0283   full filter FPR95
6.6      R0 keypoint median, original-image px (6.616)
7.2      full filter keypoint median (7.210)
```

## Deliberate omissions and why

```text
"our filter improves night detection"     naive reaches 0.960 unaided;
                                          confidence-only reaches 0.980
detection p_better = 0.121                too technical for an abstract, but the
                                          abstract must not therefore imply the
                                          detection gain is decisive — hence
                                          "raises nighttime detection", not
                                          "significantly improves detection"
axis permutation 0.047 -> 0.041           Tier B development result
any 6D pose quantity                      POSE_METRICS_STATUS = BLOCKED
"held-out"                                PAPER_EVAL role is DEV
"label-free"                              manual real annotations exist for
                                          evaluation; the filters also consume a
                                          dimension registry
```

## Phrasing rules applied

```text
"without any real pose label" describes source training — accurate, R0 saw none
"unlabeled target RGB" describes adaptation — accurate
"without manual target-domain pose labels during training" is the full form,
  used in the introduction where there is room for it
```
