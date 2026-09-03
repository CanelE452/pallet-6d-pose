# Abstract draft

Every number below was checked cell-by-cell against
`data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json`. All ten match.
Traceable map: `generated/RESULT_SOURCE_MAP.json`.

## Abstract

> Accurate pallet pose perception is important for autonomous forklift alignment,
> yet obtaining precise pose annotations from real warehouse images is
> labor-intensive and difficult to scale. Synthetic rendering provides scalable
> geometric supervision, but models trained only on rendered images face a
> synthetic-to-real domain shift. We conduct a controlled study of target-domain
> self-training for a monocular RGB pallet keypoint estimator feeding a downstream
> Perspective-n-Point solver. The study considers a structurally constrained pallet
> family and uses no manually annotated target-domain pose labels during
> training. Under an exposure-matched protocol, all adaptation arms share the same
> source initialization, optimizer budget, pseudo-label exposure, synthetic replay,
> and augmentation; only the pseudo-label selection rule varies. We compare naive
> self-training, confidence filtering, standard reprojection consistency,
> single-keypoint-removal reprojection consistency, and horizontal-flip keypoint
> consistency. On an in-house development population of 319 real positive frames and
> 2,689 real negative frames, the synthetic-only model obtains 0.975 detection
> coverage. Self-training increases observed nighttime detection coverage from 0.840
> to 0.960 with naive selection and to 0.980 with confidence selection. The full
> consistency variant achieves the best observed ranking among the frozen arms, with
> an AUROC of 0.9953 and an FPR95 of 0.0283 against 0.9921 and 0.0417 for the
> synthetic-only model. However, none of the six continuation or adaptation arms
> improves fine 2D keypoint localisation: the pooled supervised keypoint median error
> is 6.616 pixels for the synthetic-only model against 7.210 for the full
> consistency variant. Carried through to the downstream pose, scored
> against a geometry-reconstructed 6D reference, no variant shows a
> session-cluster-resolved improvement over the synthetic-only baseline. Post-hoc
> diagnostics further show that measurably improving expected pseudo-label quality
> need not improve the student. Unlabeled target adaptation can therefore
> benefit detection coverage and confidence ranking without those gains transferring
> to fine 2D keypoint localisation or downstream 6D pose. The findings are
> consistent with a teacher-quality bottleneck and motivate stronger or multi-frame
> supervision for future synthetic-to-real pallet pose adaptation.

```text
word count   310
```

## Number check — all verified against the frozen artifact

```text
claim in abstract          value      artifact value     verdict
──────────────────────────────────────────────────────────────────
detection coverage         0.975      0.974922           MATCH
AUROC synthetic-only       0.9921     0.992131           MATCH
night detection R0         0.840      0.840000           MATCH
night detection naive      0.960      0.960000           MATCH
night detection confidence 0.980      0.980000           MATCH
AUROC full consistency     0.9953     0.995311           MATCH
FPR95 full consistency     0.0283     0.028263           MATCH
FPR95 synthetic-only       0.0417     0.041651           MATCH
keypoint median R0         6.616 px   6.615678           MATCH
keypoint median full       7.210 px   7.209888           MATCH

"none of the six ... arms improves"   6 of 6 arms are above R0's median   MATCH
   (R0-CONT, naive, confidence, +reprojection, +removal, +flip)

"no session-cluster-resolved         0 of 24 metric blocks exclude zero  MATCH
 improvement in 6D pose"               in the improvement direction
   source: data/pallet/results/paper_pose_metric_closure_v1/POSE_PAIRED_BOOTSTRAP.json
           (6 comparisons x 4 metrics; traced in PAPER_CANONICAL_NUMBER_SOURCES.json)
```

Source: `data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json`,
keys `models.<arm>.subgroups.{ALL,Nighttime}.{detection_rate_iou50, auroc, fpr95, corner_median_px}`.

## Phrasing decisions and why

```text
"We conduct a controlled study"        not "We propose [Name]". There is no method
                                       being proposed; the contribution is the
                                       controlled measurement.

"increases observed nighttime          not "substantially improves detection".
 detection coverage"                   The overall detection difference has
                                       p_better 0.121 (frame) / 0.244 (session).
                                       Nighttime is the subgroup where the
                                       movement is visible, and it is attributed
                                       to self-training as a whole — naive
                                       selection reaches 0.960 unaided.

"achieves the best observed ranking     not "improves ranking". A paired
 among the frozen arms"                 frame-level interval is now available and
                                        the AUROC contrast is positive
                                        (+0.00318, CI [+0.000092, +0.006898]),
                                        while the FPR95 contrast is unresolved.
                                        Session-clustered ranking uncertainty
                                        still cannot be estimated because the
                                        negative rows carry no session identifier,
                                        and the interval was computed after the
                                        point estimate was seen (Tier B). Best
                                        observed remains what the evidence
                                        supports; "statistically confirmed" or
                                        "significant ranking improvement" does not.
                                        (Historical: no interval existed at all
                                        when the original lock was written.)

"consistent with a teacher-quality      not "we show the teacher is the bottleneck".
 bottleneck"                            It is an interpretation that fits every
                                        diagnostic, not a measured quantity.

"single-keypoint-removal                never "leave-one-keypoint-out" or "LOO".
 reprojection consistency"

"horizontal-flip keypoint               never "flip reprojection consistency" —
 consistency"                           no reprojection is involved.

"structurally constrained pallet        no claim of generalisation to unseen or
 family"                                arbitrary pallets.

"no manually annotated target-domain    not "label-free". Manual real annotations
 pose labels during training"           exist and are used for evaluation.

"in-house development population"       states the population's role in the
                                        abstract itself. Nothing here is held-out.
```

## Deliberate omissions

```text
6D pose improvement                      the abstract reports that 6D was
                                         measured and that no arm shows a
                                         session-cluster-resolved gain. The gain
                                         itself is what is omitted, because there
                                         is none — 0 of 24 metric blocks resolve
                                         in the improvement direction
axis permutation 0.047 -> 0.041          Tier B development result, not an
                                         abstract number
indoor/outdoor independent adaptation    no such experiment exists
held-out pallet                          no such experiment exists
"our filter improves night detection"    naive reaches 0.960 unaided
```

## If the venue requires a shorter abstract

Cut in this order:

```text
1  the sentence listing the five compared selection rules
2  "Post-hoc diagnostics further show ..." (one sentence)
3  "The study considers a structurally constrained pallet family ..." — but only
   if the limitation is stated prominently elsewhere
```

Never cut the localisation negative result, and never cut the clause noting that
naive selection already reaches 0.960.
