# Results section story

Order matters. The negative result appears in 5.3, before the ablation that
explores it and before the diagnostics that explain it. It is not deferred.

## 4.1 The synthetic-only baseline

**Question: how strong is R0 without any real supervision? (Q1)**

```text
detection ALL      0.975   (n = 319)
detection daytime  1.000   (n = 70)
AUROC              0.9921  (319 positive vs 2,689 real negatives)
box AP50           0.9363
box AP50-95        0.7688
pooled kp median   6.616 px  (2D keypoint layer)
```

The point of opening here is that the baseline is not weak. A study that finds
adaptation does not improve localisation is only interesting if the starting point
is strong, and it is: a model that has never seen a real image detects almost every
real pallet and separates positives from 2,689 real negatives at 0.9921 AUROC.

State the failure profile immediately, because it motivates everything after:
17.2 percent of supervised keypoints are more than 20 px from truth, and 30.2
percent in daytime. R0 is strong at finding pallets and imprecise at placing corners.

Reference row: the same-data DOPE control sits at 10.916 px and 0.737 detection,
with the box-derivation and score-scale asymmetries footnoted.

## 4.2 Target-domain self-training

**Question: what does adaptation do to detection? (Q2)**

```text
                       night det   AUROC    FPR95
R0                       0.840     0.9921   0.0417
naive                    0.960     0.9913   0.0558
confidence               0.980     0.9923   0.0469
full consistency filter  0.960     0.9953   0.0283
```

Two things must be said plainly and in this order.

First, **nighttime detection improves**, from 0.840 to 0.960 or above for every
adapted arm. Second, **the improvement is not the filter's**: naive self-training
reaches 0.960 unaided and confidence-only reaches 0.980. Any sentence attributing
the detection gain to geometric consistency is unsupported.

Report the uncertainty in the same breath: the overall detection difference between
R0 and the full filter has p_better = 0.121 at frame level and 0.244 clustered by
session. The nighttime subgroup is N = 50 and plastic-only.

Where the full filter **is** best is ranking. It has the highest AUROC and the
lowest FPR95 of all seven arms, and nighttime FPR95 falls from 0.1949 to 0.0588.
If the paper makes a positive claim about the proposed filter, this is it.

## 4.3 Fine 2D keypoint localisation

**Question: does the same adaptation improve 2D keypoint placement in the image
plane? (Q3)**

Define the metric on first use: *2D keypoint localisation error is the Euclidean
distance between the predicted and the annotated keypoint in original-image pixels,
pooled over supervised keypoints of matched detections.* It is the keypoint layer of
the frozen metric contract, not a pose metric.

**No.**

```text
                       pooled kp median px   gross20
R0                            6.616          0.172
synthetic-replay control      6.911          0.182
naive                         7.120          0.180
confidence                    7.037          0.194
+ reprojection                7.044          0.194
+ keypoint removal            6.999          0.194
full filter                   7.210          0.197
```

No arm beats R0. The best adapted arm, 6.999, is still above the baseline. The
direction is a small degradation, supported at frame level (p_better 0.028) and not
at session level (0.065), and that caveat is stated rather than dropped.

The synthetic-replay control matters here: it also degrades, which shows that part
of the movement is additional optimisation rather than adaptation to real images —
but it does not account for all of it, and no arm recovers.

This subsection is the paper's central result and is written as such.

## 4.4 Pseudo-label selection ablation

**Question: do the consistency signals find better labels, and does that help? (Q4)**

Split the answer.

*Do the signals separate good labels from bad?* Yes, weakly. Frame-level combined
AUC is 0.8116 on a population where roughly half the frames contain a gross error;
corner-level combined AUC is 0.7259. The signals are informative and far from clean.
One number bounds what confidence filtering can contribute: the per-keypoint
confidence floor removes **zero** supervised corners.

*Does better selection produce a better student?* No. Adding reprojection, then
keypoint removal, then flip consistency does not move localisation off the baseline.

Keeping these two questions visibly separate is the point of Table 3. A paper that
reported only the first would misrepresent the result.

Label the separability analysis development evidence: it was computed after the fact
on a population already consumed by earlier arms.

## 4.5 Diagnostic analysis

**Question: what limits the benefit? (Q5)** All Tier B; all labelled as such.

**Selection population effects.** A rule that discards hard keypoints improves its
own score by shrinking its population. Every localisation comparison in this paper
is paired for that reason, and coverage is reported alongside. The teacher-consensus
probes show the size of the artefact directly: accepting only agreed keypoints drops
1,979 to 1,284 and looks like an improvement, while the paired comparison on the
same keypoints shows the tail getting worse.

**Semantic-axis ambiguity.** The catastrophic 2D keypoint errors are 90-degree index
permutations, not mislocalised points: the corners are in the right places and the
labels are rotated. They concentrate where the projection is near-square. Such
frames must be judged by maximum 2D keypoint error — the distribution is bimodal
and the median hides the failure entirely.

**Pseudo-label purity is not the binding constraint.** A GT-free reliability score
ranks frames at AUC 0.7625, above any single signal it combines, and measurably
cleans what the student sees: expected corner-gross falls from 0.208 to 0.182 and
expected median error from 20.0 to 18.2 px. Student localisation does not move. This
is the most direct evidence in the study for a teacher ceiling.

**Repair has almost nothing to act on.** Roughly one percent of supervised corners
are repair candidates, and the competing geometric hypotheses disagree precisely
there. No student was trained; the track stopped at the mechanism stage.

**More observations do not raise the ceiling.** Three teacher-consensus probes —
flip averaging, multi-resolution median, cross-checkpoint median — all move the same
way: median marginally better, tail clearly worse. Averaging pulls good predictions
toward bad ones exactly where the views disagree, which is where the hard keypoints
are.

## Writing rules for this section

```text
every subgroup number carries its N
every Tier B result is labelled development evidence in the sentence that uses it
raw pixel error and NME are never mixed in one column
NME is labelled a post-hoc scale-normalised diagnostic wherever it appears
absolute px is never compared across subgroups to rank condition difficulty
paired comparisons say they are paired; coverage is reported next to them
6D pose quantities appear only in the pose table and its subgroup tables, always
  against the geometry-reconstructed reference, never as an improvement claim
"held-out" appears nowhere — PAPER_EVAL role is DEV
```
