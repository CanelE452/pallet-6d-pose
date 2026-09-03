# Discussion and limitations

The discussion is where this paper earns its place. The experiments are careful and
the headline result is negative; what makes it publishable is explaining *why* the
negative result is informative rather than a failed attempt.

## 6.1 Detection and localisation are different objectives

Self-training supplies a signal about **where a pallet is**, and that signal is
genuinely new to the model: a nighttime image the source model never rendered
teaches it that a dim, low-contrast rectangle is a pallet. Nighttime detection rises
from 0.840 to 0.960, and confidence-based separation from negatives improves.

It supplies no new signal about **where the corners are**. The pseudo-label's
corners are the teacher's own predictions. Training on them asks the student to
reproduce what the teacher already believed, including the 17.2 percent of corners
the teacher places more than 20 px from truth. Selection can decline to train on a
bad corner; it cannot produce a better one.

That asymmetry explains the whole pattern. A box only has to be approximately right
to count as a detection, so a noisy pseudo-box still carries usable information
about presence. A corner has to be right to be worth training on, and the teacher's
corners are exactly as good as the teacher.

## 6.2 Pseudo-label purity does not transfer to the student

This is the finding we consider most useful to other practitioners, because it
contradicts a natural assumption.

The reliability-weighting experiment improved the expected quality of the labels the
student saw — corner-gross from 0.208 to 0.182, expected median error from 20.0 to
18.2 px — while the labels themselves and their count were held identical; only how
often each was shown changed. Student localisation did not move.

If cleaner labels do not produce a better student, then label purity was not the
binding constraint. What remains binding is the accuracy the teacher can express at
all. Under single-frame adaptation from a static teacher, selection operates strictly
inside that envelope.

The practical implication is a measurement one: **report downstream student quality,
not filter precision.** A filter that improves pseudo-label precision has not
demonstrated a downstream gain, and the two are easy to conflate.

## 6.3 Consistency signals are useful, but not sufficient

The signals are not noise. Combined frame-level separability reaches AUC 0.8116 and
corner-level 0.7259, and the reliability score ranks above any single component it
uses. The full consistency filter also produces the study's clearest positive
result: the best AUROC and the lowest FPR95 of all seven arms, with nighttime FPR95
falling from 0.1949 to 0.0588.

So the honest reading is not "geometry filtering does not work." It is:

```text
geometric consistency is a real reliability signal
it improves confidence ranking
it does not, by itself, lift keypoint localisation past the teacher
```

One measurement sharpens this. The per-keypoint confidence floor removes **zero**
supervised corners — every corner clears it. Confidence gating, at the corner level,
is inert on this data. Geometric consistency is doing whatever work is being done,
and it is still not enough.

## 6.4 Near-square projections carry a 90-degree ambiguity

The catastrophic 2D keypoint errors are not mislocalised points. They are **index
permutations**: the corners are in the right places and the labels have rotated by
90 degrees. They concentrate where the projected pallet is close to square, which is
where the width and depth axes become visually interchangeable.

Two consequences are worth stating for readers building similar systems:

```text
a  such frames must be judged by maximum 2D keypoint error. The distribution is
   bimodal — a few corners are enormously wrong and the rest are fine — so the
   median reports a healthy frame.
b  the ambiguity is a property of the viewpoint, not of the model. No amount of
   pseudo-label selection resolves it, because the evidence needed to resolve it
   is not in the frame.
```

An ambiguity-aware development variant reduced the permutation rate from 0.047 to
0.041, and to 0.084 from 0.096 on the ambiguous subgroup. That is development
evidence, it is reported as such, and that variant is not the proposed method.

## 6.5 Scale sensitivity of the 2D metric

Raw image-plane pixel error depends on projected object size, so condition-to-
condition comparisons of absolute pixel error are potentially confounded by scale.
A pallet that fills the frame yields a larger absolute error than a distant one at
the same relative accuracy.

Our primary interpretation is therefore **model-to-model comparison within the same
subgroup**, never a ranking of subgroup difficulty by absolute pixel value. The
daytime median of 10.556 px against the nighttime 7.686 px does not establish that
daytime is the harder condition.

Post-hoc scale-normalised diagnostics were used only to test whether an observed
condition difference could be explained by scale. They are **not** substituted for
the frozen metric, and no result in this paper was re-reported under a different
metric after the fact.

## 6.6 Night and occlusion are the hardest conditions

R0's error concentrates predictably: daytime gross-error rate 0.302 against 0.119
at night, occlusion 0.243 against 0.123 clean, truncation 0.274. Adaptation helps
nighttime *detection* most, which is consistent with 6.1 — nighttime is where the
source model's presence signal was weakest, and where new real images add most.

## 6.7 What would actually be needed

Selection has been explored to the end of its usefulness on this population. Three
directions remain, and each supplies something selection cannot:

```text
a stronger source teacher      raises the envelope selection works inside.
                               Untested here: the capacity question is open, and
                               the no-train probes do not settle it.
temporal or multi-frame        supplies genuinely new observations rather than
supervision                    re-weighting existing ones. The consensus probes
                               show that pooling views of the *same* frame from
                               the *same* teacher does not help; different frames
                               are a different proposition.
limited real supervision       a small number of human-labelled real frames
                               replaces the teacher on exactly the corners the
                               teacher cannot place.
```

We do not claim any of these works. We claim the evidence points at them rather than
at further selection rules, because every mechanism internal to selection has been
probed and none moved the student.

## 7 Limitations

Stated plainly, not softened.

```text
development population
  PAPER_EVAL 319 has role DEV and held_out_final = false. It was consumed as a
  development population by every diagnostic track. No number in this paper is a
  held-out result.

no independent confirmation
  V2, V3, V4, V5, the separability analysis and all three teacher probes were
  designed after seeing results on this population. They support mechanism
  analysis and nothing more. For three of them — V4, the strong-teacher audit and
  the fast-teacher probes — the contract and the result were committed together,
  so their ordering cannot be established from version history at all.

pose metrics blocked
  POSE_METRICS_STATUS = BLOCKED. No claim about rotation, translation, yaw, ADD,
  ADD-S, 3D IoU or 6D pose AUC appears anywhere. The blocker is algorithmic: the
  best axis selector measured reaches 0.65 against a gate of 0.95, so further
  annotation would not open it.

2D is not 6D
  keypoint pixel error is not final pose accuracy. A study measuring 2D keypoints
  cannot claim a pose result, and does not.

constrained object category
  a small number of pallet types under one deployment setting. No claim is made
  about pallets outside the studied category, and no experiment supports one.

acquisition coverage
  daytime and nighttime dominate. The nighttime subgroup used for the headline
  comparison has N = 50 and contains plastic pallets only.

adaptation pool separation
  the adaptation pool contains no evaluation-session frame. This is a property of
  how the pool was built, and it is what makes the comparison meaningful.

wood readiness
  the wood object's canonical migration and symmetry contract are not frozen,
  which is one of the reasons pose metrics remain blocked.

single seed for most arms
  replicate runs exist for a subset only, and those replicates were introduced
  after the main results were seen. Effects smaller than the seed spread are not
  claimed.
```

## Tone rules

```text
do     explain why the negative result is informative
do     name what the study cannot settle, in the same paragraph as what it can
don't  end a limitation with a sentence that takes it back
don't  describe prior work as having failed where we succeeded
don't  promise the future directions will work
```
