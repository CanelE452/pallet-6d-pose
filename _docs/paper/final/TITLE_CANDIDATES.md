# Title candidates

Constraint: the title must not promise more than the evidence supports. The study
does **not** show improved keypoint localisation, and although 6D pose is now
measured and reported, no arm shows a session-cluster-resolved improvement in it.
Any title containing *improved*, *accurate*, or *robust ... pose* is excluded.

Colon-free titles are preferred.

## Candidates

```text
T1  Synthetic-to-Real Self-Training for Pallet Pose Keypoint Estimation

T2  What Improves and What Does Not in Synthetic-to-Real Pallet Keypoint Adaptation

T3  Separating Detection and Localisation in Self-Training for Pallet Pose Keypoints

T4  An Empirical Study of Pseudo-Label Selection for Pallet Pose Keypoint Estimation

T5  Unlabeled Target Adaptation for Pallet Detection and Pose Keypoints

T6  Projective and Equivariant Consistency Tests for Pallet Pseudo-Label Selection

T7  Selective Self-Training for Pallet Pose Keypoints under Warehouse Domain Shift
```

## Assessment

```text
id  strength                                    risk
──────────────────────────────────────────────────────────────────────────────────
T1  neutral, matches the study exactly          says nothing about the finding
T2  states the contribution honestly            reads slightly informal
T3  names the paper's central distinction       longer; "separating" needs the abstract
T4  accurate scope                              "empirical study" can read as low-novelty
T5  covers both branches                        does not signal the negative result
T6  names the actual method components          narrow; hides the main finding
T7  matches the original framing                "selective" implies a gain we cannot claim
```

## Recommended

```text
RECOMMENDED = T3
             "Separating Detection and Localisation in Self-Training
              for Pallet Pose Keypoints"
```

**Why.** The paper's single most defensible contribution is that detection coverage
and fine keypoint localisation respond *differently* to the same adaptation budget.
T3 puts that distinction in the title without claiming an improvement in either.
For an IEEE Sensors Journal readership — where the sensing pipeline and its
operating characteristics matter as much as a leaderboard number — a title that
names the measurement distinction is more informative than one that names the method.

**Second choice = T1** if a reviewer or editor prefers a conventional, method-naming
title; it is accurate and carries no overclaim, at the cost of saying nothing about
the result.

**Do not use.** Any variant containing *Improved 6D Pose*, *Accurate 6D Pose*,
*Robust Pose Improvement*, or *Label-Free*. The first three are contradicted by the
localisation result; the fourth is contradicted by the existence of manual real
annotations used for evaluation.
