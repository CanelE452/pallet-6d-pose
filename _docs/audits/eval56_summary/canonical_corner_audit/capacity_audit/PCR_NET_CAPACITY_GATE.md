# Frozen-feature capacity gate

Zero main-model training.  The only fitting is a fixed logistic probe
(StandardScaler + LogisticRegression, C=1.0, balanced, max_iter 2000, seed 1),
grouped by frame so no frame appears on both sides of a split, with an assert
enforcing it.  **Probe scores are an upper bound on information present, not a
prediction of model performance.**

Feature taps confirmed at runtime, not hardcoded:

```
F100 = vgg[17] ReLU        256 x 100 x 100
F50  = vgg[26] = net.vgg() 128 x 50  x 50
C0 = F50 3x3 mean|max   C1 = F100 5x5 mean|max   C2 = C0+C1   C3 = C2 + GAP(F50)
```

27,395 samples from 101 canonical frames, 778 positives.  Negatives are the
fixed set defined before looking at any result: wrong top-1, top-5 maxima beyond
20px, a 3-radius 8-angle ring, and other corners' GT positions.

## Near — HCRM

```
feat   eval56    wood      cross-set mean
C0     0.8680    0.8231    0.7770
C1     0.8465    0.7856    0.5987
C2     0.8904    0.8457    0.6791
C3     0.8958    0.8415    0.6951
```

```
gate 1  C1 or C2 >= 0.75 on both sets           PASS  (C2: 0.8904 / 0.8457)
gate 2  cross-set mean >= 0.70                  FAIL  (best is C0 at 0.7770;
                                                       C2 0.6791, C3 0.6951)
gate 3  C1 or C2 beats C0 by >= 0.08 on MISSED  FAIL  (C1 is *worse* than C0 on
                                                       both sets: -0.02, -0.04)
```

**HCRM_CAPACITY: FAIL.**  The decisive number is gate 3.  The whole premise was
that the high-resolution F100 retains near-corner evidence that F50 has lost.
It does not: F100 alone is **worse** than F50 alone (0.8465 vs 0.8680 on eval56,
0.7856 vs 0.8231 on wood), and it also transfers far worse across pallets
(0.5987 against 0.7770).  Combining them helps a little within a set (C2 0.8904)
but that gain does not survive the cross-set test.

Near-corner information is separable within a set at 0.87-0.89, so it is not
absent -- but it lives in F50, not in the high-resolution tap, and it is
partly pallet-specific.

## Far — RCIM

```
GT vs wrong top-1
feat   eval56    wood      cross-set mean
C0     0.6551    0.6490
C1     0.7245    0.5724
C2     0.7032    0.5951    0.4920
C3     0.7533    0.6053    0.5083

four-way corner identity at GT locations (chance 0.25)
feat   eval56 macro-F1   wood macro-F1
C0     0.8882            0.8933
C2     0.9335            0.9054
C3     0.9387            0.9222
```

```
gate 1  C2 or C3 pairwise >= 0.75 on both sets   FAIL  (wood 0.6053)
gate 2  cross-set pairwise mean >= 0.70          FAIL  (0.5083, near chance)
gate 3  four-way identity macro-F1 >= 0.55       PASS  (0.92-0.94)
gate 4  C3 beats C0 by >= 0.08 pairwise          PASS on eval56 (+0.098),
                                                 FAIL on wood (-0.044)
```

**RCIM_CAPACITY: FAIL**, and the split between the two tasks is the informative
part.

Corner **identity** is highly separable: a linear probe on frozen features tells
corners 4, 5, 6 and 7 apart at macro-F1 0.92-0.94 with no coordinate input, and
it transfers to an unseen pallet.  So the features do encode which far corner is
which.

What they do not encode is **where that corner is** as against where the model
currently peaks.  Discriminating the GT location from the wrong top-1 runs at
0.60-0.75 within a set and **0.51 across sets, indistinguishable from chance**.
The wrong peak looks like the right corner to these features.

That is the precise failure: the far problem is not identity confusion, it is
that the feature at the wrong location is not separable from the feature at the
right one.

## Decision

```
HCRM_CAPACITY_PASS   NO
RCIM_CAPACITY_PASS   NO
LINE_CAPACITY        NOT RUN
```

**ENCODER_REPRESENTATION_CHANGE_REQUIRED.**

Neither proposed head has the information it would need.  Adding a
high-resolution branch cannot recover near corners that F100 represents worse
than F50 already does, and a role-conditioned head cannot separate a wrong far
peak from a right one when that separation is at chance across pallets.

## Not run

Phase H (structural line capacity and oracle discrimination), Phase I (per-corner
feature taxonomy), the old PPD line diagnostic, figures, and the full test list.
The line branch is therefore **undecided, not rejected** -- but note that with
both local gates failing, a line branch would be feeding structural context into
a representation that cannot localise, which is the same ordering problem the
previous audit hit.

Final-test: UNOPENED.  Main model training steps 0, main optimizers 0, no
checkpoint modified.  Probe coefficients were not saved.
