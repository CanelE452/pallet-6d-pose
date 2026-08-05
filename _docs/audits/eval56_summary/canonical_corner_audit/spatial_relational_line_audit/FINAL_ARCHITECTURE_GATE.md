# Final capacity gate

Supersedes `FINAL_CAPACITY_GATE.md` in this directory for the far and line axes;
the near numbers there stand unchanged.  Zero main-model training, zero main
optimizers, no checkpoint touched.  Probes are diagnostic and never saved.

## Near — unchanged, provisional

```
D0 pooled F50 cross-set mean     0.8718
D2 spatial F50 cross-set mean    0.9425      +0.0707
D4 spatial multiscale            0.9454      +0.0736
gate 1 (>= 0.75 both sets)       PASS
gate 2 (cross-set >= 0.65/0.70)  PASS
gate 3 (beat D0 by >= 0.08)      FAIL at +0.0707 / +0.0736
```

**SPATIAL_HCRM_PROVISIONAL_SUPPORT.**  The threshold is not relaxed and the
numbers are not rounded up.

## Far — R4 nonlinear, and it fails harder than the linear arms

124 rows, fixed MLP 256-64-1, AdamW 1e-3, 30 epochs, seeds 1/2/3.

```
             seed1    seed2    seed3    mean     range
eval56       0.3917   0.5028   0.5444   0.4796   0.1528
wood         0.5595   0.5167   0.5964   0.5575   0.0798
cross-set    e->w 0.5378   w->e 0.4915   mean 0.5146     (R0 linear: 0.5915)

gate 1  eval56 >= 0.75          FAIL  0.4796
gate 2  wood   >= 0.70          FAIL  0.5575
gate 3/4 cross-set              FAIL  0.5146
gate 5  R4 - R0 >= +0.10        FAIL  -0.0769
```

eval56 lands **below chance** and the seed range is 0.153, so the model is
fitting noise on 124 rows rather than finding structure.  Adding capacity made
it worse than the linear readout.

**NONLINEAR_RELATIONAL_READOUT_FAIL.**  Sample is small; this is not a claim
that the information is absent, only that no fixed readout tried here reads it.

## Line — the sharpest result in this audit

Topology generated automatically from the 3D model: 12 physical edges found by
"exactly one axis differs", 3 incident edges per corner, class counts
2/2/2/4 for top-width, top-depth, base-width, base-depth, vertical.  No manual
index mapping.

Corner generation uses **no corner heatmap and no top-K**: for each corner,
score(x,y) = -exp(mean distance to its three incident edges / tau), tau = 5
belief pixels, global argmax.

```
mode  set        n   <=20px   <=50px   >100px   median err   PnP
O5    eval56    56     2.7%     8.5%    69.4%     148.66px     0
O5    wood      45     0.6%     1.4%    95.8%     292.53px     0
O12   eval56    56    98.7%    99.6%     0.0%       4.68px    56
O12   wood      45    96.1%    97.2%     1.4%       7.99px    45
```

```
ORACLE_SEMANTIC_LINE_GENERATION (O5)   FAIL   2.7% and 0.6% within 20px
ORACLE_INSTANCE_LINE_GENERATION (O12)  PASS   98.7% and 96.1%, PnP 56/56 and 45/45
```

**The five-class semantic representation cannot locate a corner; the twelve-edge
instance representation locates it almost perfectly.**  The reason is structural,
not empirical: `top_width` covers two parallel edges, so conditioning a corner on
"top-width, top-depth, vertical" selects six edges instead of three and the
distance field has minima at several corners at once.  Under O5 the median error
is 149px and 293px -- it is not a weak signal, it is the wrong quantity.

This is the cleanest explanation available for why every structural-line
programme in this project has failed on real data: the PPD head predicts five
semantic classes, and five semantic classes are insufficient to identify a
corner even with perfect ground truth.

**INSTANCE_AWARE_LINE_REQUIRED.**

## Decision

```
SPATIAL_HCRM                 PROVISIONAL SUPPORT   (gate 3 short by 0.007-0.009)
RELATIONAL_RCIM              FAIL                  (nonlinear worse than linear)
SEMANTIC_LINE (5-class)      FAIL                  (2.7% / 0.6% within 20px)
INSTANCE_LINE (12-edge)      PASS                  (98.7% / 96.1%, oracle)
```

**INSTANCE_AWARE_LINE_BRANCH_FIRST**, with SPATIAL_HCRM as the secondary.

`ENCODER_REPRESENTATION_CHANGE_REQUIRED` stays withdrawn: it requires no near
signal and no line capacity, and both exist.

## What O12 does and does not license

It is an **oracle**: it uses ground-truth edge geometry.  It shows that a
twelve-edge instance representation *contains* enough information to place every
corner, which the five-class one does not.  It says nothing about whether such a
representation can be learned from images.  That is the next question, not a
result of this audit.

## Not run

Semantic line readout probes L0-L4, the canonical PPD L0/M1 evaluation, the
per-corner capacity taxonomy, figures, and the new test suite.  The PPD
evaluation is now lower value than when it was specified: its head predicts the
five-class representation that O5 has just shown to be structurally insufficient.

Final-test: UNOPENED.
