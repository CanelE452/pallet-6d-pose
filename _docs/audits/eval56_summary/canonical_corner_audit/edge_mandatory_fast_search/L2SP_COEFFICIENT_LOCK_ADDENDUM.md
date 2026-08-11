# Replacing the L2-SP coefficient rule

`d543529` stands.  It records `SP_CALIBRATION_REFERENCE_MISSING` as provenance,
not as a failure, and nothing about it is amended.

## Withdrawn

```
"lambda_sp = CE_ref / R_SP_ref, evaluated at the historical F1 state at 1,703"
```

The reason is a fact about the files, established by reading them: the recorded
F1 checkpoint holds a 28-tensor decoder state and an audit dict whose only "vgg"
entries are an int and a float, and a scan of all 92 checkpoints finds no
adapted `net.vgg[19:27]` tensor anywhere.  Phase B saved `model.state_dict()`
and a report rather than the backbone.  `R_SP_ref` cannot be evaluated at a
state that was never written down.

## Forbidden phrasing

```
"the fresh F1 calibration reproduces the historical state at 1,703"
"the historical W was approximately reconstructed"
```

Neither is true and neither is claimed.  The training path is not
bit-reproducible -- 1.435e-03 of parameter divergence after twenty steps is
already on record -- so a fresh 1,703-step run reaches a different `W`, not an
approximation of the historical one.  The new state is named accordingly.

```
SP_CALIBRATION_STATE_1PASS      what is actually used
HISTORICAL_F1_1703              what it is not
```

`FRESH_TRAIN_ONLY_CALIBRATION_TRAJECTORY` is an instrument for fixing a unit.
It is not a historical result, not a model baseline, not a selection arm, and it
never evaluates held-out geometry.

## The new rule

Equalising loss magnitudes does not equalise what the optimizer actually feels,
because a term's contribution is its gradient, not its value.  So the
coefficient is fixed on gradients instead:

```
ONE_PASS_GRADIENT_BALANCED_L2SP

lambda_sp = ||g_task||_2 / ||g_sp||_2

g_task   gradient of the mean task cross-entropy over all of LINE_TRAIN,
         with respect to the late trainable parameters
g_sp     gradient of R_SP at lambda = 1, same parameters
```

At the calibration state this makes `||lambda_sp * g_sp|| == ||g_task||`.  That
is a unit choice and nothing more.

```
LAMBDA_OPTIMALITY_NOT_ESTABLISHED = True
LAMBDA_SELECTED_WITH_DEV          = False
LAMBDA_SWEEP                      = False
```

## Why the calibration cannot happen at step 0

At `W == W0` the penalty and its gradient are both exactly zero, so
`||g_sp||` is zero and the ratio is undefined.  The calibration state therefore
has to be somewhere the weights have moved, and the smallest such point that
matches the screen's own schedule is one pass:

```
fresh S0, task loss only, 0 -> 1,703 optimizer steps, no SP term present
```

## Two numerical regimes, deliberately different

```
CALIBRATION_NUMERICAL_REGIME  deterministic algorithms, so the coefficient is
                              reproducible to the bit and auditable
ACTUAL_TRAINING_NUMERICAL_REGIME  the default kernels, the same regime F1 ran in,
                              so S1 and F1 remain comparable
```

These are not the same and are not meant to be.  Calibration fixes a unit and
must be repeatable; the experiment measures a difference against F1 and must
share F1's conditions.  Mixing them would either make the coefficient
irreproducible or make the comparison invalid.

## The penalty, per module

```
r_l  =  ( ||W_l - W_l0||_F^2 + ||b_l - b_l0||_2^2 )
        / ( ||W_l0||_F^2 + ||b_l0||_2^2 + eps )

R_SP =  mean over the four late Conv2d modules of r_l
```

Weight and bias share one denominator per module, so a small bias norm cannot
dominate the ratio -- which is what per-tensor normalisation would have allowed.
The reference is the canonical `W0`, immutable.

## What a result here can and cannot say

Whatever comes out belongs to this one gradient-balanced coefficient.  Not
"optimal lambda", not "weight drift causes specialization", not "L2-SP solves
adaptation".  The G1-G6 definitions from `d543529` are unchanged, including G3's
thresholds, and seeing the coefficient does not license touching them.
