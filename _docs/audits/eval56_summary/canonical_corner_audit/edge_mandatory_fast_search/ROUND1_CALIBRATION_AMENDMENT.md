# Round-1 calibration amendment

Made **before** validation was opened.  At the time of writing: validation
evaluations 0, EDGE_ZERO 0, EDGE_SHUFFLE 0, GT_EDGE_ORACLE 0, untouched 0,
canonical 0, final-test 0 -- recorded with file-existence evidence in
`pre_validation_amendment.json`.

## The defect

The calibration rule is `lambda = target_share * median(L_belief) /
median(L_component)`, clipped to `[1e-3, 100]`.  With a belief median of 0.00283
and edge medians of 37.09 and 31.21, four of five lambdas landed on the lower
bound -- and the bound **raised** them.

```
component      median     lambda   contribution   intended
belief        0.00283    1.0000        0.00283          --
centre       37.0945     0.0010        0.03709     0.00141      26.2x
incidence    31.2137     0.0010        0.03121     0.00141      22.1x
```

`TARGET_SHARE` says centre should carry half the belief term.  It carried 26
times it.  The shares stopped describing the objective, so R1A optimised
something other than the declared loss.

## What changed

The lower clamp is removed.  Lambda is the raw normalisation again.

Numerical safety is now a block rather than a clip: a lambda outside
`[1e-8, 1e4]` or non-finite raises `HARD_BLOCKED_LOSS_SCALE_PATHOLOGY`.  A
safeguard that silently rewrites the objective is what caused this.

The anchor is now the frozen A1 base belief rather than each arm's own final
belief, so E1 and E2 calibrate against one identical scale.

A fidelity gate was added: `realized_share = lambda * median(L_k) /
median(anchor)` must match `target_share` within 1%, 5 of 5.  Below the gate,
smoke does not run.

```
component      median       lambda   realized  target   rel err
centre        37.0945    3.808e-05     0.5000    0.50   0.00e+00
orientation    0.3690    1.914e-03     0.2500    0.25   0.00e+00
length         1.2239    5.771e-04     0.2500    0.25   0.00e+00
support        0.7687    3.675e-04     0.1000    0.10   0.00e+00
incidence     31.2137    4.526e-05     0.5000    0.50   0.00e+00
CALIBRATION_V2 PASS
```

## This is not threshold tuning

No gate moved and no target share changed.  The declared semantics were restored
after a safeguard broke them, and the defect was found in a train-only
calibration read, before any validation number existed.  Had it been found after
seeing validation, the correct action would have been to keep the result.

## R1A

```
run_id               R1A_FLOOR_CLAMP
selection_eligible   false
reason               LOSS_CALIBRATION_FIDELITY_FAIL
checkpoints          weights/paper_s2_edge_fast/{E1,E2}/round1.pth  (preserved)
validation           never run, and will not be
```

R1B trains from a fresh initialisation under separate paths
(`weights/paper_s2_edge_fast/R1B/`).  Continuing from R1A weights would carry a
different objective's optimum into a run reported under this one.

## A correction to an earlier claim

The previous history entry read the 3.8% scalar contribution as "the belief head
barely trains".  A scalar share is not a gradient share, and that inference was
not supported.  Gradient norms are recorded per component from the calibration
batches as a diagnostic only; they never enter the choice of lambda.

## Unchanged

split `9a755438`, projection `blender_math.build_view_matrix`, loader contract
(image 400x400, refine/belief 50-grid, factor 8), subsets, difficulty, A1
checkpoint.  The index.csv mask-path repair does not touch the group split, which
is computed from label metadata.
