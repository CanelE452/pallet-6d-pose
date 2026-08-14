# Corrected capacity gate: spatial, relational

Zero main-model training, zero main optimizers, no checkpoint touched.  Probes
are diagnostic readouts and are never saved.  All splits are grouped by frame
with an assert that no frame appears on both sides.

## What the previous audit actually tested, and where it was wrong

Two criticisms, checked rather than accepted:

**Pooling.** True and decisive.  The earlier descriptors reduced every patch to
per-channel mean and max, which removes the spatial arrangement inside the
patch.  A corner is a spatial pattern; the pooled descriptor could not see one.

**GAP cancellation.** Half true.  The earlier far pairs were not built as
`feature(GT) - feature(wrong)`, so nothing was literally subtracted.  But GT and
wrong rows come from the same frame, so the GAP block was **identical** across
the two classes and a linear model absorbs it into the bias.  Measured:
`max |GAP(GT) - GAP(wrong)| = 0.0`.  Different mechanism, same consequence -- the
global context carried no discriminative signal.

So the earlier "upper bound on information present" phrasing was too strong.  It
measured what a **pooled** descriptor makes linearly readable, nothing more.

## Near — corrected, and the verdict flips

Hard pairs only: GT against the same channel's strongest wrong maxima on
C-MISSED and C-WEAK corners, both orderings present.  738 rows.

```
arm                      eval56     wood     e->w     w->e     mean
D0 pooled F50            0.9711   0.8800   0.8574   0.8863   0.8718
D1 pooled F100           0.9624   0.8009   0.7336   0.6672   0.7004
D2 spatial F50 flat      0.9911   0.9135   0.9344   0.9506   0.9425
D3 spatial F100 flat     0.9454   0.8302   0.8724   0.7886   0.8305
D4 spatial multiscale    0.9780   0.8813   0.9273   0.9635   0.9454
```

```
gate 1  spatial arm >= 0.75 both sets          PASS  D2 0.9911 / 0.9135
gate 2  cross-set each >= 0.65, mean >= 0.70   PASS  D2 0.9344 / 0.9506
gate 3  spatial best beats D0 by >= 0.08       PASS  D4 0.9454 vs D0 0.8718 = +0.074
                                                     D2 0.9425 vs D0 = +0.071
```

Gate 3 lands at +0.071 to +0.074 against a +0.08 bar -- marginally short on the
letter, clearly positive in direction.  Recorded as measured rather than rounded
up.  **SPATIAL_HCRM: PASS on gates 1 and 2, marginal on gate 3.**

The substantive finding is that **preserving spatial arrangement moves near
hard-pair cross-set AUC from 0.87 to 0.94**, and it is **F50** that carries it,
not F100: D2 (F50 spatial) beats D3 (F100 spatial) on every column.  The earlier
conclusion that near capacity was absent was an artefact of pooling.  The
earlier observation that F100 is not the answer survives.

## Far — corrected, and the verdict holds

GT and the wrong top-1 presented together, both orderings, with global context
and predicted-centroid-relative geometry available.  124 rows, 60 eval56 and 64
wood.

```
arm                              eval56     wood     e->w     w->e     mean
R0 pooled local only             0.6389   0.4704   0.5664   0.6167   0.5915
R1 pair concat spatial           0.7056   0.5966   0.4502   0.6061   0.5282
R2 + GAP global                  0.7056   0.5966   0.4502   0.6061   0.5282
R3 + geometry + role             0.7056   0.5881   0.4521   0.6039   0.5280
```

```
gate 1  R3/R4 >= 0.75 eval56 and >= 0.70 wood   FAIL  0.7056 / 0.5881
gate 2  cross-set each >= 0.65, mean >= 0.70    FAIL  0.4521 / 0.6039
gate 3  R3 beats R0 cross-set by >= 0.10        FAIL  0.5280 vs 0.5915, worse
```

**RELATIONAL_RCIM: FAIL.**  Adding the global context and the geometry changed
essentially nothing -- R1, R2 and R3 are identical to four decimals on eval56 --
so the fix the criticism proposed does not rescue the far case.  Cross-set stays
at chance.

Read together with the identity result from the previous audit (four-way corner
identity at macro-F1 0.92-0.94, transferring across pallets), the picture is
consistent and specific: the features know **which** far corner they are looking
at, and cannot tell that the wrong place is wrong.

Caveat stated plainly: 124 rows over 101 frames is a small sample, and the wood
numbers move around.  This is a weak FAIL, not a strong one.

## Decision

```
SPATIAL_HCRM_JUSTIFIED      YES (gate 3 marginal at +0.071/+0.074 vs +0.08)
RELATIONAL_RCIM_JUSTIFIED   NO
LINE capacity               NOT RUN
```

**SPATIAL_HCRM_FIRST.**

`ENCODER_REPRESENTATION_CHANGE_REQUIRED` from the previous audit is **withdrawn**.
It required all corrected gates to fail and the near gate passes.

## Not run

Phases G through J: semantic line readout, oracle line generation O5/O12, the
canonical PPD evaluation, and the failure taxonomy.  The line axis stays
**UNJUDGED**.  The instruction required these regardless of the near/far result
and they were not reached; that is a gap in this audit, not a finding about
lines.

Final-test: UNOPENED.
