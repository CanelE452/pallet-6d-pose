> **SUPERSEDED 2026-08-06.**  Every number below was measured on D13/C13, which
> come from `data/_eval_sets/*combined` -- the set CLAUDE.md bars from evaluation.
> See `CANONICAL_REEVALUATION.md`: three of these verdicts flip on canonical data.

# Branch result so far — not the final mechanism

Two of the seven hypotheses are closed.  Topology coverage and the C13
regression are still open, so this is a branch record, not a completed failure
mechanism.

## H1 SYNTHETIC_INTERVENTION_FAILURE — REJECTED

On the population the training actually intervened in, 96 fixed S-TRUNC
samples chosen by TACA acceptance without looking at any model output:

```
       centroid    R4    in-frame <=20px    detected corners
A0     67/96       39         65.3%               239
A1     69/96       43         63.8%               246
A2     72/96       46         75.4%               305
```

A2 over A1: centroid +3.1%p, R4 +3.1%p, in-frame localisation **+11.6%p**,
detected corners **+24%**.  The J1 gate needs every one of the four below
10%p; localisation is above it, so H1 is rejected.

Stated carefully: the response gains are small (+3.1%p).  What clearly moved is
in-frame localisation and how many corners survive detection.  This is not
"the synthetic problem is solved".

S-LEGACY shows no regression: 96/96 centroid and 95 R4 on all three arms,
in-frame <=20px 98.0 / 98.0 / 98.2%.

## H4 LOCAL_HEAD_COUPLING_GAP — REJECTED at the current measurement

```
                       D13 (A2)    C13 (A2)
F50 inside norm          0.325       0.340
F50 background norm      0.376       0.344
F50 enrichment           0.866       0.989
PRH inside               0.3621      0.5150
PRH enrichment           1.36        2.31
H6 centroid peak         0.0561      0.6336
H6 corner peak           0.0744      0.8385
```

```
D13 F50 inside >= C13 10th percentile      11/13
D13 PRH_ALIVE (inside >= C13 10th pct)      2/13
D13 PRH enrichment >= 2.0                   3/13
D13 PRH alive AND H6 dead                   0/13   <- gate needs >= 8/13
```

The coupling hypothesis requires object evidence to exist and the local branch
to fail to consume it.  On no frame does that hold.  **OCSH/GCFM has no
supporting evidence at this point.**

Note the shape of the F50 number: the encoder does produce activation inside
the pallet (11/13 above the control floor), but the enrichment is **below 1**,
meaning the inside is *weaker* than the background.  Activation exists;
discrimination does not.

## Still open

- H5 topology coverage: whether S-TRUNC and D13 share visible topology, not
  just bbox statistics
- H6 late fine-tune drift vs H7 TACA mixture regression on C13
- per-frame stage trajectory and first-break stage
- whether the F50/PRH deficit is geometry-driven or appearance-driven

## Holdout

E44 SEALED, W45 SEALED, final-test unopened, zero training steps in this audit.
