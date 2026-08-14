# Verdicts, per decoder

Gate definitions are re-read from the recorded files, not re-invented:
E2 from `role_stage_static_gate.json`, N2/N3 from `pfdr/pfdr_gate.json`, S1 and
C1 from their screen records in `_docs/history/2026-08-04.md` (#11 and #9).
S1's F2-far median and signed far bias, and C1's proposal adoption rate, are
arm-specific metrics this harness does not reproduce; per the audit rule they
are carried as **unavailable** and never counted as PASS.

```
   set  arm      P0      P1            P2  P0 == P1
───────────────────────────────────────────────────
eval56   E2  REJECT  REJECT  INCONCLUSIVE       YES
eval56   S1  REJECT  REJECT  INCONCLUSIVE       YES
eval56   C1  REJECT  REJECT  INCONCLUSIVE       YES
eval56   N2  REJECT  REJECT  INCONCLUSIVE       YES
eval56   N3  REJECT  REJECT  INCONCLUSIVE       YES
  wood   E2  REJECT  REJECT  INCONCLUSIVE       YES
  wood   S1  REJECT  REJECT  INCONCLUSIVE       YES
  wood   C1  REJECT  REJECT        REJECT       YES
  wood   N2  REJECT  REJECT  INCONCLUSIVE       YES
  wood   N3  REJECT  REJECT  INCONCLUSIVE       YES
```

Every arm REJECTs on both decoders that can be evaluated, on both sets.  P2 is
INCONCLUSIVE because it produced no pose to judge -- not because it passed
anything.  The single exception is wood C1, where one garbage hypothesis was
enough to evaluate and it failed.

## What reproduces and what does not (Q1)

Reproduces under the D2 extractor:

- the far-stage effect: E2 far -15.5% (P0) / -18.6% (P1) on eval56,
  N2 -16.2% / -18.3%
- the corner-improves-but-pose-does-not pattern: E2 eval56 reprojection
  +1.6% (P0) and +2.5% (P1), both the wrong direction
- S1's reprojection gain: -26.3% (P0) / -9.5% (P1), same sign
- C1's pose damage: paired median +0.296px (P0) / +0.587px (P1)

Does **not** reproduce:

- **N3's PnP gain.**  eval56 50 -> 52 on P0, 46 -> 46 on P1.  The two frames it
  rescued were rescued by pushing near corners over a raw-peak gate that P1
  does not use in the same way.
- **C1's PnP gain.**  50 -> 55 on P0 but 46 -> 43 on P1: the sign reverses.
- **S1's near-face regression.**  The +30% near damage that helped block S1 is
  4.6755 -> 6.0759 on P0 but 6.6494 -> 6.4561 on P1, a small improvement.

So the corner-level findings are decoder-invariant; the **detection-count**
findings are D0-specific.  That is consistent with the previous audit, which
showed N3's gain was a detection-recall effect rather than a pose effect.
