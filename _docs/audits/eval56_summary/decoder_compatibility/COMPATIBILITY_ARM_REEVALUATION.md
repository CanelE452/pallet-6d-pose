# Arm re-evaluation -- not run

Phase F runs only on `P2_COMPATIBILITY = PASS`.  Compatibility failed at
Phase D, so B0, E2, S1, C1, N2 and N3 were not re-evaluated through the
deployment decoder, and no existing verdict changes on that basis.

The verdicts from the previous audit therefore stand as recorded: every arm
REJECT on P0 and P1 across both sets, with the P2 column unmeasured.  It
remains unmeasured after this audit -- what this audit adds is the reason, and
the quantity by which the model misses.
