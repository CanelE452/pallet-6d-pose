# eval56 -- not run

Phase E runs only after a sigma clears the N87 gate.  None did, so the eval56
holdout was not spent.  This is the intended behaviour: a one-shot holdout
evaluated without a selected configuration is not a holdout.

The relevant eval56 facts from the previous audit stand unchanged: at the
deployment sigma of 3 the decoder builds objects on 0 of 56 frames, and the
mechanism decoder's baseline is PnP 50/56 with a fixed-GT reprojection median
of 11.5578px.

A further reason not to have spent it: N87 and eval56 share 12 frames, so
eval56 would have been a partially contaminated holdout for a sigma chosen on
N87.  See `COMPATIBILITY_N87_GATE.md`.
