# v1 → v2 code difference summary

- A: removed the independent linear anchor penalty; retained normalized corrected-point SmoothL1 and one-sided no-harm.
- B: replaced null/support proxy use gating with a separate common utility head trained from detached candidate counterfactual gain >0.25px.
- C: replaced nearest-bin hard CE with fixed one-bin-bandwidth continuous theta/rho soft targets; lattice remains 36×65.
- D: replaced simultaneous multi-mode normal-equation accumulation with one MAP-local alternative per semantic edge.
- Fairness: common stem and utility head use an arm-independent RNG stream; same-seed D/H state, manifest and minibatch-order SHAs are audited.
- Unchanged: stock R0 points/backbone/P3/P4 export, 1792/256/512 populations, C1/C2, AdamW, seeds 1/2/3, 2000 steps, batch 8, 1% shift cap, center8, GT-free scoring, and real-DEV gate.
- Not added: PCGrad, pose/translation solver or loss, candidate bank, beam search, fine-tuning, rendering, self-training, real labels, sweeps.
