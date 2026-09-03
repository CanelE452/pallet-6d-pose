# Experiments deliberately not repeated

Each entry names the artifact that decided it, so none of this rests on memory.

## Predicted-seed full 6DoF DiffPnP refinement
`_docs/audits/PREDSEED_DIFFPNP_GATE.md` — REJECT. Predicted 2D reprojection fell
11.96 to 6.35 px, a 47 percent gain, while the fixed indexed GT reprojection went
23.16 to 24.20 px and signed rotation worsened 13.8 percent. Translation improved
38 percent and rotation lost 14. Fitting harder to biased keypoints is not pose
accuracy, and the gate said so before the numbers were seen.

## DiffPnP3D retraining
`paper_s2_scratch_diffpnp/CONCLUSION.md` — PARTIAL transfer. Rear localisation,
detection robustness and sub-3-degree cases carried to real; overall pose accuracy
did not move.

## Tight-bbox short-side-400 DOPE second pass
The crop turned edge-on pallets into 1100x400 and wider strips and collapsed rear
and front localisation. S4 here crops a square instead, so the object aspect ratio
is untouched — and S4 still lost, which says the damage is not only the aspect
ratio.

## bbox or cuboid hull used as segmentation ground truth
Neither is a visible mask. Not attempted.

## Fixed-lambda corner plus line multitask
`_docs/audits/MULTIHEAD_FAILURE_DIAG.md` — the gradients do not conflict; they
agree weakly at cosine +0.13 to +0.25. The magnitude ratio moved roughly 35,000x
between step 0 and step 25,000, so `lambda_corner = 0.03518` fixed at calibration
contributes about 2e-05 of the line gradient by the end. Holding the intended
ratio would need a lambda near 1,300. A fixed lambda cannot express this.

## Why this screen changed cue roles instead
None of the above is repeated because this screen does not try to fit the full
pose harder to biased 2D. It separates which cue answers which question:
translation from bbox and points, rotation from structure. That separation is what
was untested.
