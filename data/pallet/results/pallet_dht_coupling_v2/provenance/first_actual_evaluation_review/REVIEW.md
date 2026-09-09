# First actual evaluation review — balanced seed 1

Saved-artifact integration review **PASS**. This is not the full experiment completion or a comparative accuracy decision.

- All 3,008 actual predictions: 319 positive and 2,689 negative, with 14,083 finite candidates. All current image/source/output/checkpoint SHA bindings match.
- Canonical scoring keeps all 319 positive and 2,689 negative CSV rows. 2D IoU matching covers 309/319 positive frames; MAIN pose covers 319/319. Supervised point errors, median/P90, and four negative-score thresholds independently agree.
- All 14 actual evidence NPZs contain the twelve 90×113 role maps. Independent sigmoid maximum difference 4.04e−8; normalized transpose difference 1.28e−7; original-to-LetterBox affine difference 0.
- The supplied 640×480 source is pixel-identical to the canonical evaluation image. Role7 is rear_left_height, GT4 (21,347) to GT7 (22,391). The displayed evidence is broad over the left foreground and post; correct coordinate mapping does not establish correct semantic localization.
- The original point-number mismatch remains. Its official supervised median is 271.79 px over seven supervised corners plus centroid. No permutation, ranking choice, new forward, GPU use, or bound-source change was performed.

[Three-panel saved-output figure](raw_case_role7.png) · [Machine receipt](REVIEW.json) · [Direct visual review](VISUAL_REVIEW.json) · [Full input hashes](INPUT_HASHES.json)

The final 21-model gallery and browser interaction QA remain pending the complete experiment.
