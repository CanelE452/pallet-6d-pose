# Limits that remain in the paper

- Reused DEV: seven Plastic and two Wood recording groups; historical LR5 selection; no independent confirmation.
- Two evaluated categories: ordinary plastic and one Wood category. No green/unseen-material guarantee. Wood's pooled corrected-minus-raw signal is small (+2/346 corners), below R0, and recording-wise PCK10 signs differ.
- 9-image/38-corner teacher manual adaptation;66 evaluation anchors extra; upstream generic pretraining and development history disclosed.
- Raw control shares teacher-based selection and support; only the coordinate intervention is isolated.
- No annotation-time efficiency or direct-manual-supervision superiority claim.
- P90 and several pose/statified measures worsen; main student20→10px gross recovery count0.
- 16 selected Plastic images/66 visible points do not validate hidden points or unlabeled217 exact quality. Wood has zero provenance-eligible verified-visible evaluation points; its Q1 is unresolved despite the legacy-reference student comparison.
- Legacy full-panel points have mixed provenance;6D is geometry-derived, not independent measured ground truth.
- Pose-head-only adaptation; order repeats are not independent init-seed repeats.
- Private data/checkpoints are hash-bound but not automatically redistributed as a full public dataset.

- Verified66 student PCK10 ties43/66 in both arms (R0=44/66); full128 legacy-label gain is not independently confirmed on that threshold.

## Subsequent bounded visible-transfer check

The original results above remain unchanged. A single paired 640-update extension reproduced both original 320-update prefixes exactly. Verified PCK10 increased to 44/66 in BOTH students, retaining the tie. Corrected native TRAIN target residual decreased only 3.070→2.994 px. Relative to the original corrected student, full128 PCK10 worsened 507→504/985 while D9 AUC improved 0.35902→0.36189. This is mixed evidence, not a resolved transfer bottleneck or a new independent test. No additional intervention is triggered; see `pallet_visible_transfer_closure_v1/REPORT_KO.md` and the manuscript appendix.
