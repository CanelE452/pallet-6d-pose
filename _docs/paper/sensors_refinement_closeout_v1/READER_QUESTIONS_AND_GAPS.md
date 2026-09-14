# Reader questions and remaining evidence

Working title: Synthetic-Supervised Local Keypoint Refinement for Monocular Pallet Pose Estimation.

1. What problem is addressed? Residual image-coordinate error from a fixed RGB pallet estimator, with downstream geometric pose evaluation.
2. Why does it matter? Pixel localization contributes to pose measurements; actual required handling accuracy has not been supplied or validated in insertion trials.
3. Why this method? P reads existing neck features with a small trainable module and preserves the detector's boxes, scores and instance selection. The cost of sampling and processing features must be measured, not inferred from parameter count.
4. Why these controls? R0 isolates the effect of adding local refinement; historical L tests structural line refinement; D reads the same evidence and regresses movement directly. PoseFix remains a needed formal comparator. D is not its reproduction.
5. What can the data support? Development-only paired accuracy effects, complete-failure-aware PCK, geometry metrics and desktop runtime. P candidate-readout specificity depends on the completed D comparison and its convergence limits.
6. What remains? Actual prior-method training/evaluation, untouched real confirmation with blinded QA, operational precision requirements, and any embedded-device or insertion measurement.

Known dimensions, geometry-reconstructed reference, missing detections, three trained seeds and reused development observations limit the claim. IEEE Sensors relevance must be argued through a concrete sensing/measurement contribution. Neither a familiar method nor a newly named architecture guarantees publication.
