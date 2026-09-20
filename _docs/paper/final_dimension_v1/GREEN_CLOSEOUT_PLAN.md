# Green-panel closeout plan — 2026-09-19

Status: awaiting explicit user confirmation of final membership and annotation completion.
This plan is prospective to new green-panel predictions; it is not retrospective preregistration
of the already inspected development experiments. No new training or tuning is authorized by this plan.

## Selection gate

- Current condition inventory contains 407 unique candidate images; all 407 still have saved EVAL flags at this check.
- Suggested human-review budget: 150 distinct images (47 truncated, 73 non-truncated handheld, 30 evening including the 7 darker candidates). This is a practical proposal, not a power calculation or finalized sample.
- Membership must be an explicit approved ID list, intersected with current saved EVAL labels and validated annotations. Neither the 407 candidate inventory nor every EVAL record in the full workspace constitutes approval.
- Existing release `review_rows()` scans the full review workspace. Do NOT run its current freeze command to freeze the requested subset: it can include other EVAL images outside the 407 candidates.
- Preserve the old release code/model lock. Before final execution, provide an audited versioned evaluation entry point/contract supporting the approved subset; retain the same model weights and decoding.
- Audit image duplicates, training/development capture overlap, dimensions, camera provenance, keypoint visibility and manual/PnP-derived label sources. Do not select by prediction error.
- Excluded review images are not automatically training data. Evaluation membership and subsequent training authorization are separate.

## Fixed comparisons

1. R0 frozen YOLO baseline vs N0_BASE_REPLAY basic point refiner vs N2_DIM_ONLY dimension-conditioned refiner, on identical selected frames.
2. N0/N2 already trained seeds 1/2/3, reported individually and as descriptive seed summaries. Seed1 remains the fixed deployment model; do not choose a new winner or ensemble.
3. Same arms by truncation, handheld, and evening conditions. Tags overlap; provide denominators and do not sum group counts. The 7 darker candidates are descriptive examples, not an adequately powered night-robustness benchmark.
4. Paired improvement/degradation analysis, including both N2 vs R0 and N2 vs N0, and the already defined good-to-bad criterion. Report tails and failures, not only mean gains.

## Measurements and uncertainty

- Primary: frame-mean normalized corner8 E_sym with the existing full-denominator missing-detection penalty.
- Secondary: matched corner8 median/P90 pixel error, full-denominator PCK5/10/20, coverage and good-to-bad counts.
- Whole-object C4 equivalence is used only for scoring; no GT-driven inference selection. Unknown corners are not converted into observed ground truth. Report label provenance; PnP-filled coordinates are not independent physical measurements.
- Preserve the raw-image matching/bbox contract explicitly and report truncation-related limitations rather than quietly changing matching thresholds after results.
- `capture_20260902` and `capture_20260902_kimjihoon` are one contiguous recording, not independent clusters. Audit other related clips before defining capture groups. The current scorer uses session-folder names and must not be used unchanged for an independence claim.
- Report capture-level summaries. Use paired capture-cluster intervals only where defensible; with too few independent captures state that uncertainty is not reliably estimable. Repeated seeds/adjacent frames do not create new independent captures.

## Interpretation and final deliverables

- This panel is additional condition evaluation unless independence from all prior training/development is established. A constant-dimension square cohort alone does not isolate the causal value of dimension input.
- No new physical 6D accuracy, handling success, external-occlusion robustness, or superiority over untested prior methods is claimed.
- If N2 does not improve this panel, retain that result without retuning; restrict the conclusion accordingly.
- Keep the already measured same-session R0/N0/N2 runtime results separate from accuracy evaluation. No retraining is needed.
- Deliver approved membership/provenance and immutable hashes, predictions, aggregate/condition/seed tables, failure examples, updated manuscript and reproducibility instructions.
- Compile and visually inspect the final PDF once a LaTeX environment is available. Author details, affiliations, funding, rights and submission consent require author confirmation. No journal submission is automatic.
