# Reader questions and evidence boundaries

1. Why the problem? Residual landmark displacement affects monocular geometry even when detection is unchanged. Operational tolerance is not supplied, so no insertion/safety margin is invented.
2. Why this method? Reuse a frozen estimator's local features and train a small correction. It is an empirical system hypothesis, not a new expectation operator or proof of optimality.
3. Why these controls? R0 isolates adding refinement; D changes the readout/supervision package; L retains historical structured refinement; PoseFix-derived pallet9 supplies a source-based image-conditioned prior under matched exposure.
4. Why these settings? They are bound historical P settings and the authorized prior protocol. No new setting is chosen on DEV or confirmation.
5. What supports the result? Hash-bound original-pixel raw outputs, fixed GT denominators, same-session paired statistics, canonical pose reference and measured runtime. See generated tables and NUMBER_SOURCES.json.
6. What remains? Independent sessions with blinded labels/reference QA, actual author declarations, and any resource-stage technical gaps recorded in FINAL_STATUS.json. Existing DEV cannot fill the independence gap.

The completed fixed-budget prior favors the separate-backbone comparator over P for central DEV error. The exact contrast and its interval are generated from `P_VS_PRIOR_PAIRED.json`, not copied here by hand. This rules out describing P as the most accurate tested model on that statistic. The expanded same-session runtime panel is now completed in `RUNTIME_PANEL.json`: P has lower measured latency, with a smaller added network. This is an accuracy--cost tradeoff, not all-metric superiority. Its numerical-replay and isolated-memory recovery limitations are disclosed in the supplement. No model or training budget was changed in response to the result.
