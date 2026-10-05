# Separate offline metrics

`evaluate.py` consumes the isolated review manifest, frozen full-frame plan,
corner contract, submitted human reference JSON, and normalized predictions
from `run_pipeline.py`. It imports only the local review validator and standard
Python modules. It writes `LIFTER_METRICS.json` and `LIFTER_CORNER_ERRORS.csv`;
it does not replace source data, the legacy 846 result, or the manuscript.

The CLI checks all selected image hashes through `review.serve.Context` and
rejects drafts, identity/hash mismatches, duplicate IDs, out-of-range clicks,
and incomplete visible references. Missing human reference yields
`WAITING_HUMAN`; missing or partial frozen predictions yields
`BLOCKED_CONTRACT`. No prediction is synthesized for an absent file.

Canonical corner accuracy uses original-image pixels and a 10-pixel inclusive
PCK threshold. Every approved direct visible corner remains in the PCK
denominator, including missing predictions and wrong-object failures. Median
and P90 errors are conditional on valid predicted corners, and their counts
are reported separately. Per-session values and paired same-corner changes
remain separate from the difference between method medians.

Target object correspondence must be independently confirmed. An exact
`object_id` match can bind the prediction to the reviewed target. Alternatively,
`object_match` must be a boolean with `object_match_source_kind: human_reviewed`.
Null correspondence blocks accuracy when directly visible corners exist.
This confirmation must never alter the original confidence-based candidate
selection. The normalized frozen inference rows start with null correspondence.

The command-line path evaluates canonical IDs only. The programmatic
`evaluate_visible_corners` also supports a separately frozen whole-object
permutation set supplied identically for both methods. A single full bijection
is chosen per object; coordinates, masks, and source IDs move together.
Its caller must store the permutation contract before reading prediction
errors. Canonical and permutation results have distinct `mode` fields and
must be reported separately. Per-point nearest matching is absent.

Continuity uses explicit `fresh`, `held`, and `no_pose` states. It preserves
stored duplicate records, counts distinct camera observations and inference
executions separately, lists camera-number recording gaps, and computes
nonfresh/no-pose intervals on the recorded sensor timeline. An ending gap is
censored: its duration to next success is null, with its observed span retained.
Sensor clock segments and sessions are never joined across resets.
Adjacent output change removes only 360-degree representation wrapping;
raw 90/180 branch changes remain visible. Camera gaps do not become model
failures, and these quantities are not accuracy or stationary noise.

Stop variation is null until a human-reviewed interval list is fixed before
viewing predictions and binds to the same plan hash. Each interval is summarized
separately, including actual duration and fresh/held counts. No independent
physical reference is present in this workflow, so physical position/yaw
accuracy remains null with `BLOCKED_REFERENCE`.

From the workspace root, run the synthetic unit checks with:

```powershell
py -3.12 -m unittest discover -s scripts/research/pallet_lifter_case_review_20261003_v1/metrics -p test_metrics.py -v
```

The tests use visibly named synthetic in-memory fixtures. They never create a
submitted human reference file or real experiment values.

An actual prediction file must have sibling `RUN_IDENTITY.json` with matching
source commit, plan hash, all 15 original code/settings/checkpoint bindings,
and unchanged budget/reference guards. The evaluator reads these constants
from the hash-verified frozen source with Python AST; it never imports the
inference or control entrypoint. Cached prediction records are distinct from
newly executed model calls in continuity counts.
