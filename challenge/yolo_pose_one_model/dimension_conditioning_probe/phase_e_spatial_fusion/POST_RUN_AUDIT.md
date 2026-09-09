# Phase-E post-run audit and interpretation note

This file was added after the immutable Phase-E outputs were produced.  It
does not replace or alter the source-selection lock, checkpoints, result JSON,
or their recorded hashes.  It is the authoritative clarification for reading
`SPATIAL_FUSION_REPORT.md` and `SPATIAL_FUSION_SUMMARY.json`.

## Verified execution state

- All four arms ran for seeds 0, 1, and 2 (12 tiny-probe runs total).
- The source selection and median-seed artifacts recompute exactly from the
  per-run synthetic-DEV histories.  No synthetic-TEST metric, real metric, or
  searched threshold contributes to those choices.
- The frozen protocol/contract/runner/test hashes, 14 upstream source
  bindings, 81 spatial shards, final spatial cache, 12 checkpoints and 12 run
  sidecars all pass integrity validation.
- The original YOLO checkpoint SHA is unchanged, and YOLO optimizer
  membership, YOLO parameter updates, full-YOLO runs, wrapper integration and
  deployment are all zero.
- The unit test suite passes 9/9.

## Lock-boundary wording correction

The sentence in `SPATIAL_FUSION_REPORT.md` saying that synthetic TEST and real
DEV were re-accessed only after the durable lock is too broad.  Before the
lock, the runner performed label-independent spatial extraction for all rows
and eagerly loaded the completed Phase-C cache, including the full synthetic
label array and cached TEST/real rows.  The TEST subset was not evaluated, no
TEST result influenced selection, and real GT parsing/evaluation began only
after the lock, but literal pre-lock *re-access* was not zero.

Accordingly, the lock fields
`phase_e_synthetic_test_reaccessed_before_lock=false` and
`phase_e_real_reaccessed_before_lock=false` must be interpreted narrowly as
"not evaluated or used for Phase-E selection before the lock", not as "no
bytes or arrays were loaded".  This is a reporting/protocol-boundary deviation
and reinforces the already-frozen exploratory-only status.  It does not
authorize a promotion claim.

## Exact-center denominator clarification

The extraction audit reports `center_exact_n=40185`.  Of those rows, 40,183
have a valid detected source cell and match the prior float32 center feature
exactly.  The remaining two are real abstention rows with no detected center;
they correctly retain the all-zero patch, all-false mask and `-1` source
sentinels, but are counted as exact sentinel agreement rather than as detected
center comparisons.  Candidate count, confidence, box and keypoint parity are
still exact, and the valid detected-center comparison remains 40,183/40,183.

## E9 status clarification

E9 is **not run**, not a measured latency failure.  Its recorded status is
`NOT_RUN_E1_TO_E8_FAILED`, because E1, E2, E3, E5 and E6 failed.  The report's
Boolean `False` should be read as "the complete E1-E9 gate was not satisfied".

## Additional evidence limitations

- The current synthetic TEST is not an end-to-end holdout: the frozen YOLO
  checkpoint had upstream exposure to all 10,095 TEST images, and both its
  labels/results and COMMON128 were historically opened in Phase C.
- The real shuffled control contains 173 rows, of which 171 are forward-valid;
  the wrong-dimension control changes 185 triplets, of which 183 are
  forward-valid.  Reported metrics correctly retain invalid rows as
  abstentions in the full population denominator, but the control metadata
  does not separately state these effective-valid counts.
- Per-frame real probabilities were not persisted.  The absence of real S2/S3
  evaluation is supported by the restricted runner loop, aggregate arm keys,
  empty CSV real columns, and artifact hashes rather than a per-frame log.
- The built-in source-lock validator does not enforce every descriptive
  scope/rule field.  The current step-zero, sampler, selected-epoch, DEV-score
  and scope declarations were therefore also checked independently and were
  found to match the run artifacts.

## Audited conclusion

The numerical verdict is unchanged: spatial concat is the source-locked
winner, FiLM and cross-attention do not beat it under the pre-registered
selection rule, and the selected concat probe fails the real parity,
dimension-use and oracle-recovery requirements.  Phase E remains an
adaptively reused DEV diagnostic only; no full-YOLO training or deployment
follows from it.
