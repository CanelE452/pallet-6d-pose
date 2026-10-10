# Prepared commands

The fixed protocol and CPU checks have already run. These commands require the original read-only feature cache, the separately prepared ready targets, and the original source checkout. Public stored checks can be inspected without those private dependencies; reproducing the actual head backward additionally requires the frozen private feature batch.

Use the same pallet Python environment. Replace the local path placeholders, and use a fresh check output rather than overwriting the published snapshot.

```bash
export PALLET_SOURCE_ROOT=/path/to/original/source/checkout
export PALLET_BASELINE_ROOT=/path/to/immutable/original/a22/baseline/checkout
task_python=/path/to/pallet/environment/bin/python
task_cache=/path/to/readonly/learned_cache
task_ready=/path/to/depth_recovery_v3
task_pkg=scripts.research.pallet_kp_supervision_repair_20261010_v1
task_docs=_docs/experiments/pallet_kp_supervision_repair_20261010_v1

"$task_python" -m "$task_pkg.retrain" check \
  --features "$task_cache/features.npy" --order "$task_cache/order.npy" \
  --manifest "$task_cache/CACHE_MANIFEST.json" \
  --targets "$task_ready/READY_PREPARED_TARGETS.npz" \
  --ready-validation "$task_ready/DEPTH_RECOVERY_VALIDATION.json" \
  --protocol "$task_docs/RETRAINING_PROTOCOL.json" \
  --checks /tmp/pallet-supervision-repair-recheck.json
```

There is currently no approved additional-update receipt. `AUTHORIZATION_PENDING.json` deliberately fails the authorization gate. After explicit authorization, the approved receipt must bind the published protocol and checks hashes and identify the user's authorization. It is a separate file; preserve the pending snapshot.

The prepared authorized sequence is below. It has **not** been run. Training writes into a new isolated directory, uses the existing targets/features/order, executes exactly 9,000 formal updates, and writes only the three last checkpoints. No throwaway optimizer steps are included. An interrupted training claim must be audited before any further updates.

```bash
task_fits=/path/to/new/isolated/repair_fits
task_approval=/path/to/APPROVED_ADDITIONAL_9000.json

"$task_python" -m "$task_pkg.retrain" train \
  --features "$task_cache/features.npy" --order "$task_cache/order.npy" \
  --manifest "$task_cache/CACHE_MANIFEST.json" \
  --targets "$task_ready/READY_PREPARED_TARGETS.npz" \
  --ready-validation "$task_ready/DEPTH_RECOVERY_VALIDATION.json" \
  --protocol "$task_docs/RETRAINING_PROTOCOL.json" \
  --checks "$task_docs/ZERO_UPDATE_CPU_CHECKS.json" \
  --authorization "$task_approval" --output "$task_fits"

"$task_python" -m "$task_pkg.downstream" infer \
  --protocol "$task_docs/RETRAINING_PROTOCOL.json" --fits "$task_fits" --output "$task_docs"
"$task_python" -m "$task_pkg.downstream" evaluate \
  --protocol "$task_docs/RETRAINING_PROTOCOL.json" --fits "$task_fits" --output "$task_docs"
```

The adapter reads the original `INPUTS.json`, sealed Base initial observations and fixed controls from the prior experiment; it does not copy or modify them. It uses the original decoder, solver, hidden reprojection and separate point-line path. `infer` produces 957 learned observation rows; `evaluate` seals 1,914 geometry rows before posthoc reference scoring. A completed stage refuses to overwrite its raw output.

Actual statistics, independent verification, the complete-path runtime measurement, report and normal Git publication follow these stages under the same frozen protocol. The ordinary Git workflow does not merge into main or use force push. The present public bundle contains preparation evidence only, not learned weights or a claimed successful pose result.
