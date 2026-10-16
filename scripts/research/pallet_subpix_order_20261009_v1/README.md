# Fixed SUBPIX / N3 order comparison

The runner reuses the original predictor, fixed N3 seed1, pose function, saved four-route rows, and existing DEV319 image grades. It adds one reverse route, `SUBPIX_N3`, with the final movement limit anchored at Base. Labels are used only after inference.

Set the read-only input roots to installations containing the recorded inputs and the unchanged a22 baseline dependencies. Commands run from the experiment checkout; its own output namespace is `_docs/experiments/pallet_subpix_order_20261009_v1/`.

```bash
export PALLET_SOURCE_ROOT=/path/to/input-checkout
export PALLET_BASELINE_ROOT=/path/to/immutable-a22-checkout
```

Reaggregate the four existing routes **before** reverse inference:

```bash
/path/to/pallet-environment/bin/python -B -m scripts.research.pallet_subpix_order_20261009_v1.reporting --stage existing
```

This creates `GRADE_BINDINGS.json` and the separate `EXISTING_GRADE_*` results with zero model or pose calls. It verifies the grade join against the prior image bindings and reference-annotation hashes. It uses the current audit `severity` field, not an earlier approval status.

For a **new isolated execution**, supply a private scratch directory through `PALLET_SUBPIX_ORDER_SCRATCH`. Before changing the source checkout, record its current `head`, `status_sha256`, `tracked_diff_sha256`, `tracked_sha256` mapping and `status_text` in that directory's `INITIAL_STATE.json`. This is the local user-change preservation snapshot consumed by the evaluator, distinct from the completed public results. Do not publish the raw private status snapshot or copy another run's snapshot.

The actual accuracy and timing commands used for this fixed run were:

```bash
/path/to/pallet-environment/bin/python -B -m scripts.research.pallet_subpix_order_20261009_v1.evaluate
/path/to/pallet-environment/bin/python -B -m scripts.research.pallet_subpix_order_20261009_v1.runtime measure --remaining-seconds 1790
```

Do not rerun these commands into the completed namespace. The evaluator's protocol/attempt guards preserve completed or interrupted rows, and the runtime journal preserves consumed calls. Reproduction requires a fresh isolated output namespace and local input bindings; the committed artifacts here already contain the completed 319-frame accuracy evaluation and 750 timing pipelines. Changes in performance do not authorize another execution or changed settings.

After the fixed reverse accuracy evaluation and five-route runtime measurement finish, reaggregate all five routes:

```bash
/path/to/pallet-environment/bin/python -B -m scripts.research.pallet_subpix_order_20261009_v1.reporting --stage final
```

The final report preserves the earlier grade results. All grade intervals use the same original 13-session draws and original observation weights. Mean ± SD describes error dispersion; the separate paired interval describes uncertainty in a mean difference. Image grades are descriptive human labels with unconfirmed physical severity semantics.

The protocol, checks, reverse raw rows, runtime ledger, and independent verification record the actual inference and pose calls. Original RGB, annotations, and weights are read locally and are not copied into the publication. Publication is confined to `research/subpix-n3-order-20261009`.
