# Fixed N3 seed1 → cornerSubPix comparison

The completed 319-frame experiment and 600-call deployment benchmark are in
`_docs/experiments/pallet_n3_subpix_20261008_v1/RESULT_KO.md`.
No models, original RGB, or private annotations are included in this publication.

Set `PALLET_SOURCE_ROOT` to the existing input repository and
`PALLET_BASELINE_ROOT` to the unchanged a22fb14b baseline worktree, as required by
the existing baseline resolver. Run from this repository using its original
`pallet-yolo26` Python environment:

```bash
export PALLET_SOURCE_ROOT=/path/to/existing-input-repository
export PALLET_BASELINE_ROOT=/path/to/immutable-a22-worktree
python -B -m scripts.research.pallet_n3_subpix_20261008_v1.test_contracts
python -B -m scripts.research.pallet_n3_subpix_20261008_v1.reporting
```

The first command checks inference contracts with synthetic images and performs
no final-pose or model calls. The second recomputes statistics from the saved
rows, reusing the existing visibility categories only for post-hoc analysis;
it also performs no final-pose or model calls.

`evaluate.py` is the one-shot accuracy runner: it seals the existing input/code
hashes, checks the three controls on the fixed 26-frame panel, and calls the
existing final pose function once for each of the 319 combined predictions.
Its initial user-change snapshot is a local `INITIAL_STATE.json` in
`PALLET_N3_SUBPIX_SCRATCH`. Completed results cannot be overwritten by this
runner. `runtime.py measure` similarly protects its completed/partial attempt,
uses the same frozen panel, and measures the four actual complete routes.

In reused control rows, the legacy top-level `selected_index` retains its old
action-index meaning. Actual detector object selection is recorded consistently
in `fixed_metadata.selected_index`. The new combined rows save q0, qN, qS,
qFinal and the actual pose captured from their single final-pose call.

`CHECKS.json` documents the historical float32 N3 cap rounding and the six
fallback corners with a minute final cap adjustment. No configurations were
changed in response to performance. Publication to main follows the user's
explicit final instruction; the research branch preserves the same commit.
