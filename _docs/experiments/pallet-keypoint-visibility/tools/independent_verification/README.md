# Verify the published primary results

This verifier uses only the Python standard library (Python 3.9+). It does not
load a model, weights, images, feature arrays, or dataset packages.

From a clone containing the publication, run:

```bash
python3 verify_published_results.py --self-test
python3 verify_published_results.py --root _docs/experiments/pallet-keypoint-visibility
```

The default inputs are the archived V184 `ALL1784_CURRENT_NATIVE_LEDGER.jsonl`
and `ROOT_RESULTS_KO.json`. `--ledger` and `--results` accept paths relative to
the publication root. A path may name plain JSON/JSONL, `.gz`, or a `.parts`
JSON descriptor. No original absolute source path is followed.

When a publication manifest is available, also check its published files:

```bash
python3 verify_published_results.py --root PUBLICATION_ROOT --manifest MANIFEST.json --report NEW_REVIEW.json
```

`--manifest` is relative to the publication root. It accepts a `files` or
`entries` list/mapping. Manifest v2 preserves `published_relative` as the logical
path. Raw copies use `published_sha256`. Large texts use an embedded `gzipchunks`
descriptor with ordered parts, stored-byte SHA256 values, and the original
decompressed `content_sha256`. The verifier streams all stored-file SHA checks;
only the ledger, result JSON, and compressed textual parts are decoded. An
inventory still marked `CANDIDATE_*` is not a published manifest.

The independent scalar calculation uses the original primary cohort flag and
original D/E states, keeps model U calls in recall denominators, and verifies
all four models for ALL, WOOD, and SQUARE (12 rows). It compares exact confusion
counts and precision/recall/U rates with the saved report. All original 1,784
slots, 45 input-U rows, and primary D1147/E90 are checked.

This verifies saved arithmetic and publication integrity. It does not provide
new physical native-point gold, certify a new model, reinterpret historical
phase/review authority, or use the evaluation data for training or selection.
The program prints JSON and exits nonzero on failure. `--report` creates a new
file and refuses to overwrite an existing review.
