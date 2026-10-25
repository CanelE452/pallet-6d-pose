# Public RGBD surface-AUX batch compiler

Prepared CODE only. Public source pixels, arrays, mesh bodies and poses were not read during preparation. Only pinned CODE/policy metadata and two invented temporary RGBD fixtures were used.

```sh
python -B /ABS/compile_public_batch.py --entry /ABS/ROOT_TRUE_ENTRY.json
```

The separate Root entry must set `approved_public_surface_batch_teacher=true`, pin this driver, retain the exact projection/compiler/policy pins, and list every frame with six complete `{path,sha256,size_bytes}` bindings: RGB, DEPTH, POSE, SAM, K, OBJ. `frame_id` is an inert unique identifier; output directories use sequential F000000 names. Optional `provenance` is copied as JSON and never followed. No download, dataset discovery, external material/texture, model or training API exists. NumPy and Pillow must already be installed.

The original geometry and11×11 surface policy are reused directly. T is fixed main; inverseT is diagnostic only. No scale, pose, depth bias, threshold or tolerance change is exposed. Every frame and all fixed r/q records remain; coarse capture groups do not prove independent scenes. Geometry UNKNOWN and integrity/decode failure have separate statuses. Processing PASS does not certify pose/registration/physical gold.

Each frame stores main T depth/raw residual in lossless compressed NPZ, label/eligibility/reason maps in NPZ, all r/q JSONL in gzip with mtime0 and empty filename, geometry/label summaries and COMPLETE. The gzip bindings include uncompressed UTF8 SHA/size/record count. InverseT full arrays are explicitly NOT_SAVED; both interpretation statistics remain. There are no per-frame illustrations. Batch FRAME_INDEX preserves every listed ID and exact inputs; SUMMARY totals count only compiled frames and exposes every failure. Partial outputs and FAILURE are preserved; no retry.

The r/q record schema is the original pilot schema. A policy reference E count is not a model correct count or recall. U is exclusion, not model abstention; SAM-negative alone is not E/SELF. These labels are generic front-surface AUX, not new native gold, query-pixel D/E gold, actual initializer queries, or automatic training admission.
