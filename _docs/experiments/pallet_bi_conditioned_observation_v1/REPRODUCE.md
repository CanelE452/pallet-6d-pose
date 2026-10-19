# Reproduce and verify

The public pack supports offline reaggregation of the published scalar results and inspection of all saved output poses. Full model/observation/pose replay requires the private source-data archive: original RGB, cached N0/state/mask observations, synthetic references, historical predictions, fitted checkpoints, and the original hash-bound run metadata. Those inputs are not all distributed in this repository.

Commands below are the implemented interfaces, in logical execution order. Run them from the repository root in an isolated research checkout. Existing immutable outputs must be retained; use a new output directory for a recheck. These commands document replay and do not authorize changing the historical experiment or operational defaults.

## Environment and private archive

The recorded base is `41ebc5a7f236abf6526eede5f6a8aa48176993b0`. The execution used Linux x86_64, Python 3.10.20, NumPy 1.26.4, SciPy 1.12.0, PyTorch 2.1.1+cu118, OpenCV 4.9.0, matplotlib 3.10.9, and Pillow 12.2.0. PyTorch's CUDA build was 11.8; the new jobs ran on CPU. Package versions describe the measured environment, rather than a portable installation lock.

Set the following variables to your restored private archive and intended output locations. `PYTHON_BIN` is the interpreter in that environment. `REF` is the directory containing `FIXED_REFERENCES.json`. `SYNTH` is the existing synthetic_v1 directory, not a new renderer output. `ASSET_REPO` contains the original dataset/annotation paths and frozen old sources. `PUBLIC` is a public derivative destination in the isolated checkout. For a fresh rebuild, start with a clean destination and retain the published pack separately.

```bash
: "${RUN:?Set RUN to the private run/archive directory}"
: "${D1:?Set D1 to the frozen boundary-action-repair run}"
: "${SYNTH:?Set SYNTH to the original synthetic_v1 directory}"
: "${REF:?Set REF to the canonical score_v2 reference directory}"
: "${PUBLIC:?Set PUBLIC to the public derivative directory}"
: "${HISTORY_OBS:?Set HISTORY_OBS to the frozen observation-contract run}"
: "${HISTORY_RISK:?Set HISTORY_RISK to the frozen risk-information run}"
: "${ASSET_REPO:?Set ASSET_REPO to the restored original asset repository}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the recorded Python environment}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MOD=scripts.research.pallet_bi_conditioned_observation_v1
```

Keep total concurrent CPU observation/pose workers at most three and BLAS threads at one. The bank commands below run sequentially with two workers. No new H1 fit, detector/backbone forward, synthetic rendering, or GPU job is needed. The cached path requires these actual inputs:

| Archive | Required content |
|---|---|
| D1 upstream | `d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz` and `SYNTHETIC1003_UPSTREAM.jsonl.gz`: original native points, N0 pose, camera, dimensions, IDs, splits and RGB bindings |
| D1 observations | `d1_real_observations_v1` and `d1_synthetic_observations_v1`, including aggregate/frame seals and each frame's `OBSERVATION.jsonl.gz` and `MASK.npz` |
| D1 FACE | Corresponding `d1_real_candidates_v1` and `d1_synthetic_candidates_v1` frame `E_RUN.jsonl.gz` plus seals, containing same-N0 typed partial observations |
| Synthetic archive | `SYNTH_GT_LABELS.jsonl.gz`, original split metadata, and all 1003 original images |
| Historical archives | Frozen OBS/RISK input/source contracts and saved N0/E0/H1/PoseFix output poses; do not regenerate these systems |
| Run metadata/models | Original P0, extraction/producer locks, canonical plan, amendments, source snapshots, normalizer, and six `training/OBS_{COMMON,BI}_s{seed}/best.pt` checkpoints with receipts |
| Real references | Canonical `FIXED_REFERENCES.json`; its numeric content remains unread until the prediction seal |

The P0 audit checked all 1241 RGB bindings, same-N0 state/mask/geometry ancestry, and 700 frozen source hashes. The inherited automatic state classifier used historical CAL human DIRECT/EXTERNAL labels, and fixed covariance/reliability used historical CAL supervision. The new six fits added no real supervision; this does not make the complete inherited system free of real supervision. Original detector/segmenter/state-model checkpoint provenance remains bound through those archived upstream contracts; cached replay does not replace it with new inference.

## Immutable bindings and reference conventions

The published [EXPERIMENT_PLAN.json](EXPERIMENT_PLAN.json) is the byte-exact original 23,547-byte frozen plan, SHA `a24bcc9204778d48927805ab2fc8c4f230c06868715a529c7bb144952d3e6a6f`. It passed the private-path/bulk-data checks without transformation and can be used as `--plan`. The base artifact builder initially emits a sanitized projection; the final public build restores the exact canonical plan bytes, as shown below. Other public contracts/manifests are sanitized projections, and their public file hashes do not replace the private input/checkpoint/lock hashes they record. Original P0 SHA is `3776dcbc4fbd3dca2053edddb9621992276c650f4bd06d907c7a2e7e7b435618`; synthetic GT SHA is `578a24d67db0959a0b014ece9802279c803ee05c9845c2fbe38e8e8c389bdeb6`; canonical real reference SHA is `d488564817a8cac8d0103350f03ac61d78bdeb56163c6a5a4daffe414f0f2af1`.

The archived final extraction lock is `EXTRACTION_LOCK_TECHNICAL_1.json` (SHA `9873bb9e3fe0c678ad31ea522919b0966812aa7c6ab0290880e3e5be23cecc34`); final producer lock is `BANK_PRODUCER_SOURCE_LOCK_TECHNICAL_2.json` (SHA `db4caa17793c4847b20fae144632a2938c23594936e26215fc0316c06e978fbf`). These and the canonical plan were frozen metadata, not products of a lock/plan-creation CLI. The public canonical plan alone does not supply the private P0, extraction/producer locks, checkpoints or data. Restore those archived originals for historical verification. A relocated/new execution requires explicitly rebuilt local bindings and a new plan frozen before its first fit; retain the original experiment separately. Cache rows include original paths, and timed audit reports need not have identical bytes after rerunning.

The original split is 768 train / 235 validation, disjoint by asset/run: `scene.usd` 243 C2, `scene_1.usd` 259 C2, and `eur_pallet_bk_cc0.glb` 266 C1 are train; `woodpallet_block_jtoastie_ccby.glb` 235 C1 is validation. Physical vertices are `N0.native_xyz @ Q`, where `Q = R_cf.T @ R_physical`. Original renderer `R_fixed`, translation, units, origin, camera, and native correspondence are retained. C1 permits identity; C2 permits identity and the 180-degree Y symmetry. No 90-degree symmetry or QR/SVD repair of raw GT matrices is applied.

Supervision picks one whole-object symmetry by total squared projection distance to all finite original N0 first-eight points, with identity winning ties. It does not choose a separate symmetry for each corner. Targets use the actual blended candidate coordinates, `log1p(error_px / 2)`. Fixed partial-line paths have no neural target or score. Separate target rows and alignment diagnostics never enter the feature tensors. Real references can be hashed as opaque bytes before prediction, but numeric real targets, clicks, and new real metrics are gated until all automatic outputs are sealed.

## 1. Recheck inputs and construct the common bank

P0 is read-only and runs no new line fit, pose solve, model forward, or real metric. Write a recheck report without replacing the archived P0 used by the extraction lock:

```bash
"$PYTHON_BIN" -B -m "$MOD.input_contract" \
  --prior-d1 "$D1" --synthetic "$SYNTH" \
  --history-obs "$HISTORY_OBS" --history-risk "$HISTORY_RISK" \
  --source-root "$ASSET_REPO" --real-reference "$REF/FIXED_REFERENCES.json" \
  --output "$RUN/P0_INPUT_CONTRACT_RECHECK.json"

"$PYTHON_BIN" -B -m "$MOD.bank" \
  --p0 "$RUN/P0_INPUT_CONTRACT.json" \
  --upstream "$D1/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz" \
  --extraction-lock "$RUN/EXTRACTION_LOCK_TECHNICAL_1.json" \
  --producer-lock "$RUN/BANK_PRODUCER_SOURCE_LOCK_TECHNICAL_2.json" \
  --domain synthetic --workers 2 --output "$RUN/common_bank"

"$PYTHON_BIN" -B -m "$MOD.bank" \
  --p0 "$RUN/P0_INPUT_CONTRACT.json" \
  --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --extraction-lock "$RUN/EXTRACTION_LOCK_TECHNICAL_1.json" \
  --producer-lock "$RUN/BANK_PRODUCER_SOURCE_LOCK_TECHNICAL_2.json" \
  --domain real --workers 2 --output "$RUN/common_bank"

"$PYTHON_BIN" -B -m "$MOD.audit_bank" \
  --run "$RUN/common_bank/synthetic_all_all_RUN.json" \
  --run "$RUN/common_bank/real_all_all_RUN.json" \
  --extraction-lock "$RUN/EXTRACTION_LOCK_TECHNICAL_1.json" \
  --output "$RUN/COMMON_BANK_RECHECK.json"
```

Both arms receive the same candidates from both cues on the original v2 15×49 pixels and cached tight mask, up to six lines / twelve legal crossings plus q0. Automatic V/states and geometry come from the same N0, not an E0/H1-refined pose. Original same-N0 FACE partials retain their archived mask/provenance. `bank` validates the exact source/lock/input/frame hashes and reuses only matching sealed rows. Do not rerun old full E0 or historical candidate production.

## 2. Separate synthetic labels, check constructed behavior, then fit six models

```bash
"$PYTHON_BIN" -B -m "$MOD.supervision" \
  --upstream "$D1/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz" \
  --features "$RUN/common_bank/synthetic_all_all_FEATURES.jsonl.gz" \
  --gt "$SYNTH/SYNTH_GT_LABELS.jsonl.gz" --p0 "$RUN/P0_INPUT_CONTRACT.json" \
  --output "$RUN/supervision"

"$PYTHON_BIN" -B -m "$MOD.test_models" \
  --artifact-root "$RUN/constructed_tests/model_training_v1"

"$PYTHON_BIN" -B -m "$MOD.training" \
  --run-dir "$RUN" --features "$RUN/common_bank/synthetic_all_all_FEATURES.jsonl.gz" \
  --targets "$RUN/supervision/TARGETS.jsonl.gz" --plan "$RUN/EXPERIMENT_PLAN.json"
```

The label builder requires the exact ordered 1003-frame join and 768/235 split and refuses an existing output directory. The test command writes a separate constructed receipt: nine tests, three MockAdamW constructions, fifteen mock step calls, **zero real optimizer constructions or scientific updates**. Use a new artifact root when rechecking it. Analytic observer/solver/supervision/audit tests are also available in `test_observer.py`, `test_solver.py`, `test_supervision.py`, and `test_independent_audit.py`.

`training` creates `TRAIN_ONLY_NORMALIZER.json` from valid TRAIN numeric values only. Each scorer has 181 normalized numeric values + 181 raw validity values + two condition values, then 64-ReLU / 64-ReLU / 1-Softplus, totaling 27,585 parameters. COMMON uses `(0.5,0.5)` and BI uses frozen automatic B/I one-hot. Each seed pair starts with identical parameter tensors and frame permutations. The objective is SmoothL1(beta=1), averaged candidate → corner → frame; fixed partials are excluded. AdamW uses learning rate 0.001, weight decay 0.0001, frame batch size 32, at most 50 epochs, patience five, and the first strict minimum of its own synthetic validation loss.

The default command fits COMMON and BI for seeds 20261018, 20261019, and 20261020 exactly once. It records optimizer constructions, all steps, initial/best/last checkpoints, histories, failures and attempts. Completed matching fits are reused; unfinished fits are not overwritten. A same-setting technical continuation requires explicit `--resume-technical`, and a changed execution source additionally requires `--technical-amendment <preserved-amendment.json>`. That option is not a new scientific configuration. The actual scientific execution used six optimizers, 360 steps each, 2160 steps total, fifteen epochs per fit and best epoch ten. No extra scientific overfit/smoke fit was run; constructed tests do not add to those counts.

## 3. Blind automatic prediction and sealing

```bash
PLAN_SHA256=a24bcc9204778d48927805ab2fc8c4f230c06868715a529c7bb144952d3e6a6f
"$PYTHON_BIN" -X faulthandler -B -m "$MOD.predict" \
  --run-dir "$RUN" --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --bank-dir "$RUN/common_bank" --bank-run "$RUN/common_bank/real_all_all_RUN.json" \
  --historical-run "$HISTORY_OBS" --plan "$RUN/EXPERIMENT_PLAN.json" \
  --plan-sha256 "$PLAN_SHA256" --reference-dir "$REF"
```

For a new preregistered replay, use that replay's canonical plan hash instead. `--reference-dir` identifies a forbidden read root here; it does not load targets. Run prediction in a fresh process. Its Python I/O audit hook whitelists cached numeric/checkpoint/source inputs and run-local outputs, denies reference/evaluation/diagnostic roots and subprocess/network escapes, and seals observed access. This is a Python audit hook, with the documented native-I/O/inherited-descriptor limitations, rather than an OS sandbox.

The final source points SciPy's temporary files to `$RUN/prediction_runtime_tmp` before guarded computation. An initial QHull/native temporary-file guard failure exited before any saved real-pose outputs; its source, stderr and receipt were preserved. The local-temp technical repair retained scientific settings and the reference deny rules. Use the final source instead of relaxing the guard or reading targets to debug prediction.

Prediction includes eight fresh matched arms and five frozen historical systems, 238×13 rows. Historical N0/E0/H1/PoseFix output poses retain their original identities; they are not relabeled as fresh common-solver results. Before scoring, require complete `PREDICTION_MANIFEST.json`, all frame/output/checkpoint/source hashes, and `before_scoring=true`. The actual complete manifest SHA was `a82f67e7644cf24e8723a16de50d7efd2e0233aa1964f54fc65ffd09bf55dc6f`; actual work was 2562 new pose optimizer calls and 525 exact observation-packet reuses.

## 4. Postseal evaluation, reference diagnostics and independent audit

Only after that complete automatic seal, permit the following reference-reading commands. Each diagnostic is separate from the automatic main table; no new fit is permitted.

```bash
"$PYTHON_BIN" -B -m "$MOD.evaluate" \
  --run-dir "$RUN" --reference-dir "$REF" \
  --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --bank-root "$RUN/common_bank" --output "$RUN/evaluation"

PREDICTION_SHA256=a82f67e7644cf24e8723a16de50d7efd2e0233aa1964f54fc65ffd09bf55dc6f
"$PYTHON_BIN" -B -m "$MOD.diagnostics" \
  --run-dir "$RUN" \
  --upstream-synthetic "$D1/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz" \
  --synthetic-gt "$SYNTH/SYNTH_GT_LABELS.jsonl.gz" --bank-root "$RUN/common_bank" \
  --output "$RUN/diagnostics_v1" --source-root "$ASSET_REPO" \
  --upstream-real "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --real-reference "$REF/FIXED_REFERENCES.json" \
  --prediction-manifest-sha256 "$PREDICTION_SHA256" --root-GO

"$PYTHON_BIN" -B -m "$MOD.diagnostics" --audit-legacy-ancestry \
  --source-root "$ASSET_REPO" --diagnostic-output "$RUN/diagnostics_v1"

"$PYTHON_BIN" -B -m "$MOD.candidate_diagnostics" \
  --run-dir "$RUN" --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --bank-root "$RUN/common_bank" --reference-dir "$REF" \
  --output "$RUN/candidate_diagnostics"

"$PYTHON_BIN" -B -m "$MOD.independent_audit" \
  --run-dir "$RUN" --bank-root "$RUN/common_bank" \
  --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --reference-dir "$REF" --evaluation-dir "$RUN/evaluation" \
  --output "$RUN/independent_audit_recheck" --allow-postseal-reference-read
```

`--root-GO` records the required postseal execution gate; it must follow checking the manifest, not replace that check. Use a new manifest SHA for a new sealed replay. Diagnostics use all original validation235 and representative `OBS_BI_s20261018`: automatic, geometric-role condition replacement, and projected-corner point replacement. The automatic visible set/bank/covariance/source IDs/partial handling and solver stay fixed. Geometric hull role is not independent physical B/I annotation; projected-point success is circular because the same GT supplies the point and evaluation pose. All 235 frames, noops, out-of-canvas rejects, failures and extreme errors remain. The validation asset was already used for early stopping.

The real provenance audit traces original CAL46+TEST49 annotation sources. All 95 migration hashes matched, but all 760 per-corner sources remained unknown and all 95 lacked the per-point extrapolated mask needed to distinguish direct clicks from PnP/extrapolation. Thus authenticated physical-point replacement and independent physical B/I/line diagnostics are NA. No integer-coordinate heuristic or mask-derived physical label is substituted.

Candidate reference/regret analysis reads only saved candidates and performs zero new pose solves/fits. The independent audit replays scores with NumPy, checks train-only normalization, paired initialization/order, optimizer ledgers, first strict minima and checkpoints, then independently recalculates saved raw-pose metrics and paired statistics. It creates no optimizer or pose solve and does not import the production metric/aggregation helpers.

Success is jointly translation≤5 cm and rotation≤5 degrees, using the original real C2 convention and unrepaired raw reference matrices. PRIMARY is TEST49+REMAINDER143=192; CAL46 is separate. Three seeds repeat these same 192 images and do not create 576 independent observations. Paired recording-cluster bootstrap uses seed 20260917 and 10,000 draws, with separate session sensitivity. All finite tails, unavailable poses, fallbacks, and full denominators are retained.

## 5. Build and audit public derivatives

This stage projects completed evidence; it does not run scientific experiments. Build base projections before final reviewed prose or diagnostic publishing, because `artifacts` writes generated report files. Original numerical diagnostic source and its immutable snapshot must remain available for the publication proof.

```bash
"$PYTHON_BIN" -B -m "$MOD.artifacts" --run-dir "$RUN" --public "$PUBLIC"
cp "$RUN/EXPERIMENT_PLAN.json" "$PUBLIC/EXPERIMENT_PLAN.json"
"$PYTHON_BIN" -B -m "$MOD.diagnostics" --publish --run-dir "$RUN" --public "$PUBLIC"

"$PYTHON_BIN" -B -m "$MOD.figures" \
  --run-dir "$RUN" --output "$RUN/figures_rebuild"

: "${PRIVACY_PLAN:?Set PRIVACY_PLAN to the authenticated display-only privacy plan}"
"$PYTHON_BIN" -B -m "$MOD.gallery" \
  --run-dir "$RUN" --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --bank-root "$RUN/common_bank" --privacy-path "$PRIVACY_PLAN" \
  --output "$PUBLIC/gallery"
```

Copy the generated ordinary figure PNGs and `FIGURES.json` from the new figure directory into the public `figures` directory, keeping the separate diagnostic figure. Gallery output must be a new directory; it contains authorized derived displays of all 192 PRIMARY frames, with privacy edits confined to display. Raw RGB, masks, annotation bulk, GT matrices/coordinates, caches and weights are not copied. `artifacts.clean` removes private path strings from public projections. The final narrative, provenance documents, reproduction guide and root pointer are reviewed publication files rather than outputs of a single all-in-one build command.

After every final file is ready, run the full pack audit. The published location for this experiment is `_docs/experiments/pallet_bi_conditioned_observation_v1`; the source directory is fixed below. `prepare` creates an immutable payload manifest, and `verify` must run against unchanged payload/source bytes. Choose new private receipt filenames for every attempt.

```bash
"$PYTHON_BIN" -B -m "$MOD.publication_audit" prepare \
  --worktree "$PWD" --public-dir "$PUBLIC" --run-dir "$RUN" \
  --source-dir "$PWD/scripts/research/pallet_bi_conditioned_observation_v1" \
  --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --private-output "$RUN/PUBLIC_PACK_PREPARE_RECHECK.json" \
  --public-output "$PUBLIC/PUBLIC_PACK_AUDIT.json" \
  --base 41ebc5a7f236abf6526eede5f6a8aa48176993b0

"$PYTHON_BIN" -B -m "$MOD.publication_audit" verify \
  --worktree "$PWD" --public-dir "$PUBLIC" --run-dir "$RUN" \
  --source-dir "$PWD/scripts/research/pallet_bi_conditioned_observation_v1" \
  --upstream "$D1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --private-output "$RUN/PUBLIC_PACK_VERIFY_RECHECK.json" \
  --public-output "$PUBLIC/PUBLIC_PACK_AUDIT.json" \
  --base 41ebc5a7f236abf6526eede5f6a8aa48176993b0
```

The audit checks actual file bytes and decoded content, all 238×13 saved poses/metric rows, 192 gallery rows, scalar derivation, raw RGB byte/pixel exclusion, private model/GT/cache exclusion, file sizes, new-scope source inventory and the root readme's exact preserved original suffix. It records no browser or human visual-review claim. The separate [USER_WORK_PRESERVATION.json](USER_WORK_PRESERVATION.json) binds the original checkout's unchanged HEAD/local-main/branch/index/status bytes and six pre-existing dirty files, plus unchanged old sources/backend dependencies. All work occurs in the isolated checkout; do not reset, stash, or rewrite the user's original checkout.

## Recorded execution and source history

[EXECUTION_AND_COST.json](EXECUTION_AND_COST.json) contains sanitized actual command arrays, exit codes, durations, counts and log hashes. Private originals live in `execution/<phase>/RECEIPT.json` with stdout/stderr, plus diagnostic process-cost files. Named process and nested phase timings overlap and must not be summed as independent wall time.

| Actual measured phase | Recorded work |
|---|---|
| Common synthetic bank | 1003 frames, including four exact pilot reuses; 113.140 s batch wall |
| Common real bank | 238 frames; 59.143 s batch wall; no new state/mask forward |
| Six scientific fits | Six real optimizers, 2160 steps, zero failed scientific fits; 9.064 s whole-process wall |
| Blind prediction | 2562 actual pose optimizer calls, 525 exact reuses; 137.774 s numeric phase / 138.952 s completed process |
| Synthetic reference replacements | 235×3 groups, 452 actual pose optimizer calls, zero fits; 31.154 s internal / 32.37 s process wall |

The private source history retains initial observer/bank revisions and the two technical extraction/producer corrections, the pre-repair prediction source and failed guard receipt, independent-audit attempts, and the display-only gallery repair. The latter changed no scientific output. Numerical diagnostics were generated by source SHA `39443569c4480b0e742b7dba196e665e99e99a7bdf0c6abc1cefbb69317856ac`, retained as `source_attempts/diagnostics_numerical_v1.py`. Later provenance/publication functions were added without changing any original function AST; the metadata-extension proof preserves both source hashes. No numerical diagnostic rerun or added fit followed that extension.

## Offline checks using only the public pack

The standard-library example below recomputes each PRIMARY arm's denominator, success count and available translation mean/sample-SD/max from `ROW_METRICS.csv.gz`. It does not trim extreme values. Set `PUBLIC` to the checked-out pack and use any compatible Python 3 interpreter:

```bash
"$PYTHON_BIN" -B - "$PUBLIC" <<'PY'
import csv, gzip, math, statistics, sys
from collections import defaultdict
from pathlib import Path
groups = defaultdict(list)
with gzip.open(Path(sys.argv[1]) / 'ROW_METRICS.csv.gz', 'rt', newline='') as f:
    for row in csv.DictReader(f):
        if row['partition'] in ('TEST', 'REMAINDER'):
            groups[row['arm']].append(row)
for arm, rows in sorted(groups.items()):
    assert len(rows) == 192 and len({r['id'] for r in rows}) == 192
    values = [float(r['translation_cm']) for r in rows if r['translation_cm']]
    assert all(math.isfinite(x) for x in values)
    print(arm, len(rows), sum(r['success'] == 'True' for r in rows),
          len(values), statistics.mean(values) if values else None,
          statistics.stdev(values) if len(values) > 1 else None,
          max(values) if values else None)
PY
```

`SELECTED_POSES.jsonl.gz` contains the actual saved 13-arm R/t outputs and statuses for all 238 frames. Without the private GT, a public reader cannot independently derive geometric errors from these R/t values; reaggregating published scalar metrics is a narrower check. A complete independent raw-pose audit requires the private canonical reference file and exact hashed inputs/checkpoints/seals. `DIAGNOSTIC_ROWS.csv.gz` has 235×3 scalar rows from the separate synthetic diagnostic population and must not be pooled with the automatic real192 result.

## Ordinary remote-main verification

Publication uses the repository's ordinary main path, with a fresh remote-main check, fast-forward only when valid, or the normal PR path if protections/races require it. No force push or protection bypass is part of this procedure. After the result is integrated and remote objects are available in the isolated checkout, verify the exact expected main head and all reviewed blobs:

```bash
: "${RESULT_COMMIT:?Set RESULT_COMMIT to the integrated result commit}"
: "${EXPECTED_MAIN_HEAD:?Set EXPECTED_MAIN_HEAD to the freshly observed remote main head}"
"$PYTHON_BIN" -B -m "$MOD.verify_remote_publication" \
  --worktree "$PWD" --public-dir "$PUBLIC" --run-dir "$RUN" \
  --private-output "$RUN/REMOTE_MAIN_VERIFY_RECHECK.json" \
  --result-commit "$RESULT_COMMIT" --expected-head "$EXPECTED_MAIN_HEAD"
```

This read-only verifier uses actual `ls-remote`, ancestry, complete Git-object blob equality and SHA checks, and a second stable remote-ref read. It requires normal remote authentication and the relevant fetched Git objects. It verifies Git publication, not CDN cache propagation or a browser-rendered page.
