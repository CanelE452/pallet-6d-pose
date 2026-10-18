# Reproduction commands and required original inputs

This guide describes the actual research CLIs. Completion, run counts, resource
use and actual numerical results are established by the final report and bound
manifests. A command below is not evidence that a pending stage was executed.
The generation listings are a command inventory for the registered campaign,
not a turnkey bootstrap script. They require the authorized original inputs,
frozen locks and prerequisite receipts. Only the generation primitives are
reusable in another appropriately registered run; the current campaign gate is
specific to this campaign's recorded interruption and recovery.
The original RGB, frozen descriptors/weights, historical stage caches and renderer
references remain private inputs. Public result panels and small scalar ledgers
do not provide those original training/evaluation files or a public-only full
rerun. For already archived results, use the separate numeric-recheck section
below rather than rerunning production, freeze or scoring-go.

Use the existing environment recorded in `ENVIRONMENT.json`; do not update model
libraries or retrain YOLO/N3/PoseFix. Run from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P_PY=python
P_RUN=/path/to/restored/registered-boundary-action-repair-run
P_OLD=/path/to/original/statewise-end-to-end-recovery-v1
P_SCORE=/path/to/original/statewise-correspondence-diagnosis/score_v2
P_FAIR=/path/to/original/accuracy-closeout-run
P_ORACLE=/path/to/original/statewise-correspondence-diagnosis/oracle_v1
P_DOC=_docs/experiments/pallet_boundary_action_repair_v1
```

The run must retain the original PRIMARY192=TEST49+REMAINDER143 and separateCAL46
IDs, camera, units, physical axes and C2 convention. Synthetic data retains its
original1003 IDs, train768/validation235 asset split and per-row C1/C2 targets.
Exact hashes and method settings are in the frozen plan, source bindings and
input contracts. Public path aliases require substitution with authorized local
cache paths; they are not downloadable input files.

## Registered generation command inventory

For an authorized stage replay, restore the exact registered inputs and all
prerequisite locks/receipts, and use an empty or nonexistent stage output directory
as required by that CLI. Existing immutable outputs are not overwritten; exact
completed-cache reuse is allowed only by the stated resume path. Do not delete
old locks or fabricate receipts to make an existing run look fresh. The commands
below are not a sequence to rerun over an archived completed root.

`campaign freeze` itself requires `PLAN_DRAFT.json` and the existing
`P0_BASELINE_INPUT_INVENTORY.json`, `B_SOURCE_LOCK_v2.json`,
`B_LOCAL_REGULARIZATION.json`, `D1_PREFLIGHT_V2.json`,
`L_D0_TRAINING_GO_LOCK.json` and `LEARNING_PLAN.json`. It is not a blank-directory
initialization command. Preserve an archive's existing `PLAN_LOCK.json`,
`EVALUATION_PROTOCOL.json` and later prediction/GO seals rather than freezing again.

## Finite B branch

Use the frozen original full238 input/bank and the train-only local information
receipt. Each new batch has its own immutable lock. The original RGB profiles and
all failures remain; the bank control performs no new solve. Full/edge/local
limits are8/8/2 per frame, with restoration counted inside the iteration budget.

```bash
"$P_PY" -B -m unittest scripts.research.pallet_boundary_action_repair_v1.test_finite_boundary -v

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.finite_batch run \
  --inputs "$P_OLD/inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz" \
  --old-config "$P_OLD/CONFIG_FROZEN.json" \
  --original-batch "$P_OLD/real_batch_all_available_v1" \
  --regularization "$P_RUN/B_LOCAL_REGULARIZATION.json" \
  --output "$P_RUN/b_finite_v1" --workers 1

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.independent_finite_audit \
  --inputs "$P_OLD/inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz" \
  --original-batch "$P_OLD/real_batch_all_available_v1" \
  --batch-dir "$P_RUN/b_finite_v1" --output "$P_RUN/independent_finite_reproduction"
```

The original failed arithmetic audit and its raw differences must remain. If
rechecking an existing failed receipt, provide `--prior-failed-check` with its
actual file. The independent code records finite support, raw numerical
representation differences and any secondary expression replay separately;
physical material ownership remains unverified.

## D1 automatic inputs and actual E0

The sealed upstream files are
`d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz` and
`d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz`. They contain actual frozen N0
outputs, known camera/model information and raw prediction coordinates, without
per-frame GT pose/visibility/mask input. The shared observer runs states, automatic
masks, B/I recovery, partial/O/T/U and the same automatic E0 procedure. Synthetic
E0 is not an N0 copy.

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.d1_batch observe \
  --inputs "$P_RUN/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz" \
  --output "$P_RUN/d1_real_observations_v1" --old-run "$P_OLD" \
  --domain real --device cuda --root-GO

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.d1_batch observe \
  --inputs "$P_RUN/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz" \
  --output "$P_RUN/d1_synthetic_observations_v1" --old-run "$P_OLD" \
  --domain synthetic --device cuda --root-GO
```

Only one GPU job is permitted. Existing exact, sealed rows are reused rather than
counted as new network work. Tiny/ResNet/head call counts, direct SAM hook counts,
provider-based reconstructed counts and the observation host-wall upper bound
must remain separate where the persistent predictor hook has a capture gap.

The common producer can execute an uncached domain directly:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.d1_batch solve \
  --inputs "$P_RUN/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz" \
  --observation-dir "$P_RUN/d1_synthetic_observations_v1" \
  --output "$P_RUN/d1_synthetic_candidates_v1" --old-run "$P_OLD" \
  --domain synthetic --workers 4 --root-GO
```

For the campaign's shared dispatcher and exact original stage reuse, set
`P_ORIGINAL143` to the bound original v4_14 P1/WD0 stage run and `P_TYPED` to the
bound TYPED_MULTI run:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.d1_memoized_batch \
  --private "$P_RUN" --old-run "$P_OLD" \
  --original-run "$P_ORIGINAL143" --typed-run "$P_TYPED" \
  --workers 4 --root-GO
```

This is an alternative dispatcher, not an additional candidate generator to run
on top of a completed scientific bank. Reuse requires exact observation,
consumed-input, warmstart, source, numerical-branch and actual candidate parity.
Only stages from that one fixed P1→WD0→G10→G13 procedure are admitted. TYPED_SINGLE,
later V15/V17 additions and unions of unrelated historical proposals are excluded.
Existing completed rows and their original seals remain. Shared-parent wall time
is counted once across domains; per-worker time is retained separately.

## Recorded interruption/recovery inventory

This campaign's original shared process ended with observed tool exit143; its
sender/cause and unsealed partial optimizer/CPU counts remain unknown. The
recorded recovery retained238 real and454 synthetic sealed frames, archived four
unsealed attempts, and scheduled the remaining549 synthetic rows using the same
703-file frozen source closure. These are fixed historical provenance conditions,
not expected events for a new experiment. Never invent interruption receipts for
a fresh run or use the recovery adapter as a generic restart framework.

The following are the one-shot operational entrypoints used for that recorded
state, not instructions to execute them again on a completed archive:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.d1_recovery_batch prepare \
  --private "$P_RUN" --root-GO

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.d1_detached_supervisor \
  --private "$P_RUN" --old-run "$P_OLD" \
  --original-run "$P_ORIGINAL143" --typed-run "$P_TYPED" --workers 4

"$P_PY" -B -m unittest scripts.research.pallet_boundary_action_repair_v1.test_recovery_gate -v
```

The actual launcher recorded the supervisor's detached session, working directory,
thread limits and two source hashes in `D1_RECOVERY_LAUNCH_LOCK.json`; the argv
above alone does not establish those conditions. The supervisor invokes
`d1_recovery_batch run` with the same path arguments and `--workers 4 --root-GO`.
Do not launch that child separately while the supervisor owns it. The20 gate tests
use small fake chains and a frozen-supervisor source-prefix check; they do not
certify actual batch completion or read real pose/GT data.

Actual completion requires the original `D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json`,
`D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json`, launch/start receipts, the supervisor's
actual waited-child exit receipt `D1_RECOVERY_SUPERVISOR_RECEIPT.json`,
`D1_RECOVERY_FINALIZATION_RECEIPT.json`, final feature seals and the independent
recovery-aware `DATA_PIPELINE_PARITY.json`. `recovery_gate.verify_recovery_for_scoring`
checks their hashes, original journal preservation, reused frame seals and final
runtime-copy linkage. A command, toy-test PASS or missing exit receipt is not
evidence that recovery finished.

## L action training and common-bank prediction

[L_REPRODUCE.md](L_REPRODUCE.md) contains the complete label, checkpoint reuse,
fixed-seed training and action-audit commands. It uses three reused D0/H0 models
and at most18 new small fits. The representative remains seed20261018 regardless
of the other seeds. Train-only normalization is shared by H0/H1 for the same D;
pair models are trained separately and pool only actual N0/E0 actions.

All21 checkpoint outputs are newly inferred from the exact same
`d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz`; historical115/192 is a separate
old-bank comparator. Canonical real metadata comes from the original
reference-free238 input file. Final L prediction output is
`action_predictions_v1/PREDICTIONS.jsonl.gz`. New B-repair candidates are not added
to this L bank. Do not launch training/audit workers when a four-worker producer
already occupies the CPU allowance.

## Seal all predictions before real-reference evaluation

The assembler only joins prediction outputs. For this campaign's first postseal
evaluation, the scoring gate checks the frozen method hashes, exact populations,
all registered outputs/checkpoints, resource limits and the fixed recovery chain
above. It is intentionally not a generic gate for a new uninterrupted run. This
is the historical command inventory; a completed archive retains its original
GO and must not rerun assemble or scoring-go:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.assemble \
  --run-dir "$P_RUN" --old-run "$P_OLD" --original-score "$P_SCORE" --fair-run "$P_FAIR"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.campaign scoring-go --run-dir "$P_RUN"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.evaluate \
  --run-dir "$P_RUN" --original-score "$P_SCORE" \
  --output "$P_RUN/evaluation_v1" --scoring-GO

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.posthoc \
  --run-dir "$P_RUN" --evaluation "$P_RUN/evaluation_v1" \
  --oracle-root "$P_ORACLE" --output "$P_RUN/posthoc_v1"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.independent_numeric \
  --run-dir "$P_RUN" --original-score "$P_SCORE" \
  --evaluation "$P_RUN/evaluation_v1" --output "$P_RUN/independent_numeric_v1" --scoring-GO
```

Independent numeric code does not import the production metric or aggregator.
Stable atan2/fsum values and separately replayed original expressions for strict
zero-delta signs are distinguished. All192/49/143/CAL46 denominators, missing
poses, bad seeds, large errors and binary/continuous damage remain. These are
repeated development data, not a new independent real test.

## Archived-result numeric recheck

With authorized access to the complete preserved private archive, restore the
registered `PLAN_LOCK.json`, `EVALUATION_PROTOCOL.json`, `PREDICTIONS.jsonl.gz`,
`PREDICTION_SEAL.json`, original `ALL_OUTPUTS_SCORING_GO.json`, their bound inputs
and the original reference/score seals. Keep those bytes and all original result
directories unchanged. The evaluator verifies the stored prediction/protocol/GO
chain; `--scoring-GO` here authorizes this separate reference-only recheck and does
not regenerate the historical GO receipt or repeat its interrupted production.
The independent check also reads the preserved `posthoc_v1/` tables and mechanism
receipt, original oracle, and bound candidate/feature evidence. Retain their
recorded paths or provide filesystem links to the same bytes; do not rewrite
sealed JSON paths or omit posthoc evidence to obtain a shorter check.

Choose fresh, nonexistent output paths. These commands recompute metrics from
the saved predictions and independently reaggregate them; they do not train,
generate candidates or solve poses:

```bash
P_ARCHIVE=/path/to/preserved/registered-boundary-action-repair-run
P_RECHECK=/path/to/new/numeric-recheck-output

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.evaluate \
  --run-dir "$P_ARCHIVE" --original-score "$P_SCORE" \
  --output "$P_RECHECK/evaluation" --scoring-GO

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.independent_numeric \
  --run-dir "$P_ARCHIVE" --original-score "$P_SCORE" \
  --evaluation "$P_RECHECK/evaluation" --output "$P_RECHECK/independent_numeric" --scoring-GO
```

Do not call `campaign freeze`, `campaign scoring-go`, assemble, recovery prepare,
observation/solve or training against that archived root. Rechecking the B geometry
audit also requires its original private inputs and a new audit output directory.
Public scalar summaries and panels alone cannot reproduce the sealed pose-level
evaluation. A recheck of these same development images is not a new independent
test or permission to alter the original references or operating defaults.

## Figures, actual image review and local publication

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.figures \
  --run-dir "$P_RUN" --evaluation "$P_RUN/evaluation_v1" --output "$P_DOC/figures"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.gallery \
  --run-dir "$P_RUN" --old-run "$P_OLD" --evaluation "$P_RUN/evaluation_v1" \
  --original-score "$P_SCORE" --output "$P_DOC/gallery" \
  --rgb-public --rights-receipt "$P_RUN/RIGHTS_RECEIPT.json" \
  --privacy-plan "$P_RUN/DISPLAY_PRIVACY_PLAN.json"
```

Use a new empty gallery directory for a retry. The user confirmed publication
rights for the PRIMARY192 result panels; any image outside that scope requires its
own right or a geometry-only public view. Display-only privacy masks must retain
the original pixel coordinate system and measurement/RGB input hashes. The actual
gallery CLI/privacy options and receipts record the rendering scope. Generating
PNG files is not pixel review: root opens representative actual images and records
the reviewed hashes, problems and rechecks separately.

Measured root batches use `run_recorded --receipt ... --batch-id ... --scope ... -- COMMAND`
to preserve actual exit code, whole batch wall, child CPU and peak RSS. This is a
cost wrapper, not another scientific run. Failed starts also leave a receipt.
The final cost assembly preserves exact measured wall and conservative parent
lifetime bounds separately; unmeasured preparation remains explicitly unknown.

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.cost_validation cost --run-dir "$P_RUN"
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.cost_validation validation --run-dir "$P_RUN"
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.cost_validation bindings --run-dir "$P_RUN"

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.build_report \
  --run-dir "$P_RUN" --output "$P_DOC" --report-GO
```

Run the last command only after actual evaluation, independent checks, figures
and pixel-review receipts exist. An absent receipt remains N/A; it is not evidence
of a successful stage. The final report can be refreshed with `--replace-generated`
after root-reviewed publication receipts, preserving earlier private snapshots.
No new model, selection or fit is made by reporting. `cost_validation` preserves
prior derived receipt bytes on refresh; immutable scientific locks are never
rewritten.

Root writes the final report/README and adds the existing `_docs` index pointer.
The README embeds the actual summary PNGs and B regression/detail panel; HTML is
for local download/opening because GitHub blob does not execute it. The packer
preserves those documents and images and prepares only allowlisted small result
derivatives:

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.public_pack prepare \
  --run-dir "$P_RUN" --repo-root "$PWD" --output "$P_DOC" --publication-GO

"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.public_pack verify \
  --repo-root "$PWD" --output "$P_DOC" --publication-GO
```

`--aliases` can point to a private JSON mapping from exact absolute cache roots to
symbolic `${NAME}` aliases. `--extra-files` can point to an explicit list of small
JSON/CSV/Markdown source/target mappings. Raw pose journals, candidate banks,
original RGB bulk, checkpoints, fonts and secrets are not accepted. An explicit
`replace_default:true` mapping may select the actual immutable receipt path when
a technical retry uses another directory; original failures should be included
under distinct public names. This changes publication provenance, not scientific
model selection.

`PUBLICATION_MANIFEST.json` binds every original and public derivative SHA,
redaction/summary operation and preserved artifact. `PUBLICATION_CHECK.json`
checks local hashes, links/anchors, full192 gallery IDs,37-arm scalar/paired
denominators and per-image HTML rows, rights/display-privacy receipt chains,
PNG/JPEG decode/EXIF and README image embeds.
It does not claim actual pixel review or remote publication. Root stages only the
new research allowlist, performs an ordinary safe main push (or PR if necessary),
and verifies the actual remote SHA and files separately.

### Verify the actual pushed result commit

After root commits and normally pushes scientific result `A`, run the separate
Git verifier with its full SHA. This command performs actual `ls-remote` and
`fetch --no-tags origin main`; it never stages, commits, pushes, merges, resets or
edits either worktree. Root provides the approved explicit stage allowlist and
actual prepush/push receipts, with captured staged NUL bytes and push output
hashes under the private run directory.

```bash
"$P_PY" -B -m scripts.research.pallet_boundary_action_repair_v1.publication_verify \
  --repo-root "$PWD" --run-dir "$P_RUN" --result-commit "$P_RESULT_A" \
  --prepush-receipt "$P_RUN/ACTUAL_PREPUSH.json" \
  --push-receipt "$P_RUN/ACTUAL_PUSH.json" \
  --stage-allowlist "$P_RUN/ACTUAL_STAGE_ALLOWLIST.json" --verification-GO
```

The receipt schemas are `RESULT_A_STAGE_ALLOWLIST_V1`,
`RESULT_A_PREPUSH_RECEIPT_V1` and `RESULT_A_PUSH_RECEIPT_V1`. The allowlist contains
repo-relative paths and each actual precommit file SHA256. Prepush binds the
execution base, fetched main SHA, actual staged paths/raw stream SHA and root's
review attestations. Push binds exact `A:refs/heads/main` argv, exit status and
actual stdout/stderr file hashes; force pushes are rejected.

The verifier hashes the committed A blobs, proves A is contained in actual remote
main, and repeats the original worktree's exact START status command and protected
file checks. It preserves A's offline check and manifest under
`publication_snapshots/result_A/`, then writes immutable
`PUBLICATION_GIT_PROOF.json` and `REMOTE_PUBLICATION_RECEIPT.json` in private.
Failed attempts retain raw command outputs and a failure receipt; canonical
proofs and A snapshots are never overwritten. A later metadata commit B may
reference this older A proof without asking B's current manifest to hash itself.

An isolated local bare-Git fixture test exercises this publication machinery
without accessing the research remote or any scientific reference:

```bash
"$P_PY" -B -W error::ResourceWarning -m unittest \
  scripts.research.pallet_boundary_action_repair_v1.test_publication_verify -q
```
