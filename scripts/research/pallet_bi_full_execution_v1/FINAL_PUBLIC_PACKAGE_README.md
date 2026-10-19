# Final publication selector

These commands select saved outputs and write publication metadata. They do not
run pose estimation, neural inference, fitting, rendering, or scoring. The final
build refuses planned completion: it requires the actual 33 registered completed
H1 fits, 11 full prediction seals, final sequence and derived terminal receipts,
completed cost projection, and ROOT's limited visual-review receipt.
The resulting snapshot says `scientific_and_derived_complete:true`,
`publication_ready:true`, `whole_campaign_complete:false`, and
`publication_status:"PREPARED_NOT_YET_COMMITTED_OR_PUSHED"`. R stage completion
refers to reports and analyses. ROOT records actual post-push remote commit
completion separately, outside this commit, to avoid a circular claim.

Use the isolated execution checkout and the recorded Python environment. Set
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`. Replace
`REPO`, `RUN`, and the fresh output filenames below with reviewed local paths.
The commands below are supported interfaces, not claims of historical execution.

```sh
python -B -m scripts.research.pallet_bi_full_execution_v1.public_package final-template --repo REPO --run RUN --output RUN/FINAL_SELECTION_TEMPLATE.json
```

ROOT reviews the explicit code filenames and exact RUN-owned execution scripts,
supplies the actual terminal and document hashes, seven-enum stage statuses,
typed provenance/cache/call/argv/environment/model/calibration receipt hashes,
and sets `template_only` to false in a new selection spec. Every actual service
launcher is preserved as source with its original byte hash. Scripts copied to
`execution_sources` retain original absolute paths; map local run, worktree,
interpreter, and input-data paths before independent reproduction. Python API
calls and unrecorded historic launches remain identified as such.

The ROOT-reviewed size policy has schema
`ROOT_REVIEWED_PUBLIC_SIZE_POLICY_V1`, `approved:true`, `author:"ROOT"`, positive
integer `maximum_file_bytes` and `maximum_total_bytes`, a review `reason`, and
`no_case_or_outlier_omission:true`. The per-file maximum cannot exceed 90 MiB,
below Git's 100 MB file limit. Total size is explicit and configurable. A failed
size audit preserves its metadata and refuses the manifest; it never truncates
an artifact, removes a case, or silently increases the limit. Issue a fresh
policy/spec/manifest after reviewing the actual audit.

ROOT's `ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW_V1` receipt must bind all eleven
run-relative `GALLERY_MANIFEST.json` hashes, assert completed limited spot
review and approved public display, keep `human_owner_certifications:0`, and
assert `raw_RGB_copied:false`. It is an AI display check, not physical-owner
certification or review of every image. Existing approved case JSON remains
byte-identical, including q, small covariance matrices, source, dimension,
partial-support and scalar observation reliability information. Neural model
state, tensors, training normalizers, raw reference poses, source RGB, CAL RGB,
private structure RGB, assets, and full pose/observation banks are excluded.

```sh
python -B -m scripts.research.pallet_bi_full_execution_v1.public_package final-build --repo REPO --run RUN --spec RUN/ROOT_FINAL_SELECTION.json --manifest RUN/FINAL_PUBLIC_MANIFEST.json
python -B -m scripts.research.pallet_bi_full_execution_v1.public_package verify --manifest RUN/FINAL_PUBLIC_MANIFEST.json
```

Build writes a private exact-file allowlist, size audit, public provenance and
code index, and CSV projections of the 120 saved population/contrast records
and all paired category IDs. All eleven galleries, both scatter bundles,
orientation supplement, D scalar export, all forty scalar estimator summaries,
full thirty-contrast public JSON, and two distinct cost projections remain
included. The separate final `ERROR_BUDGET40_V1` terminal supplies all six
whitelisted scalar files, bound to the exact closed forty-arm metric export:
800 joint-category, 1120 band-contribution, and 320 conservation-check rows.
It is explicitly posthoc descriptive accounting, with zero new pose/reference,
solve, or fit calls. Initial twenty-five-arm tables are excluded from the final
table bundle; their execution receipt may remain in the typed cost index.
The separate `report_appendices` spec binds a private
`FINAL_KO_REPORT_SAVED_SCALAR_SEAL_V1` receipt. Exactly five appendices are
selected under `tables/final_report/`: `FULL40_PRIMARY_METRICS.csv` (40 rows),
`REPRESENTATIVE_PRIMARY_METRICS.csv` (16), `PAIRED30_PRIMARY_COMPACT.csv` (30),
`FIXED_COHORTS.csv` (165), and `FIXED_COHORTS.json`. Their reporter source,
closed forty-arm export, paired export, final error budget, cost summary, and
four completed postbank receipts are authenticated. ROOT must first copy the
exact sealed `FINAL_REPORT_KO.md`, `S1_D_EVIDENCE.md`, and
`S3_S4_EVIDENCE.md` into the public-document source folder and select their
hashes. The private report receipt and arbitrary input metadata are not copied.
Its separate `metadata_documents` map also binds final
`METHOD_DIFFERENCE_LEDGER.json`/`.csv` and `STAGE_STATUS.json`/`.csv`, which ROOT
copies to the public-document folder before selection. Stage CSV uses
`stage,status` columns and JSON uses a `stages` list. Both must match the exact
eight-stage index and seven allowed status enums. Historical draft snapshots
remain preserved privately as input hashes.
Cluster CI, named-session sensitivity, and leave-one-cluster outputs
are preserved in the full contrast JSON. No statistics are recomputed. Public
cost metadata retains `UNKNOWN_NOT_ZERO` for unmeasured costs; completed fits do
not imply complete cost measurement.

After ROOT reviews the manifest hash, the separate copy command is authorized
only with both explicit flags. It creates a fresh private staging directory,
rejects source checkouts and source directories, verifies all copied hashes,
and includes `PUBLIC_CONTENTS.json` in the reviewed total size cap.

```sh
python -B -m scripts.research.pallet_bi_full_execution_v1.public_package copy --manifest RUN/FINAL_PUBLIC_MANIFEST.json --output RUN/FRESH_PRIVATE_STAGING --root-release-go --expected-manifest-sha256 ROOT_REVIEWED_SHA256
```

Only ROOT stages that exact allowlist into the worktree and performs Git commit
and push. Earlier draft manifests, source histories, execution receipts, and
superseded displays remain private and unchanged.
