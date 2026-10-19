`final_report.py` is an unbound reporting helper; none of the frozen747 scientific sources is changed. The module uses only the standard library. It copies closed public scalar summaries and paired results, authenticates canonical/independent receipts, and writes a fresh report. It never runs a predictor, pose metric, optimizer, renderer, calibration, bootstrap or F selector. Pose/reference/model payloads are never numerically decoded; recorded file SHAs can be rechecked byte-for-byte for ancestry.

The implementation preparation folder is `RUN/FINAL_REPORT_SUPPORT_V1`. `REPORT_SPEC_TEMPLATE.json` is explicitly a template, not a completion result. The saved `PREPARE_REPORT_SPEC.py` is a future metadata-only preparation script. No actual report has been materialized by the authoring agent. Historical report/ledger drafts remain unchanged.

The real prerequisites are the full12-stage final sequence; full8-stage derived completion; PUBLIC_METRICS40 (40arms×4 populations,160 summaries,9520 saved scalar rows); all30 paired contrasts in each of4 populations; the final40 six-file error budget; the completed external cost waiter/actual33 fits/11 prediction seals/7854 tiny real H1 forwards; actual frozen F own-VAL choice; and completed generic post-bank metadata for S1/S3_S4/S2/F. An initial25 export or error budget is rejected.

The report answers the eight TXT questions. It leads with overall F_COMMON/F_BI versus the original and PF anchors, separately reports BI−COMMON, keeps all192 outcomes and the fixed31/41/38/PF-new3/old12 ID cohorts, and retains sample variance/SD, available-only missingness and all tails. PRIMARY and CAL remain development data; independent_real_test_n=0. The actual choice is copied from the frozen own-VAL receipt, never chosen from the PRIMARY scores. N3/SubPix remain fixed, with an explicit retain recommendation because no removal experiment exists. Native S remains model projection. PL does not use the CAD renderer; IMAGE feature/rank associations are descriptive. Missing/uncertified assets and physical ownership remain NA reasons, not whole-stage blockers. Source/pixel dedup does not certify physical independence, and UNKNOWN_CORRELATION is disclosed.

Five public appendices are exact REPORT_SEAL-bound files: `FULL40_PRIMARY_METRICS.csv` (40), `REPRESENTATIVE_PRIMARY_METRICS.csv` (fixed seed18 plus static baselines,16), `PAIRED30_PRIMARY_COMPACT.csv` (30), `FIXED_COHORTS.csv` (165), `FIXED_COHORTS.json` (all five fixed ID cohorts). Full160 values are already selected at `tables/PUBLIC_METRICS40/ALL_POPULATION_METRICS.csv`; they are not duplicated. Three Markdown documents and four metadata documents have separate exact SHA maps. The metadata files are final `METHOD_DIFFERENCE_LEDGER.json/.csv` (38 rows) and `STAGE_STATUS.json/.csv`. The historical ledger's source/status evidence is preserved while current callable/line/SHA and actual completion binding are added. Machine stage statuses use only the seven registered enum values.

The private `REPORT_SEAL.json` schema is `FINAL_KO_REPORT_SAVED_SCALAR_SEAL_V1`: `complete`, `scientific_and_derived_complete`, `report_ready` are true; `whole_campaign_complete` is false and `publication_pending` true. `artifacts` is the exact five-appendix map, `documents` the exact three Markdown map, `metadata_documents` the exact four metadata map. It binds metrics/paired/error-budget/cost seals, four post-bank aliases S1/S3_S4/S2/F and all private input hashes. All raw-reference/pose-numeric/model-tensor reads and new metric/solve/fit/forward/render/bootstrap/F-choice counters are zero. It is not a claim that Git/publication is finished.

After the actual terminal gates exist, ROOT can prepare a fresh runnable spec. This recipe has not been executed during implementation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/minjae/anaconda3/envs/pallet-pose/bin/python -B \
  RUN/FINAL_REPORT_SUPPORT_V1/PREPARE_REPORT_SPEC.py \
  --template RUN/FINAL_REPORT_SUPPORT_V1/REPORT_SPEC_TEMPLATE.json \
  --output RUN/FINAL_REPORT_SPEC_V1.json
```

ROOT then writes a fresh `FINAL_REPORT_ROOT_GO_V1.json` containing `approved:true`, `metadata_only:true`, `source_sha256` equal to the reviewed `final_report.py` SHA and `spec_sha256` equal to that newly prepared spec SHA. The executable call is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/pallet-bi-full-execution-20261019-v1 \
  /home/minjae/anaconda3/envs/pallet-pose/bin/python -B \
  -m scripts.research.pallet_bi_full_execution_v1.final_report \
  --spec RUN/FINAL_REPORT_SPEC_V1.json \
  --root-GO RUN/FINAL_REPORT_ROOT_GO_V1.json \
  --output RUN/FINAL_REPORT_MATERIALIZED_V1
```

ROOT records the literal command and outer CPU/wall including output writes. The output directory must be new. The spec's relative-link allowlist must match the final selected files. ROOT reviews the actual prose, copies the exact three Markdown and four metadata document bytes into the new report namespace, and passes their SHAs plus the five appendix source files/seal to the final public selector. The selector authenticates these again and performs no scientific work. No package copy, Git commit/push or final workflow completion is performed by this reporter.
