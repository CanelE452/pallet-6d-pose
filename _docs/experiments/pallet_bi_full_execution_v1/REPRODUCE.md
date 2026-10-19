# Reproduction and resume

Scientific production, all33 registered H1 fits and11 full238 prediction seals, canonical/independent evaluation, full bank/rank analysis, all registered derived outputs, error40 and the materialized report are sealed. The actual final readonly inventory is `FINAL_METADATA_READINESS_V1/REPRODUCTION_INVENTORY_FINAL_V1.json`; it records33 completed fits and no missing registered metadata paths. This is scientific/derived/report readiness. Publication, Git commit/push and remote-SHA confirmation remain ROOT steps: `whole_campaign_complete=false`, `publication_status=PREPARED_NOT_YET_COMMITTED_OR_PUSHED`.

## Inputs, sources and environment

The execution checkout is `/tmp/pallet-bi-full-execution-20261019-v1`, based on commit `3553202a3682faeb7be2e7df051b230683965382`; private run data is under `/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1`. The original dirty checkout `/home/minjae/Documents/github/pallet-pose` supplies explicitly bound local assets and is preserved. ROOT alone stages, commits and pushes the exact reviewed allowlist. Scientific sources stayed frozen through the recovery/resource amendments; source/protocol hashes belong to their actual historical version rather than a rewritten receipt.

Frozen initial environment: python=3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]; numpy=1.26.4; scipy=1.12.0; torch=2.1.1+cu118; opencv=4.9.0; platform=Linux-6.8.0-138-generic-x86_64-with-glibc2.35. Interpreter: `/home/minjae/anaconda3/envs/pallet-pose/bin/python`. Use `-B` and `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. These are recorded versions, not advice to upgrade or retrain. The instruction SHA is `208663999af68480922a6f8d2c9cfc1f26abcc6c267ca62356b6110099f6f88c`; campaign plan SHA is `e5b965e312fc96e6ef245d868285be2bf48e7ba4ee5a7dc5a612368d9485c023`.

All238 real inputs are retained: PRIMARY192=TEST49+REMAINDER143, with separate CAL46. Synthetic train768/validation235 keeps the original1003 identities. Production has no per-frame real GT pose/box/mask/visibility input. Registered CAL fitting and synthetic-label preparation remain distinct from postprediction real evaluation. Independent new-real test count and human physical-owner certifications are0. Private data, checkpoints, normalizers, registered assets, raw references and original RGB are not distributed with the small public package; local path mapping and the original authorized inputs are required for a rerun.

Six scientific families are S1, D/QCOV, S2, S3, S4 and F; P0 reuse and R reporting are separate status components. S1 fits3; S2 fits9; S3/S4 fit15; F fits6, totaling33 under the36-attempt cap. Each fit uses188→64→64→1/16321 parameters, three predetermined seeds20261018/19/20, train-only own normalization, AdamW and the frozen own-VAL checkpoint rule. Three head seeds repeat the same192 real frames and are not576 independent images. The two static PF estimators are each a single frozen output.

Actual F configuration is **S2_CP**, selected by the frozen own-VAL rule before F optional union/H1. It adds no optional PF/material/render actions. F still owns its new observation, CAL and fullnormal banks and its own six H1 fits; it does not transplant S2 validation performance or remove every downstream B/I feature. [Method details](METHODOLOGY_KO.md), [stage status](STAGE_STATUS.csv), [method differences](METHOD_DIFFERENCE_LEDGER.csv), and [D/S2 aggregate evidence](D_PUBLIC_SCALARS_README.md) describe the precise contrasts and limits.

## Recorded commands and read-only inventory

The final readonly inventory ran after the report seal closed; its separate execution receipt records the actual argv, CPU and wall time. The inventory contains196 literal argv records from36 metadata sources and39 separately identified Python API records. Record counts are not process invocation counts. The command below is the supported metadata-reader interface. It does not launch production, calibration, scoring, fitting, neural forward, rendering, plotting or master accounting; the example is not a substitute for the historical execution receipt.

```bash
cd /tmp/pallet-bi-full-execution-20261019-v1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-pose/bin/python -B \
  -m scripts.research.pallet_bi_full_execution_v1.reproduce \
  --run /home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1 \
  --metadata-sources /home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1/PUBLICATION_READINESS_PREPARATION_V5/REPRODUCE_METADATA_SOURCES_FINAL_CANDIDATE_V5.json \
  --snapshot NEW_FINAL_METADATA_SNAPSHOT.json
```

The exact V5 metadata source map is published as [PROVENANCE_INDEX.json](PROVENANCE_INDEX.json), SHA`9a957e21429d4a024cf7a6a0bc4ac14ba27a9fcc2bea3bf0ef5b9a2ca8d3efd6`. It identifies metadata files, receipt fields and readiness paths; it does not contain the private inputs. The map preserves its preparation-time pending language as history. The completed snapshot and terminal seals establish the later actual status.

With the original private RUN available, the published map can replace the private `--metadata-sources` path:

```bash
cd /tmp/pallet-bi-full-execution-20261019-v1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-pose/bin/python -B \
  -m scripts.research.pallet_bi_full_execution_v1.reproduce \
  --run /home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1 \
  --metadata-sources _docs/experiments/pallet_bi_full_execution_v1/PROVENANCE_INDEX.json \
  --snapshot NEW_READONLY_METADATA_SNAPSHOT.json
```

This is a supported mapped-file example, not an additional historical execution. Relative metadata paths resolve below `--run`; the few original absolute state paths also require local mapping when the private RUN moves. Preserve the published map's hash and keep any adapted copy under a new name.

The public [literal execution index JSON](EXECUTION_INDEX.json) and [CSV](EXECUTION_INDEX.csv) retain recorded argv, evidence hashes, receipt status, scope and selected versions/environment. They preserve Python APIs and missing historical argv as distinct evidence. The separate generated [source and artifact provenance index](PUBLIC_PROVENANCE_INDEX.json) describes the selected publication files.

Add `--show-commands` for exact literal argv and its receipt status. START, COMPLETE and multiple provenance records can describe the same process; their count is not an invocation count. Metadata `CHECKPOINT_ACTION` and delegated Python API calls are never rewritten as executed CLI. Historical P0, D and some earlier S1 scoring launch argv remain `NOT_RECORDED_ARGV` where no literal journal exists; exact source/output/API settings remain evidence and plausible reconstructed argv is not substituted.

| Scope | Actual command/receipt source | Interpretation |
|---|---|---|
| S1 | `S1_EXECUTED_COMMANDS.json`; own training/prediction seals | Three actual fits and714 tiny-head real forwards; original bank and saved backbones reused |
| D | Original/fixedΓ source/output seals; later `D_SCALAR_EXPORT_EXECUTION` actual exporter ledger | Observation seed18 limited-family diagnostic; other two seeds have saved observation-change counts, no invented pose ablations |
| S2/F original D3/D4 | Historical live argv snapshots, original controllers/locks, closed attempt receipts | Historical processes and timing are preserved; the original eight interrupted arm directories retain unknown unrecorded work |
| S2/F actual recovery | `ROOT_FULLNORMAL_RECOVERY_SERVICE_LAUNCH.json`, recovery invocation locks and completion receipts | Actual ROOT user services use unchanged numerical row helpers; copied intrinsic costs are not new calls |
| S2 H1/final sequence | `ROOT_POST_CORE_RECOVERY_SERVICES_LAUNCH.json`, `FINAL_SEQUENCE_RECOVERY_V1/ACTUAL_COMMAND_LEDGER.jsonl` | H1 prepare/train/predict inside the S2 script are actual Python APIs, not separately launched child CLI |
| S3/S4 | `s3_s4_production/ACTUAL_ARGV.json`; postprocess and root postseal ledgers | Full1241 producer, five banks and15 fits; original and supplemental completion-ledger prefix bindings remain intact |
| F finalization | `F_FINALIZE/POSTPROCESS_LEDGER.jsonl`, choice/finish seals | ROOT choose/union/labels/own6fits/predictions with the selected fixed common auxiliary policy |
| Derived completion | `ROOT_DERIVED_COMPLETION_ACTUAL_LAUNCH.json`; `DERIVED_COMPLETION_V1/ACTUAL_COMMAND_LEDGER.jsonl` | Actual outer service argv; eight sequential saved-output stages; controller `actual_argv` and API metadata stay separate |
| Cost/error40 | `ROOT_COST_WAITER_ACTUAL_LAUNCH_V2.json`, `ROOT_ERROR40_WAITER_ACTUAL_LAUNCH.json`, own child ledgers | Separate metadata/scalar children with recorded CPU/wall; late error40/publication work is outside the earlier master cutoff |
| Final report | `ROOT_FINAL_REPORT_ACTUAL_SPEC_PREPARE_V1.json` and `ROOT_FINAL_REPORT_ACTUAL_EXECUTION_V1.json` | Actual `actual_argv` is recorded in both completed receipts; report is `FINAL_REPORT_MATERIALIZED_V1/REPORT_SEAL.json` |

RUN-owned exact controller and ROOT launcher/lifecycle source is selected under `execution_sources`. Its byte hashes are preserved even when original absolute paths remain; map local run/worktree/interpreter/input paths before an independent rerun. Exact numerical source/packet/model/cache bindings are indexed in [public provenance](PUBLIC_PROVENANCE_INDEX.json); the code-source index does not publish private tensors or raw data.

## Recovery, resources and resume

The actual recovery adapter is `fullnormal_recovery.py` SHA`5c5914b4f7dbcaea027b34cb195c2eee79437f4f25c74149e16cceec70c39eb1`; the frozen numerical row helper is retained. S2/F recovery each used four workers with BLAS1. ROOT's CPU-placement amendment pinned S2 to logical CPUs0/2/4/6 and F to1/3/5/7, eight distinct physical cores, without restarting workers or changing numerical source, packets, caps or cache identity. This placement change is not a measured speedup claim. ROOT owns resource and technical-resume decisions; a user pause/STOP never authorizes automatic resume.

H1 durable `resume.pt` retains the same model/optimizer/RNG/shuffle/batch cursor and accumulated counters. A resumed optimizer is the same registered attempt. Preserve `best.pt`, `last.pt`, `HISTORY.json`, receipts, training manifest, labels and train normalizer together. Completed calls authenticate and return saved outputs. This metadata preparation opened none of those tensor/data payloads.

Raw material production is complete and its extras writer is not rerun. Completed controllers authenticate exact source/input/checkpoint/artifact bindings and return without appending old terminal ledgers. Partial nonempty results, old partial seals and technical logs are preserved for ROOT disposition. Earlier F resume adapters that were only implemented/tested are not claimed as executed; the actual recovery service launch is the historical execution proof.

Mesa llvmpipe/EGL is CPU software rendering, not hardware GPU rendering. The S3/S4 producer used its recorded LP4 resource amendment. PL uses native point/typed-normal plus pose-updated native outline and does not call the proxy renderer; CONTOUR/IMAGE do use their registered proxy rendering. Asset identity/signed axes/owner remain uncertified.

## Calls, elapsed time and output scope

Independent aggregate metadata arithmetic agrees: **479400 historical closed calls +435838 S2 cumulative-base calls +347465 F cumulative-base calls =1262703 known fullnormal minimize calls**. Equivalently,479400+112387 preserved prefix+670916 new recovery calls gives the same lower bound. The112387 prefix is already included in the two cumulative bases and must not be added a second time. Original interrupted unrecorded work is `UNKNOWN_NOT_ZERO`;1262703 is a lower bound, not a complete actual campaign count. D's786 oldΣ+Γ plus64 fixedΓ calls are a separate850 diagnostic calls. S3 objective/Jacobian, DIS/PnP/render and tiny-head/backbone forwards have different units and are never added into one grand call count.

The master measurement cutoff is not final publication/Git completion. Preserve its exact scope, historical unknowns and parent/child accounting: inclusive service CPU contains child CPU; overlapping service elapsed, sums of child wall and internal fit timers are not campaign elapsed. Error40 waiter/report/publication work has separate actual receipts outside the declared master accounting inventory;this does not assert that their wall intervals never overlap the cutoff interval. ROOT's final post-push completion/remote SHA is recorded outside the commit to avoid a circular self-completion claim. [Typed cost summary](tables/costs/PUBLIC_COST_SUMMARY.json) and [derived-service accounting](tables/costs/DERIVED_SERVICE_PUBLIC.json) retain these distinctions.

Canonical real evaluation follows sealed predictions; independent quaternion/atan2 checks reuse the same population and are numerical validation, not new-real generalization. Four populations, sample variance/SD, all failures/no-pose/fallback and all available outliers remain in [all160 metric rows](tables/PUBLIC_METRICS40/ALL_POPULATION_METRICS.csv). Full30 paired contrasts retain recording/named-session10000-draw intervals and leave-one-cluster sensitivity. Error40 is descriptive saved-scalar accounting, not a new preregistered selection test.

All11 galleries retain every three-seed PRIMARY recovery/damage ID plus fixed normal/failure/tail rules; both scatter bundles retain full axes/outliers with T(cm) horizontal/R(deg) vertical. Generation seals and ROOT's limited actual-pixel/AI review are separate. They do not certify every image or physical ownership. No CAL RGB, original/private24 RGB, reference R/t, neural checkpoints, assets or full banks are copied. The narrow approved case JSON retains useful q/covariance/scalar reliability/source metadata.

This document does not run the automatic finalizers or public selector. ROOT reviews the actual final report and seals, explicit size policy, all-case manifests, privacy/local-link audit and exact fresh staging allowlist before commit/push. [Final report](FINAL_REPORT_KO.md) provides results; this document provides the execution and evidence path.

## Complete derived display index

Every gallery keeps all registered three-seed recovery/damage cases and deterministic normal/failure/tail examples. Each links its derived case metadata, redacted full context and family readouts. ROOT limited pixel QA and privacy approval remain separate from generation and do not certify physical ownership.

- [S1 original H1 vs PF union](galleries/S1_ORIGINAL_H1_vs_PF_UNION/index.html)
- [S3/S4 vs original H1](galleries/S3_S4_ORIGINAL_H1_vs_UNIONS/index.html)
- [S3 COMMON vs BI](galleries/S3_COMMON_vs_BI/index.html)
- [S4 PL vs CONTOUR](galleries/S4_PL_vs_CONTOUR/index.html)
- [S4 PL vs IMAGE](galleries/S4_PL_vs_IMAGE/index.html)
- [S4 CONTOUR vs IMAGE](galleries/S4_CONTOUR_vs_IMAGE/index.html)
- [S2 vs original H1](galleries/S2/index.html)
- [F vs original H1](galleries/F/index.html)
- [F COMMON vs BI](galleries/F_BI_VS_COMMON/index.html)
- [S2 C0 vs CP](galleries/S2_CP_VS_C0/index.html)
- [S2 CP vs CB](galleries/S2_CB_VS_CP/index.html)

[Full69-dataset T-horizontal/R-vertical scatter](figures/FULL_T_horizontal_R_vertical/index.html) · [Earlier30-dataset S1/S3/S4 scatter](figures/S1_S3_S4_T_horizontal_R_vertical/index.html) · [Saved orientation supplement](figures/S1_orientation/ORIENTATION_NATIVE_AXES.jpg) · [Final report](FINAL_REPORT_KO.md) · [Typed cost summary](tables/costs/PUBLIC_COST_SUMMARY.json).
