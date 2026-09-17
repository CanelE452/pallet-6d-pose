# Execution and reproduction

This is the new experiment root only. Existing R0, OLD_P and all source caches are read-only. Start SHA is in `TRAIN_PROTOCOL_LOCK.json`; live state is `TRAIN_PROGRESS.json` and `PIPELINE_STATE.json`. In-progress artifacts are not a completed experiment.

Environment: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B`, `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/pallet-dcp-mpl`. Use the repository root as cwd. No installs, driver changes or reboots.

Execution order:

1. `prepare.py`: 60k canonical dimension/symmetry joins, original protocol/order/checkpoint binding, train-only normalization. No old feature-array writes.
2. `train.py tests`, then `train.py smoke`: CPU regressions and actual N0/N4 100-step checks. Smoke checkpoints are discarded, never used as main initialization.
3. `diagnostic_locks.py`: freeze GT-independent metadata donors and measure train-only target locality.
   `actual_adapter_audit.py` additionally checks actual cached TRAIN rows: zero-logit parity for all DIM/META arms/seeds, first-step scorer gradient and second-backward encoder gradient. Its nine one-step CPU test models are discarded and accounted separately, never used as fitted checkpoints. This extra audit ran after main launch but before performance evaluation.
4. `train.py paper`: N0–N4 × three seeds ×6000 steps. Paired exact original order, final checkpoint only. CUDA grid-sample backward can be nondeterministic.
5. `pipeline.py`: the declared phase2–5 evaluation/secondary workflow, with stage logs under the ignored raw root. It requires completed paper training. `--wait-pid` is a read-only wait for the already running paper trainer, not a new training launch.
6. Scoped main commit/push and SHA confirmation are done separately after final audit, not by the training runner.

The runner skips completed stages and stops on an error. Training saves exact optimizer/RNG/state every500 updates and at completion; a restart checks code/protocol/order bindings. Do not rewrite a lock to resume changed training. Record correctness repairs separately; no result-driven tuning.

Artifact scope: code/docs/light JSON records are tracked. Weights, source feature files, raw predictions and large sidecars under `data/pallet/results/pallet_dim_conditioned_p_v1` are ignored local prerequisites, not Git payloads. `SOURCE_BINDINGS.json` binds original P source/checkpoints/order and records cache-array size/mtime; the unchanged canonical cache manifest/shard system supplies its existing data integrity contract. The current track does not falsely claim a new whole-cache SHA audit.

Deployment assumption: object type, canonical size and approved task group are supplied externally. DEV object_type comes from the fixed manifest and is used only to index the registry; this is a known-object conditional scenario, not a learned object classifier. Unknown type is explicit failure, not silent fill. GT phase/pose/clicked geometry are not network inputs.

Calibration ambiguity is resolved before results: use the same original fixed-index P candidate CE on the calibration split for all arms. Symmetry phase assignment is TRAIN-only. Original lambda/cap is shared. Square has no independent calibration population, so T=1 is fixed; mixed uses synthetic calibration only.

Canonical-input correction: the referenced exporter returns camera-facing G38 dimensions, not canonical object axes. Before any DIM main initialization/update, all40,000 G38 inputs were checked against the original `fixed_dimensions` builder and replaced with the immutable fixed-renderer-XYZ geometry table (WDH=XYZ[0,2,1]). Legacy20,000 inputs were unchanged. Renderer index metadata is used only to reconstruct/check fixed axes offline; neither GT pose nor a selected phase is read at inference. Sorted physical extents alone were insufficient to verify this contract.

Execution history: six metadata-free N0/N1 fits were retained. A missing protocol gate stopped the next DIM fit before initialization, then `correct_canonical_dimensions.py` archived the old sidecar/normalization/protocol and invalid N4 smoke, corrected inputs, and restored the lock. Corrected N4 smoke and actual-cache adapter checks both passed before DIM main training. The corrected `prepare.py` now uses fixed axes from the outset for a fresh run; do not rerun it over this historical locked root. Closeout verifies this run's correction history when present; a clean reproduction instead verifies corrected fixed-XYZ provenance and does not invent archived invalid runs. Use an isolated workspace/output root for a new reproduction, not this completed run's artifacts.

Accounting: main30×6000=180,000 updates; valid smoke200 plus archived invalid smoke100; two discarded nine-model CPU adapter audits=18 updates. Total actual optimizer calls=180,318. The earlier N0 smoke and six metadata-free main fits did not read dimension context and remain valid.
