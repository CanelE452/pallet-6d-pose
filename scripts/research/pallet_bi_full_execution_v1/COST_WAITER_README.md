`cost_waiter.py` is a new standard-library-only controller. The frozen `master_costs.py` SHA remains `3c5aa52cab971a50d4cae16892636c43304e6b7573b30fdd2a5509a02b1713d9`; the existing V3 catalog and README remain historical. No scientific producer, model, pose solver, renderer, reference decoder or public publisher is invoked by this controller.

Preparation alone wrote the explicit V4 metadata catalog and historical `MASTER_COSTS_WAIT_V1/PLAN.json`. ROOT's review added a repeated catalog-hash guard before execution; the original source/test/README bytes and V1 plan are preserved. The runnable amended plan is `MASTER_COSTS_WAIT_V2/PLAN.json`, binding the same V4 catalog as a fixed input. V4 adds the completed derived controller's child-command ledger and the later external fifth-service scalar receipt. These are nested scopes: derived child CPU/wall is contained by the inclusive fifth-service cgroup CPU/lifetime and must not be added again.

Runtime requires a fresh ROOT JSON GO with `approved: true`, `metadata_only: true`, `wrapper_source_sha256`, `master_source_sha256`, `catalog_sha256`, `plan_sha256`, `systemctl_unit: pallet-bi-derived-completion-20261019-v1.service`, `master_child_commands_authorized: 1` and `new_scientific_calls_authorized: 0`. The plan and GO bind all current sources/protocols before any child launch. The child explicitly receives `PYTHONPATH` equal to the execution worktree and BLAS/OMP/MKL threads1.

The waiter first requires `DERIVED_COMPLETION_V1/DERIVED_COMPLETION_SEAL.json` and `ROOT_SERVICE_LIFECYCLE_V1/COMPLETION.json`. It validates the 12 upstream sequence stages and 8 derived stages, PLAN/GO/source hashes, checkpoint and artifact hashes, full declared scalar/gallery/scatter populations, command ledgers and all four successful, non-restarted service terminal snapshots. This authenticates saved completion and bytes; it does not independently recompute pose metrics or visually inspect pixels. An AI/private24 qualitative review is not human physical ownership certification.

Only one systemd unit can be queried: `pallet-bi-derived-completion-20261019-v1.service`. The command is the fixed `systemctl --user show` property whitelist; the controller never starts, stops, restarts or kills any service. It waits until that fifth service exits, captures actual inclusive cgroup CPU, monotonic start/exit elapsed, status and restart count, and binds the actual ROOT service launch receipt. Failed exit costs are preserved, then the success gate stops the workflow. Successful capture produces private `ROOT_DERIVED_SERVICE_FINAL_CAPTURE.json` and a bounded public scalar copy `ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json`.

It then writes `ROOT_COST_ELAPSED_CUTOFF.json` with `campaign_complete: false` and scope `scientific+derivedcomplete;cost/report/package/Gitpending`. Its time is an explicit measurement cutoff, never final workflow elapsed. A delegated master-only ROOT GO binds the unchanged master source, V4 catalog, upstream completion hashes and cutoff. The frozen auditor runs once in a fresh `MASTER_COSTS_FINAL_V1` namespace.

The wrapper's child ledger measures actual master subprocess start-to-exit CPU/wall, including import, output encoding/write and process teardown omitted by its internal timer. These external timers contain the master's internal timers and must not be added again. Query subprocess timers are also preserved in the separate query ledger. The wrapper's own final receipt-write CPU remains `UNKNOWN_NOT_ZERO` unless a later external service capture measures it. Original interrupted solver costs and old untracked import/finalization remain unknown.

The final readiness check requires actual33 completed fit receipts, all declared prediction seals and actual7,854 tiny real H1 forward records from the frozen auditor. It does not count planned work as complete. A partial waiter or master output is preserved and requires explicit ROOT disposition; it is never silently repeated. A completed wrapper returns only after authenticating its recorded outputs and upstream closures.

ROOT, after reviewing and authorizing the wrapper, can launch the run command through its own service:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/pallet-bi-full-execution-20261019-v1 \
  /home/minjae/anaconda3/envs/pallet-pose/bin/python -B \
  -m scripts.research.pallet_bi_full_execution_v1.cost_waiter run \
  --plan RUN/MASTER_COSTS_WAIT_V2/PLAN.json \
  --root-GO RUN/MASTER_COSTS_WAIT_ROOT_GO_V2.json
```

This is a future invocation recipe. The authoring agent prepares metadata and runs temporary fixture tests only, with all subprocesses mocked in the end-to-end fixture. No actual wrapper service, systemctl query or full cost audit is started during implementation.
