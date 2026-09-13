# Task-Space Tail-Risk Active Adaptation v1

Bounded retrospective development experiment. Historical active-learning results,
labels, checkpoints and paper/final are immutable. All new outputs are confined
to this experiment's scripts, `_docs/experiments/` and `data/pallet/results/` roots.

Execution order (do not rerun completed inference or fits):

1. `contracts.py setup`, synthetic `test_contracts.py`, then `contracts.py lock`.
2. `mechanism.py phase1`: pool174-only historical selection hardness. Reserved145
   labels and mixed319 GT containers are denied.
3. `extract.py`: separate GT-denied process, exactly174×8 R0 image forwards,
   P0 bit-exact stock-cache parity, then immutable risk/cache SHA freeze.
4. `mechanism.py gate`: fixed targets/gates and10,000 paired bootstrap draws.
5. PASS only: `acquisition.py`, `prepare.py`, `train.py` (four300-update fits).
6. After all four: `evaluate.py`, `report.py`, `audit.py students`.
7. Review Korean report, commit only this experiment's scripts/docs, push main
   and verify actual local/remote SHA equality.

FAIL at step4 instead uses `audit.py` and terminates with zero new students.
There is no automatic perturbation, threshold, architecture or budget search.

## Environment

The existing working environment is `pallet-yolo26`. On this host GPU access
requires the existing process-local NVIDIA userspace path
`LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace`. This does not install drivers
or change system configuration. Use `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
`MPLCONFIGDIR=/tmp/pallet-task-risk-mpl`, `PYTHONDONTWRITEBYTECODE=1` and Python `-B`.
GPU checks stop on foreign compute or temperature>=80°C; no process killing,
reboot, sleep-inhibition or driver mutation occurs.

## Contracts and artifact names

`TASK_VIEWS.json` is the large perturbation cache; `TASK_RISK.json` is the
GT-free per-frame score cache. Both live under the ignored raw output root.
`RISK_FREEZE.json` binds them. Requested descriptive audit filenames are
exact-content aliases generated at closure, not independently recomputed analyses.

`prepare.py` exports canonical stock YOLO labels using unchanged historical
`export_target`; it does not literally copy GT JSON into YOLO TXT format.
`train.py` directly calls the unchanged historical training loop. Its wrappers
only assert synthetic input hashes and inspect GPU state after optimizer steps.
No changes to normal BN behavior, loss, augmentation, optimizer, or steps.

Fixed pool174 and reserved145 are historically consulted development data, not
untouched confirmation. One acquisition set is shared by three student seeds.
Full174 seed1 is a fixed-compute descriptive reference, never an upper bound.
