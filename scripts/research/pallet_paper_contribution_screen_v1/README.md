# Paper-contribution screen v1

The authoritative protocol is the user master attachment SHA bound in
MASTER_PROTOCOL_LOCK.json. Read the current result pointer and reports before
attempting execution.

## Authorized completion

The user authorized one additional900-update repair of the lost C2 seed1 fit
and the remaining six scheduled fits. The current result pointer is
`_docs/experiments/pallet_paper_contribution_screen_v1/CURRENT_RESULT.json`;
completed evidence goes under `C_geometry_preserving_da/RESUME_COMPLETE/`.
The valid comparison is9x900 updates; cumulative C cost includes the lost900
updates, for9000 total. This does not authorize further refits or tuning.

Completion entry points (same environment; no additional training):

```text
track_c/aggregate.py --output-dir _docs/experiments/pallet_paper_contribution_screen_v1/C_geometry_preserving_da/RESUME_COMPLETE
track_c/verify_saved.py
track_c/complete_report.py
```

The aggregator verifies actual augmented-input parity; the independent audit
checks saved C2 frozen tensors and raw pose output against R0, and recomputes
all9 canonical CSV median/P90 scores. The report applies the frozen gate and
retains the original partial records below as historical evidence.

## Historical initial partial run

This run is **partial**: A candidate frozen; C wiring PASS but only C0/C1 seed1
checkpoints/evaluations retained; C2 seed1 consumed900 updates before an audit
bug lost its checkpoint. D/B/E mechanism gates FAIL. No automatic extra fit.

Entry points (pallet-yolo26 environment):

```text
common/initialize.py        # existing evidence bindings; already run
track_c/wiring.py           # CPU actual dependency and one-update probe
track_c/driver.py           # sequential GPU fits/eval; BLOCKED pending budget direction
track_c/aggregate.py        # requires all9 valid fits; intentionally NOT run
track_d/mechanism.py        # CPU frozen local teacher +3 trust fits
track_b/jacobian.py         # CPU canonical selector perturbation gate
track_e/alignment.py        # CPU non-ground RGB/depth boundary gate
common/finalize.py          # read executed artifacts, report partial scope
```

Run tests with `python -m pytest -q scripts/research/pallet_paper_contribution_screen_v1/tests`.
Raw artifacts/checkpoints remain local under the ignored new result directory.
JSON writes accept identical receipts but refuse different replacements. Fits
never overwrite checkpoints. The corrected trainer persists unverified final
weights before final assertions so an audit error cannot lose another fit.

GPU requires the already available process-only library setting
`LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace`; no reboot/driver changes.
Other projects' GPU processes must not be killed, modified or waited on.

`EXECUTION_BLOCKER.json` is retained as the historical driver stop guard.
The driver now requires the explicit `RESUME_AUTHORIZATION.json` receipt to
continue past it; deleting the blocker is not an authorized workaround.
