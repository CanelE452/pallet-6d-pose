# Paper-contribution screen v1

The authoritative protocol is the user master attachment SHA bound in the new
MASTER_PROTOCOL_LOCK.json. Read the final reports before attempting execution.

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

`EXECUTION_BLOCKER.json` is deliberately a driver stop guard. Removing it is not
an authorized workaround: C2 seed1 already consumed its900-update budget.
