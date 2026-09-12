# Frozen P/Q signed-gain selector

One bounded experiment on corrected v2 Hough seed1 proposals. Upstream is never
optimized. The selector regresses the gain of the **actual whole-layout P/Q
action**, using one baseline-selected C1/C2 assignment for both alternatives.

The observation-only feature builder enforces an allowlist and cannot accept GT,
frame/source/session IDs, or symmetry assignments. Training targets are separate
JSON artifacts. Test Q comes directly from the immutable corrected-v2 prediction.

From the repository root, use the environment that supports the existing v2 code:

```bash
python -m scripts.research.pallet_hough_gain_selector_v1.experiment lock
python -m scripts.research.pallet_hough_gain_selector_v1.audit tests
python -m scripts.research.pallet_hough_gain_selector_v1.experiment oracle
# Continue only if the locked oracle headroom gate passed:
python -m scripts.research.pallet_hough_gain_selector_v1.experiment cache
python -m scripts.research.pallet_hough_gain_selector_v1.experiment sanity
python -m scripts.research.pallet_hough_gain_selector_v1.experiment train
python -m scripts.research.pallet_hough_gain_selector_v1.experiment evaluate
python -m scripts.research.pallet_hough_gain_selector_v1.audit final
python -m scripts.research.pallet_hough_gain_selector_v1.audit report
```

`cache` and `train` use CUDA. On the current workstation these ran with
`LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace` and Python from the
`pallet-yolo26` environment, outside the sandbox that lacks GPU device access.
No reboot, driver replacement, or CPU training fallback is required.

Paths intentionally refuse to overwrite run artifacts. These are provenance
commands, not permission to rerun the bounded experiment or tune after failure.
Protocol/source hashes were locked before oracle inspection. New auditing and
reporting code does not modify the locked experiment or model.

See [the report](../../../_docs/experiments/pallet_hough_gain_selector_v1/REPORT_KO.md)
and [the protocol](../../../_docs/experiments/pallet_hough_gain_selector_v1/PROTOCOL_LOCK.json).
Caches/checkpoints live in the ignored `data/pallet/results/pallet_hough_gain_selector_v1`.
