# Known dimensions × valid symmetry: matched 2×2 output-layer screen

Completed on 2026-09-11: 2,048 synthetic training frames, 512 validation frames,
four arms, seeds1/2/3, 1,000 AdamW updates each. Total12,000 updates.
This is pretrained YOLO26 one-to-one last-layer adaptation. The detector,
backbone, spatial pose features, BN and flow remain frozen. No Hough branch,
full-network training, real evaluation or unseen-asset claim.

Run scripts from the repository root with the `pallet-yolo26` Python environment.
Each phase refuses incompatible or existing output where appropriate; completed
models are hash-bound. Do not overwrite this completed experiment to tune it.

Phases: `prepare.py freeze`, `prepare.py cache --limit 16`, `verify.py`,
`prepare.py cache`, `train.py`, `evaluate.py`, `audit.py`, `report.py`, `deliver.py`.
The GPU phases use the verified process-local driver launcher:

```bash
python scripts/diagnostics/with_gpu.py /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_dimension_symmetry_v1/train.py
```

This launcher does not reboot or change the installed driver. The current
runtime bundle is in ignored `.cache/nvidia-runtime/580.173.02`, with source
archive provenance and file hashes. A working system runtime bypasses it.

Result: a small dimension effect versus the matched fine-tuning control in all
three seeds (0.29–0.58% on the normalized primary). Symmetry switches the initial
target for only2 of1,364 C2 training objects. None passes the predeclared stable
improvement gate over both the matched control and the original YOLO. This does
not establish that dimension conditioning or symmetry-aware full-network
training is impossible. Full numerical and visual results are in
`data/pallet/results/pallet_dimension_symmetry_v1/` and
`_docs/notes/pose_geometry/dimension-symmetry-2x2-results.md`.
