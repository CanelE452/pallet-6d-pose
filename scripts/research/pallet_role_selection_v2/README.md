# Whole-object corner selection + independent residual

Implemented and evaluated: four corner-ID candidates, learned risk selection, a separate16px residual. No coordinate averaging, no GT at inference, no Hough, no physical C4 symmetry claim. Frozen YOLO. Same model inputs as `pallet_unseen_dimensions_v1`.

Stages are preserved separately:

1. `verify.py`, `render.py`, `cache.py`, `train.py`, `evaluate.py`, `audit.py`: four arms×three seeds, Huber selector, original1536training proposals. New576observations at seed651000.
2. `mean_risk_train.py`, `mean_risk_render.py`, `mean_risk_cache.py`, `mean_risk_evaluate.py`, `training_diagnosis.py`: replace selector loss with MSE, exactly freeze old coordinate residual, three seeds. New576observations at seed851000.
3. `foreground_train.py`, `foreground_render.py`, `foreground_cache.py`, `foreground_evaluate.py`: supervise corners only at GTIoU>=.5proposals(1400/1536), no confidence threshold. Direct vs selected+residual, three seeds. New576observations at seed1051000. Every evaluation retains misses.
4. `conditional_analysis.py`, `data_audit.py`, `final_audit.py`, `report.py`, `deliver.py`: conditional statistics, image/GT replay, final replay checks and visible report.

Run from repository root. The renderer uses pallet-pose environment with process-local EGL libraries:

```bash
env LD_LIBRARY_PATH=/tmp/pallet-v44-cuda-libs PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json /home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/pallet_role_selection_v2/render.py
```

Training and feature extraction use pallet-yolo26, e.g.:

```bash
python scripts/diagnostics/with_gpu.py /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_role_selection_v2/train.py
```

Evaluation and audits use pallet-yolo26 without GPU. The compatibility paths are only required for this host's loaded580.173.02/installed580.178.04 mismatch. No reboot or system driver installation. A new reproduction should use distinct output directories: preserve completed protocols and checkpoints.

Final result: `data/pallet/results/pallet_role_selection_v2/foreground_matched/SUMMARY.json`. Interpretation: `_docs/notes/pose_geometry/role-selection-and-residual.md`. These are synthetic new scenes of two seen and two unseen-to-this-checkpoint meshes, not independent real sessions. Original four-arm and intermediate results remain available; no failing run was relabeled as the final successful architecture.
