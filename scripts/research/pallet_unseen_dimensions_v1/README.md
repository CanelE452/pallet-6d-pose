# Unseen mesh dimension / geometry residual screen

This is a new, bounded synthetic screen, separate from the previous dimension × symmetry output-layer experiment. Frozen YOLO has seen four assets; residual training uses scene.usd/scene_1.usd, test additionally uses scene_2.usd/scene_3.usd, absent from this checkpoint's audited training lineage. These meshes exist in other historical pools, so this is exploratory, not an untouched paper FINAL.

Run from the repository root, using the pallet-pose environment for rendering and pallet-yolo26 for other stages:

```bash
env LD_LIBRARY_PATH=/tmp/pallet-v44-cuda-libs PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json /home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/pallet_unseen_dimensions_v1/render.py
python scripts/diagnostics/with_gpu.py /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/cache.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/data_audit.py
python scripts/diagnostics/with_gpu.py /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/train.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/evaluate.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/audit.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/diagnose.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_unseen_dimensions_v1/report.py
```

The local NVIDIA path is required only for this machine's loaded580.173.02/installed580.178.04 mismatch. It does not install drivers or reboot. GPU calls need actual device access. Use a new output directory for a new experimental protocol; preserve completed artifacts.

Inputs are predicted points, point features, ROI features and known fixed dimensions. No target boxes, GT camera-facing dimensions, GT pose or target permutation enter the network. Four arms share parameter initialization, minibatches and fixed1200-update budgets across3seeds. The backbone and detector never update. All frames, including missed detections, remain in primary scoring. The GT-based permutation and residual-cap oracle diagnostics are explicitly posthoc and never counted as model performance. Camera-dynamic corner IDs remain the evaluation target; this does not establish fixed physical semantic pose.

Results: `data/pallet/results/pallet_unseen_dimensions_v1/`. Interpretation: `_docs/notes/pose_geometry/unseen-dimensions-screen.md`.
