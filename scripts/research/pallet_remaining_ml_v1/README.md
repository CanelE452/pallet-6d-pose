# Remaining two ML questions — fixed development screen

These are distinct from the completed active-learning sample acquisition.

1. `reweight.py`: learn online example weights from a disjoint meta loss via
   gradient alignment in the final keypoint-projection subspace. Compare with
   uniform and current-loss weighting. All three arms use the same actual
   synthetic/real/meta inputs, frozen BatchNorm statistics, and stock pose loss.
   Nine students × 300 updates, last checkpoint only.
2. `selective.py`: train a separate accept/reject gate for **frozen R0** poses.
   Confidence/reprojection baselines, one logistic model and three MLP seeds.
   Each learned gate receives 1,000 updates. No R0 update; no best-student choice.

Split:112 train /62 meta-or-calibration /145 evaluation, acquisition-session
connected components disjoint. Reused historical GT-v2, not independent test
or sensor-based safety certification. Positives only for pose acceptance;
background false acceptance remains untested. No deployment/model replacement.

Run in `pallet-yolo26` with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` and existing
process-only `LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace` for GPU commands.
Never change drivers, reboot, kill another GPU job, or silently repeat a fit.

Order: `experiment.py` (one-time immutable lock), `reweight.py prepare`,
`smoke.py` (zero updates), `reweight.py driver`, `evaluate_reweight.py`.
The independent CPU gate phases are `selective.py extract`, `fit`, `evaluate`.
Final closure is `report.py`; historical artifacts are never overwritten.

This is a reduced-subspace first-order adaptation of gradient-based example
reweighting, **not** an exact full-network reproduction and not a learned weight
MLP. The gate is a separate post-hoc classifier, **not** SelectiveNet's jointly
trained architecture. Prior-method boundaries:

- [Ren et al., Learning to Reweight Examples (ICML2018)](https://proceedings.mlr.press/v80/ren18a.html)
- [Geifman and El-Yaniv, SelectiveNet (ICML2019)](https://proceedings.mlr.press/v97/geifman19a.html)

The shared pose GT JSON must be parsed to retrieve train/calibration entries;
test-label errors are not computed or used until gate checkpoints and thresholds
are frozen. Camera intrinsics are read from annotation containers but no truth
coordinate, physical GT axis, outcome, session or filename enters gate features.
