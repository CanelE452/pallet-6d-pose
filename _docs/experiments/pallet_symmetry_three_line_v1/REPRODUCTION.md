# Reproduction and scope

Environment used: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, Torch 2.1.1+cu118, RTX3080, driver 580.178.04. Start in the repository root. No installation or system update is required for this saved environment.

Read `SOURCE_BINDING.json`, `PROTOCOL_LOCK.json`, `A/ANNOTATION_TARGET_AMENDMENT.json`, and `B/NUMERIC_LOCK_AMENDMENT.json` before replay. Required checkpoints, image data, and ignored raw caches are local dependencies, not included in Git. The source binding is authoritative; do not substitute new weights or partitions.

Read-only CPU regression:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/pallet-three-line-mpl \
 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m unittest discover \
 -s scripts/research/pallet_symmetry_three_line_v1 -p 'test_*.py' -v
```

The following are the dependency stages actually executed, not an instruction to rerun all GPU work over saved evidence. Call a script with the same Python/environment prefix above. GPU stages check foreign compute and temperature; they do not terminate other jobs.

| Stage | Entry points | Notes |
|---|---|---|
| Lock and geometry | `preflight.py`, `a_geometry_audit.py` | Immutable contracts; original sources read-only |
| A0 and identity | `a0_square.py`, `a0_paper.py`, `a_identity_audit.py` | A0 uses existing prediction caches when present; no training |
| Square lineage correction | `data_provenance.py`, `a0_annotation_correction.py` | Separate sidecar, never rewrite dataset |
| Frozen B source observations | `b_cache.py` | Existing cache entries skipped; report totals describe the dataset, not a fresh-invocation counter |
| B numerical gate | `b_numeric_diagnostic.py`, `b_validate.py` | Initial discrepancy preserved; validate includes full historical batch32 replay |
| Donor and calibration | `b_shuffle.py`, `b_evaluate.py` | Locked donor map and eta; immutable lock rejects a changed selection |
| Full real observation/evaluation | `b_real_cache.py`, `b_real_finish.py` | No FINAL; negatives inferred without GT-positive gating |
| Quality and subgroups | `b_diagnostics.py`, `b_real_diagnostics.py`, `b_subgroups.py` | Posthoc GT-assisted analysis, not inference inputs |
| Pose and visuals | `b_pose.py`, `b_visual.py` | Saved coordinates, no model retraining |
| Whole pipeline time | `b_runtime.py` | Run without competing GPU/CPU-heavy work; preserves parity checks |
| Final consistency | `finalize.py` | Rechecks source hashes and tests, does not start A training |

Re-running an output-producing stage can replace this experiment's derived report, so archive/version those outputs first. Do not replay pre-calibration diagnostics and then claim their new timestamp preceded the already completed calibration. `NUMERIC_FLAGS_DIAGNOSTIC.json` and `HISTORICAL_BATCH_DIAGNOSTIC.json` also retain the original investigative probes; the production `b_validate.py` reproduces the decisive batch32 coefficient and batch1 posterior gates.

There is deliberately **no working EQUIV trainer or claimed trained A checkpoint**. `a_loss.py` supports the verified identity adapter and guards the unresolved EQUIV objective. Approval of the joint branch objective is necessary before its implementation, wiring tests, and all 12 main fits.

Large artifacts are in `data/pallet/results/pallet_symmetry_three_line_v1` (ignored). `VISUAL_CONTACT.png` is the lightweight tracked contact sheet; full 20 panels remain under the raw output. Git contains CSV/JSON evidence and code, not weights/features/raw images. Main-only commit/push verification is reported in the execution handoff; a commit cannot embed its own resulting SHA.
