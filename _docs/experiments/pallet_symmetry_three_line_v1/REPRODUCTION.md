# Reproduction and scope

Environment used: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, Torch 2.1.1+cu118, RTX3080, driver 580.178.04. Start in the repository root. No installation or system update is required for this saved environment.

Read `SOURCE_BINDING.json`, `PROTOCOL_LOCK.json`, `A/ANNOTATION_TARGET_AMENDMENT.json`, `A/JOINT_TRAINING_LOCK.json`, `A/EVALUATION_LOCK.json`, `A/POSE_ADAPTER_LOCK.json`, and `B/NUMERIC_LOCK_AMENDMENT.json` before replay. Required checkpoints, image data, and ignored raw caches are local dependencies, not included in Git. The source binding is authoritative; do not substitute new weights or partitions.

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
| Approved A joint wiring | `a_joint_audit.py` | Complete two-head stock loss, shared GT-row branch, actual GPU gradient tests |
| A smoke | `a_train.py --stage smoke` | 12 optimizer updates total, discarded before main initialization |
| A main fits | `a_train.py --stage main` | 12 fits × 2,000; completed runs skipped, exact-state resume only |
| A training validation | `a_closeout_audit.py` | Weight/state/hash/paired sample order and all 24,000 updates |
| A last-weight inference | `a_infer.py` | 31,458 images, predictions serialized before GT scoring |
| A paired metrics and poses | `a_evaluate.py`, `a_pose.py` | Separate populations, full missing-prediction penalties, pose coverage |
| Frozen B source observations | `b_cache.py` | Existing cache entries skipped; report totals describe the dataset, not a fresh-invocation counter |
| B numerical gate | `b_numeric_diagnostic.py`, `b_validate.py` | Initial discrepancy preserved; validate includes full historical batch32 replay |
| Donor and calibration | `b_shuffle.py`, `b_evaluate.py` | Locked donor map and eta; immutable lock rejects a changed selection |
| Full real observation/evaluation | `b_real_cache.py`, `b_real_finish.py` | No FINAL; negatives inferred without GT-positive gating |
| Quality and subgroups | `b_diagnostics.py`, `b_real_diagnostics.py`, `b_subgroups.py` | Posthoc GT-assisted analysis, not inference inputs |
| Pose and visuals | `b_pose.py`, `b_visual.py` | Saved coordinates, no model retraining |
| Whole pipeline time | `b_runtime.py` | Run without competing GPU/CPU-heavy work; preserves parity checks |
| A matched/missed decomposition | `a_impact.py` | Posthoc explanation, not a replacement primary endpoint |
| Source orientation correction | `a_source_orientation_correction.py` | After A/B poses: fixed proper Y-up/Y-down basis; preserves pre-correction evidence |
| Final consistency | `finalize.py` | Rechecks source hashes and tests, does not start A training |

The source-only rotation correction was made **after** inspecting pose results, not preregistered. Renderer physical axes use +Y up while canonical PnP axes use +Y down: every source prediction receives the same proper right basis transform `diag(1,-1,-1)`. No GT-dependent phase selection is used. A pre-correction values are archived; original B files remain unchanged, with `B/POSE_SOURCE_ROTATION_CORRECTION.json` superseding only their synthetic rotation auxiliary. Primary 2D, centroid, body IoU, real DEV, weights and inference are unchanged. See `A/SOURCE_ORIENTATION_CORRECTION.json` and `A/POSE_EXECUTION_CORRECTIONS.json` for the complete correction/replay history.

Re-running an output-producing stage can replace this experiment's derived report, so archive/version those outputs first. Do not replay pre-calibration diagnostics and then claim their new timestamp preceded the already completed calibration. `NUMERIC_FLAGS_DIAGNOSTIC.json` and `HISTORICAL_BATCH_DIAGNOSTIC.json` also retain the original investigative probes; the production `b_validate.py` reproduces the decisive batch32 coefficient and batch1 posterior gates.

The user approved **joint** selection after the stock RLE nonseparability finding. The working EQUIV trainer preserves the original per-head batch-global RLE clamp and selects one shared rotation per GT row. All 12 main fits are complete. This is not a claim that the original independent-object-min objective and the approved joint objective are identical. Historical pending-state reports are preserved in Git and `A/history/`.

The initial full-val pose setup exposed one reflected camera-facing index basis (`G38__G__f11070`), before any A pose inference. `SOURCE_POSE_FRAME_AUDIT.json` records it. Source body-box GT now directly uses the existing proper renderer R and physical extents, an identical body volume representation; the frame is not excluded and reflection is not admitted as an equivalent rotation. For task-C4 square geometry, duplicate identical W/D hypotheses are collapsed before the unchanged SQPnP/LM solver; otherwise the rectangular selector would falsely reject a geometric tie. Both decisions are independent of model performance.

Large artifacts are in `data/pallet/results/pallet_symmetry_three_line_v1` (ignored). Detailed A JSON traces in the ignored `A/runs/` doc subdirectory are local; the tracked `training_step_trace.csv` and `training_audit.json` retain all update summaries, batch-ID hashes, run bindings, and audited counts. `VISUAL_CONTACT.png` is the lightweight tracked contact sheet; full 20 panels remain under the raw output. Git contains CSV/JSON evidence and code, not weights/features/raw images. Main-only commit/push verification is reported in the execution handoff; a commit cannot embed its own resulting SHA.
