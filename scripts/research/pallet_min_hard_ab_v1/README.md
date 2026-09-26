# Minimal hard A/B — staged human workflow

Use the `pallet-yolo26` environment from the repository root.

```bash
python -m scripts.research.pallet_min_hard_ab_v1.cli prepare
python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty
python -m scripts.research.pallet_min_hard_ab_v1.cli resume
```

Tag raw RGB with C/M/S/U/X; answers autosave and advance, Z undoes the last answer in the current open round. No predictions, GT coordinates, teacher or PnP are displayed. Close the GUI before running resume. Completed rounds are immutable.

Resume opens another fixed round only if the declared hard/diversity conditions fail. Otherwise it locks initial8/reserve≤2 and enables:

```bash
python -m scripts.research.pallet_min_hard_ab_v1.annotate_hard
python -m scripts.research.pallet_min_hard_ab_v1.cli resume
```

Role-confident: drag the observed pallet envelope bbox, click DIRECT_VISIBLE P0–7 (auto-advance), O=occluded, X=out-of-frame, U=uncertain. No hidden coordinate completion. Role-uncertain frames are excluded. Finish each image explicitly. Reserve and raw-own-label QA are human-gated. Public reports never include exact manual coordinates.

The original first delivery stopped before teacher inference/training. The post-label workflow is now implemented:

```bash
python -m scripts.research.pallet_min_hard_ab_v1.lock_assisted_labels
python -m scripts.research.pallet_min_hard_ab_v1.run_ab
python -m scripts.research.pallet_min_hard_ab_v1.cli status
```

The user explicitly requested the existing `annotation.py` editor with PnP enabled, confirmed confidence in direct clicks, and approved shared PnP-derived association boxes. These amendments are documented in `ANNOTATION_PROTOCOL_AMENDMENT.json`. No PnP-generated corner or P8 is used as manual supervision. The 8-frame, 36-direct-click snapshot is private and immutable. Historical incomplete custom-GUI input is preserved separately. Do not call this PnP-blind manual ground truth.

`run_ab` runs teacher, paired cache construction, loss/gradient tests, two fresh R0-initialized fixed-budget fits, raw/pose freeze, scoring, report and completion audit in isolated processes. Completed immutable stages are reused; interrupted fits are never silently restarted. GPU stages require host CUDA access. No automatic production model replacement. `cli resume` runs this workflow from `HARD_LABELS_LOCKED_TRAINING_PENDING`; `run_ab` can also resume later stages explicitly. Real reference scoring is gated on raw and pose locks.

```bash
python -m unittest scripts.research.pallet_min_hard_ab_v1.test_workflow -v
python -m scripts.research.pallet_min_hard_ab_v1.audit
python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty --smoke
```

Smoke writes temporary responses only, not real human tags. The preparation `audit` is historical-stage specific and should not be rerun against completed human responses; use `completion_audit` for the final state. GPU is not needed for tagging preparation. Current training args/weights are immutable upstream bindings. New hard labels never change past evaluation/anchor files. Only this code and `_docs/experiments/pallet_min_hard_ab_v1` may be staged, including report figures. Private `data/pallet/results/...` is never pushed.
