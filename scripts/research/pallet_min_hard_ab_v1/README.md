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

This first delivery implements inventory/tagging/selection/manual-entry/label validation tooling. It **does not yet implement or claim the post-label-lock teacher inference, matched fits and A/B evaluation**. Resume stops at `HARD_LABELS_LOCKED_TRAINING_PENDING` for that next implementation/execution stage, rather than simulating completed training. Do not bypass the human gates.

```bash
python -m unittest scripts.research.pallet_min_hard_ab_v1.test_workflow -v
python -m scripts.research.pallet_min_hard_ab_v1.audit
python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty --smoke
```

Smoke writes temporary responses only, not real human tags. GPU not needed for preparation. Current training args/weights are immutable upstream bindings. New hard labels never change past evaluation/anchor files. Only this code and `_docs/experiments/pallet_min_hard_ab_v1` may be staged, including its two progress figures. Private `data/pallet/results/...` is never pushed.
