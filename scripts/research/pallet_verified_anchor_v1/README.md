# Verified anchor — completed visible-corner evaluation

Correction on 2026-09-24: **WAITING_FOR_HUMAN_QA (two status conflicts)**.
The old evaluation below is preserved provisionally. See
[directive completion audit](../../../_docs/experiments/pallet_verified_anchor_v1/DIRECTIVE_COMPLETION_KO.md).
The missing metadata-conflict QA condition must be resolved before confirming the
final reference and adding the already-frozen teacher supplementary comparison.

Current status: **COMPLETED_VISIBLE_ANCHOR_EVALUATION**.
See [the result report](../../../_docs/experiments/pallet_verified_anchor_v1/EVALUATION_REPORT_KO.md)
for protocol, tables, limitations, and 26 embedded figures (18 case panels).

The user changed the initial protocol to existing `annotate.py` + PnP keypoint input,
followed by separate status review. This is PnP-assisted, not strictly blind labeling.
18 selected, 16 saved; 75 manual clicks, 72 in-frame reviewed, 66 DIRECT_VISIBLE.
No additional images; zero QA points; five frozen model caches scored without training/inference.
Unreviewed/nonmanual points are excluded, not filled with guessed statuses or ground truth.

Actual workflow (completed artifacts are immutable; do not rerun destructive preparation):

```bash
python -m scripts.research.pallet_verified_anchor_v1.open_existing_annotation
python -m scripts.research.pallet_verified_anchor_v1.review_saved_keypoints
python -m scripts.research.pallet_verified_anchor_v1.audit_completed_status
python -m scripts.research.pallet_verified_anchor_v1.evaluate
python -m scripts.research.pallet_verified_anchor_v1.report
python -m scripts.research.pallet_verified_anchor_v1.verify_evaluation
```

Exact inputs and coordinates stay private locally. `FINAL_DECISION.json` is current;
`DECISION.json` and `STATUS_QA_SUMMARY.json` preserve historical stage records.

## Historical initial preparation workflow (superseded by keypoints-first)

This implementation prepares the initial 18-image, model/reference-blind first pass.
No training, inference, legacy coordinate read, model comparison, or GT mutation.
Exact frame mappings, human coordinates and raw RGB stay local/private.

```bash
python -m scripts.research.pallet_verified_anchor_v1.prepare
python -m scripts.research.pallet_verified_anchor_v1.tests
python -m scripts.research.pallet_verified_anchor_v1.label_anchor
python -m scripts.research.pallet_verified_anchor_v1.validate_labels
```

Preparation is exclusive-create and cannot replace a previous selection.
After all human inputs are completed, `validate_labels --lock` verifies and locks them.
Subsequent second-batch routing, blind QA and scoring must be resumed explicitly;
they are not implemented or claimed complete at this human handoff.
Maximum total 24 images. Do not use these evaluation-only labels for training.

Annotation keys: 0–7 select corner; D/V/S/E/O/U choose status; D needs a click;
S/E/O/U forbid coordinates. Wheel zoom, right drag pan, F fit, Ctrl+Z undo.
Every edit autosaves and is appended to a local history. Image SHA is checked on load.
No status is pre-filled. P8 is excluded. Close/reopen resumes saved labels.

Name is optional (`anonymous_local` if omitted). Clicking an unreviewed corner explicitly
marks it DIRECT_VISIBLE and advances P0→P1→…→P7. S/E/O/U also advance;
All six status inputs D/V/S/E/O/U immediately advance, including D/V without coordinates.
For a later coordinate click, reselect the desired corner using 0–7. D still requires a
coordinate for completion; V is optional. P7 stays on the same image for review.
Undo restores the edited corner as well as its previous contents.
