# V29 adaptive point / guided-line / independent-line experts

Run with `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`.

```bash
python scripts/research/pallet_adaptive_experts_v29/run29.py
```

The driver resumes completed stages, runs fixed synthetic training and evaluation,
builds the report, verifies the visible Chrome window and sends the previously
requested completion-only Discord message.

- `train29.py --phase prepare`: register and check geometry, point independence and splits.
- `train29.py --phase train`: 24 models, 25,200 updates; three seeds.
- `eval29.py --phase inference`: save GT-free expert/router predictions.
- `eval29.py --phase score`: all populations, subgroup metrics and privileged bounds.
- `diagnose29.py`: frozen-router risk versus actual errors, no further fitting.
- `support29.py`: true structural-line representability under the fixed Hough/pair masks.
- `audit29.py`: check training parity, arithmetic, splits and immutable prior artifacts.
- `report29.py`, `deliver29.py`: source-driven plots, Markdown, HTML and visible delivery.

New expert fine-tuning and router originals are disjoint. Initial frozen YOLO/V6
pretraining may overlap. No V12/V25 correction outputs enter the new point branch.
Line decoders use incident-role intersections and receive no point coordinates.
Visibility labels are artificial stress proxies; line labels are structural cuboid
proxies, not newly annotated visible physical boundaries or endpoints.

[Results](../../../data/pallet/results/pallet_adaptive_experts_v29/index.html)
· [Korean analysis](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_adaptive_experts_v29.md)

This explicitly continues the real297/9-session reused DEV research lineage;
canonical140/paper/final-holdout populations are not mixed into these comparisons.
