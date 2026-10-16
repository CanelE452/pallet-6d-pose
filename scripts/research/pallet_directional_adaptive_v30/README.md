# V30 state-free per-point directional adaptation

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_directional_adaptive_v30/run30.py
```

The driver resumes fixed stages through training, GT-free inference, evaluation,
gain diagnosis, geometry/parent replay audit, report rendering and visible Chrome
delivery. It sends the user's previously requested completion-only Discord message.

- Frozen V12 and strong V25 point outputs are exact selectable keep paths.
- Frozen V29 guided/independent DHT provide candidate normals.
- No condition or visibility state is passed to the learned head or loss.
- Signed partial movement along a normal preserves the selected point anchor's tangent.
- Four matched heads x3seeds x2000updates =12models /24,000updates.
- Training originals are separate from V29 DHT fine-tuning. The V25 point parent
  already trained on these originals, so its residuals remain in-sample.
- All real comparisons continue the explicitly reused real297/9-session research
  population, not canonical140/paper/new sealed evaluations.
- Structural lines are not newly annotated physical visible boundaries/endpoints.

Individual phases: `train30.py --phase prepare|train`,
`eval30.py --phase inference|score`, `diagnose30.py`, `audit30.py`,
`report30.py`, `deliver30.py`.

[Korean report](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_directional_adaptive_v30.md)
· [HTML with actual images](../../../data/pallet/results/pallet_directional_adaptive_v30/index.html)
