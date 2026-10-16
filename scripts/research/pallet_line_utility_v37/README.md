# V37 frozen line-to-point utility diagnostic

CPU NumPy ridge fits using existing V35 geometry and V36 cached candidates. No neural optimizer steps or new real evaluation. Primary96fits, separate objective48fits, total144. Completed prior studies are read-only inputs.

Use `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python` from repository root. Run in order:

```bash
python scripts/research/pallet_line_utility_v37/analyze37.py
python scripts/research/pallet_line_utility_v37/audit37.py
python scripts/research/pallet_line_utility_v37/residual37.py
python scripts/research/pallet_line_utility_v37/objectives37.py
python scripts/research/pallet_line_utility_v37/audit_extra37.py
python scripts/research/pallet_line_utility_v37/report37.py
python scripts/research/pallet_line_utility_v37/deliver37.py
```

`run37.py --phase all` skips completed phase markers. `--phase audit` and `--phase audit_extra` rerun validations. Desktop Chrome delivery needs the user desktop environment. Notifications use the already authorized local Discord hook only after COMPLETION.

Inputs: V35 POPULATION/geometry/offset calibration, V36 training_predictions/render_check predictions, cached frozen query/visual features. Train16groups/check8groups are disjoint, same2CAD and reused check; no new real or sealed-final claim. Missing detections are excluded from conditional point metrics and counts explicitly retained.

Protocols: PROTOCOL (primary), RESIDUAL_PROTOCOL (post-primary arithmetic), OBJECTIVE_PROTOCOL (post-primary48fit objective controls). No check-dependent threshold changes. Bound .15boxdiag proposal oracle is not a universal capacity bound. Point-GT oracle can exploit wrong physical lines; not deployed performance.

Audits: stored predictions/metrics,144selector replays,12fit repetitions with corrupted check data, sign/tangent/residual identities, source grouping, V36 line metric reproduction and earlier file hashes. Twelve repetitions are the4primary+8supplementary controls, not all144fits repeated twice.
