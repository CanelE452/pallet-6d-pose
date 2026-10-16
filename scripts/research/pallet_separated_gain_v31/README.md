# V31: separated point selection and incremental correction gain

Run with the `pallet-yolo26` environment:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_separated_gain_v31/run31.py
```

The resumable driver prepares fresh parent-training-disjoint original groups,
trains fixed models, evaluates all registered populations, audits geometry and
parent replay, creates the illustrated report, opens Chrome, and sends the
previously requested completion-only Discord message.

- 153 fresh training originals /64 lower-quantile diagnostic /64 fresh check.
- Historical YOLO validation exposure remains; these are not untouched whole-pipeline final data.
- Matched old153/new153 x legacy joint2000/separated selector1000+correction1000.
- General18directions from each anchor vs DHT18normals, same selector/parameter count.
- Fixed parents, synthetic GT only; no state-class routing or self-training filters.
- Separate numeric mean/lower-quantile and ranking outputs; gain magnitude cannot exceed movement.
- 21 checkpoints,27,000 actual optimizer updates,3seeds. Selector reuse accounts for the difference from30,000 arm-equivalent updates.
- Real297/9session explicitly reused DEV lineage; no canonical140 or sealed final model selection.
- Legacy feature-cache directory is restored during legacy reads; see `CACHE_SCOPE_FIX.json` for the assertion caught before seed2 and the preserved seed1 protocol hashes.

Completed V30 and upstream files stay unchanged. The implementation reuses
frozen parent code and copies V30 feature extraction locally into `backbone31.py`.
`prepare23` utilities are rebound only within the preparation process to the new
run's `fresh` directory.

[Korean results](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_separated_gain_v31.md)
· [HTML](../../../data/pallet/results/pallet_separated_gain_v31/index.html)
