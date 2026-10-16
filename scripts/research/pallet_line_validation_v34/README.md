# V34 replication and line-geometry validation

Requires immutable completed V32 data and V33 checkpoints. From the repo root:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_line_validation_v34/run34.py
```

Actual work: 12 runs with strict requested settings and equivalent fixed pooling,
then 12 more runs with equivalent dense fixed-grid bilinear sampling.
24 models / 48,000 actual optimizer updates. The initial unsupported adaptive
pooling attempt stopped before any optimizer update and is archived.

Each pair uses the same architecture, initialization, 2,000-step batch plan and
physical supervision in two fresh subprocesses. The replicas are not extra seeds.
Three seeds per arm; line_only still includes point parents and image features.
The dense sampler covers fixed bilinear zero-padding grids with align_corners=True
and does not provide gradients with respect to grid coordinates. Current parents
are frozen. Do not silently reuse it for trainable sampling coordinates.

Stages: `train34.py`, `replay34.py` (no optimizer steps), `train_dense34.py`,
`eval34.py --phase infer|score`, `probe34.py --phase infer|score`, `audit34.py`,
`report34.py`, `deliver34.py`. Pooling/sampler preflights check mathematical
equivalence and actual repeated gradients. Results include both failed and corrected
replication evidence, complete model/trace hashes, feature counterfactuals, GT and
source audits, MD/figures and an HTML report that is opened in a real Chrome window.

Counterfactual sources: original V33 line_only and corrected V34 line_only repeat0.
Shift the GT-free highest-attention line token by +/-4/8px along its normal while
keeping sampled appearance and all other tokens fixed. Separately swap geometry
or appearance from a different session/group. These are feature interventions,
not physically rerendered scenes or a correctness guarantee.

See PROTOCOL.json and the two implementation amendments. No criteria or manual GT
changed. Reused real297 DEV is separate from canonical140/paper/sealed final.
Never overwrite completed results to run a new protocol. The explicitly requested
completion Discord message is sent separately after final artifact verification.
