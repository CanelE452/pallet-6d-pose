# Hough pseudo-label quality v1

Fixed cached-output quality screen, not student self-training. Compare unchanged F4 accepted membership with original, weak uncertainty-weighted DHT correction, and unconditional DHT correction. No real GT enters correction; no label export to a training pool.

From repository root:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/hough_pseudolabel_quality_v1/run.py --phase freeze
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/hough_pseudolabel_quality_v1/run.py --phase evaluate
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/hough_pseudolabel_quality_v1/report.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/hough_pseudolabel_quality_v1/deliver.py
```

For independent reproduction pass a new `--out /absolute/output/path` to all commands; do not overwrite completed evidence. Sources and protocol are hash-bound. The report writes the topic note under `_docs/notes/supervision_and_distillation/`; browser delivery requires desktop access. Discord completion notification is separate, with prior user authorization.

Outcome: mean 8.5156→8.1114px on 35 accepted/40 overlapping DEV frames, 307 supervised points and three DHT seeds. P90 improves, but bad (>20px) label count remains 15 and the easy-point preservation criterion fails in seed3. All three seeds worsen on the noapril session. Conditional DEV label-quality signal only; no student training is justified by this screen. The frozen-feature architecture research closure remains in effect.
