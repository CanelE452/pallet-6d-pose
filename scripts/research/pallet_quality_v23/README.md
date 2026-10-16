# V23 intact synthetic observations and quality balance

Research outputs: `data/pallet/results/pallet_quality_v23/`. Frozen YOLO/V10/V11/V12/V18 parents; train only the V20 `CauseGate`. No real GT fitting, pseudo-labeling, inference filtering, checkpoint selection, or operating-model change.

Use `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`. CUDA is required for extraction, parents, training, inference, diagnosis, matching and integration. Scripts expect existing V10–V22 artifacts. Completed training/protocol files are intentionally protected against rerun/overwrite. Use a separate result directory and adjust the context for a fresh reproduction.

Run these commands in order with that interpreter:

```text
prepare23.py --phase register
prepare23.py --phase extract
prepare23.py --phase parents
prepare23.py --phase targets
train23.py
train_mass23.py --register
train_mass23.py
eval23.py --phase inference
eval23.py --phase score
diagnose23.py
matched23.py
audit23.py
network23.py
visual23.py
report23.py
delivery23.py
```

`report23.py` combines the generated numbers with the manually written, result-specific `INTERPRETATION_KO.md`; revise that interpretation for any fresh experiment. `delivery23.py` checks rendered table values and opens the local report in desktop Chrome.

Original experiment: 4 arms × 3 seeds × 2,000 updates. Training-data-only amendment, registered before any V23 evaluation: 2 mass-balanced arms × 3 seeds × 2,000 updates. Total 18 final models / 36,000 executed updates. All arms use identical initialization and original-image batch plans within seed. Requested training originals: 253 existing + 512 new; 1 new original has no detections in any condition. All 4,409 detected observations from 764 originals are exposed to training; none are removed by quality.

Data includes a separately reserved 128-image synthetic check set. It is new to retention training but from historical YOLO validation. Existing real and synthetic evaluations are reused development sets. Point-ID/error-bin composition matching uses real GT only after training for privileged diagnosis; those weights never reach a trained checkpoint or runtime decision.
