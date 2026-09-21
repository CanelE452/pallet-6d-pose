# PoseFix replay diagnosis v1

No training. Canonical synthetic-only three-seed `PRIOR{1,2,3}/last.pt`, not later real+synthetic Replay last300. Read `FINAL_DIAGNOSIS.md` for conclusions and `REPLAY_SUMMARY.md` for every pass.

```bash
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/test_diagnosis.py
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/run.py prepare
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/run.py infer
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/run.py score
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/run.py report
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/audit.py
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/conclude.py
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_posefix_replay_diagnosis_v1/run.py verify
```

`infer` needs host CUDA permission; sandbox CUDA unavailability is not a broken driver. Prefer the sequential commands above: the final `INFERENCE_COMPLETE.json` receipt is required for aggregation. `score` can consume atomic split outputs while inference runs, but if it reaches aggregation before that receipt is published it stops safely; rerunning after inference completes reuses finished scores. Six CPU workers, unchanged solver. No external package install, restart, power/clock or driver changes.

Large artifacts remain ignored under `data/pallet/results/pallet_posefix_replay_diagnosis_v1/`. `INPUTS.json` contains predicted input only; `TARGETS.json` is separate, never read by neural inference except content hashing. Original coordinates are used in saved predictions; the existing synthetic/square padding is removed exactly once.

## Per-corner records

- `predictions/seed{seed}_{split}.npz`: `points[frame,chain,pass0..3,9,xy]`, `raw` and `capped[frame,chain,pass1..3,9,xy]`, exact IDs/checkpoint/protocol binding. Input to pass p is `points[:,:,p-1]`. RAW feeds RAW; CAPPED feeds CAPPED.
- `scores/seed{seed}_{split}.npz`: native-channel GT errors against the fixed PASS0 whole-object branch, current-branch canonical GT errors, support mask, raw/capped/actual displacement, cap-hit, both consecutive cosines, current branch, overlapping frame/corner classifications. Before/after/delta errors are consecutive slices/differences, not recomputed correspondence choices.
- `metrics/*.json`: each frame/pass current evaluator outputs, including full penalties, matching and branch.
- `pose/*.json`: all prediction-only PnP outputs and GT-side metrics, identical K/physical dimensions per frame.

Movement statistics exclude missing/mismatched or unsupported corners; main PCK retains their penalties. Reported n is per seed, not three times as many independent corners. Dimensions are labels for stratification and PnP, never a new neural input. C1/C2/C4 membership is from the locked task contract, not inferred from equal dimensions.

No winner selection, threshold tuning, auto-promotion or training occurs after reporting. Only a strong preregistered main-DEV phase ambiguity finding can trigger a **proposal/scaffold**, never an automatic training run.
