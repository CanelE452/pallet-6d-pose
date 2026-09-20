# PoseFix synthetic-GT replay control

One bounded 300-update preservation experiment. It starts from the original
synthetic-trained PRIOR1, not the damaged real-only checkpoint. Architecture,
real-image order, real input corruption, real exposure, optimizer, learning
rate and BN policy are unchanged. Each step adds eight synthetic examples to
the eight real examples. The source objective has weight1 and L2 is counted
exactly once. This is not a compute-matched comparison.

Only RGB and frozen R0 predicted boxes/keypoints enter inference. No depth,
CAD, PnP, rendering, dimensions or ground truth are inputs. Real supervision
remains the same9 images/38 verified manual corners; synthetic supervision
uses actual synthetic GT. No pseudo labels are created by this experiment.

`PROTOCOL.json` and `INPUT_LOCK.json` were frozen before model probes/training.
They bind original weights/code, source images/cache values, disjoint source
rows, the real RNG replay and all conditional self-training gates.

The main output is uncapped. The historical1%image-diagonal cap is reported
as a safety diagnostic, never selected because its evaluation result is better.

Stages from repository root:

```bash
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m unittest scripts.research.pallet_posefix_replay_v1.test_source scripts.research.pallet_posefix_replay_v1.test_replay -v
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_replay_v1.evaluate --baseline-audit
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -c 'from scripts.research.pallet_posefix_replay_v1 import core; core.prepare()'
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_replay_v1.train before
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_replay_v1.train train
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_replay_v1.evaluate
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_replay_v1.decision
MPLCONFIGDIR=/tmp/pallet-replay-mpl /home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_posefix_replay_v1.visualize
```

GPU commands require actual host GPU access. Do not interpret a sandbox's
device invisibility as a driver failure, change drivers, reboot or terminate
RustDesk. The runner rejects other GPU compute processes and checks temperature.

Training resumes only from the same protocol's fixed-order `resume.pt`
(saved every100steps). Completed artifacts are verified/reused, not overwritten.
Source image loading may prefetch using4threads; it performs no random draws.
Source noise has its own checkpointed RNG, independent of real order/noise.

`DECISION.json` decides conditional selective self-training. All gates must
pass. A failure ends this bounded run without generating new pseudo labels,
training a student, retuning thresholds, or replacing the existing final model.
A pass is only permission to lock and execute a separate bounded next stage,
not evidence of independent final-test improvement.

The unchanged PoseFix implementation retains the MIT/Apache notices under
`scripts/research/pallet_sensors_submission_v1/THIRD_PARTY_NOTICES.md`.
All earlier checkpoints, labels, reports and manuscript tables are preserved.
