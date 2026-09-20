# RGB PoseFix large-error trainability screen

This is a bounded diagnostic of the existing official-network-validated
PoseFix-derived pallet9 implementation, not its first implementation and not a
reproduction of the full human-pose benchmark. Original artifacts are immutable.

Inference inputs: RGB, frozen R0 predicted box and keypoints only. No depth,
CAD, rendering, PnP, dimensions, GT pose or symmetry code.

Read `PROTOCOL.json` for the locked 300-update manual9 supervision/noise/BN
contract and `RESULTS_KO.md` for the result. A successful same-image training
probe does not establish improvement on new images. Capturing hard images and
labeling them is a separate decision; this run creates no new annotation.

Commands from repository root (GPU commands need host GPU access):

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m unittest scripts.research.pallet_posefix_large_error_v1.test_core -v
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_large_error_v1.evaluate --baseline-audit
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -c 'from scripts.research.pallet_posefix_large_error_v1 import core; core.prepare()'
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -c 'import torch,cv2; torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; from scripts.research.pallet_posefix_large_error_v1.evaluate import evaluate; evaluate("existing")'
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_large_error_v1.train
# Only if FIT.json trainability_pass is true:
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -c 'import torch,cv2; torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; from scripts.research.pallet_posefix_large_error_v1.evaluate import evaluate; evaluate("finetuned")'
MPLCONFIGDIR=/tmp/pallet-posefix-plot /home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_posefix_large_error_v1.visualize --model existing
MPLCONFIGDIR=/tmp/pallet-posefix-plot /home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_posefix_large_error_v1.visualize --model finetuned
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_posefix_large_error_v1.report
```

Completed training intentionally refuses a second run. There is no automatic
resume after interruption; inspect STATUS/checkpoint/receipts first and do not
overwrite or silently retrain. Evaluation reuses completed verified artifacts.

The main model is imported unchanged from `pallet_sensors_submission_v1` and
remains covered by its `THIRD_PARTY_NOTICES.md`, including PoseFix MIT and
TensorFlow Authors' Apache-2.0 notices. Paper: https://arxiv.org/abs/1812.03595.
