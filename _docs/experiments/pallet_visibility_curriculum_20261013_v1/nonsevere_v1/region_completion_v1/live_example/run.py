"""One predeclared CAL example with fresh RGB inference and no annotations."""
import importlib.util
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.protected_completion_api import ProtectedCompletion
HERE = Path(__file__).resolve().parent
COHORT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("sealed_prediction_guard", COHORT / "anchored_completion_v1/experiment.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

if (HERE / "PREDICTION.json").exists():
    raise FileExistsError("Preserve live inference")
cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
start = time.perf_counter()
with guard.no_labels():
    protocol = C.read(HERE / "PROTOCOL.json")
    assert C.sha(ROOT / protocol["image"]) == protocol["image_sha256"]
    assert C.sha(HERE / "INPUT.json") == protocol["input_sha256"]
    config = COHORT / "PROTECTED_COMPLETION_CONFIG.json"
    assert C.sha(config) == protocol["config_sha256"]
    image, data = cv2.imread(str(ROOT / protocol["image"])), C.read(HERE / "INPUT.json")
    engine = ProtectedCompletion.from_config(config, "cuda")
    result = engine.refine(image, data["points"], data["K"], data["initial_pose"], data["registered_dimensions"])
    C.write(HERE / "PREDICTION.json", result)
    C.write(HERE / "PREDICTION_SEAL.json", dict(before_parity_diagnostics=True,
        prediction_sha256=C.sha(HERE / "PREDICTION.json"), protocol_sha256=C.sha(HERE / "PROTOCOL.json"),
        current_RGB_observer_calls=1, current_RGB_interior_calls=1, current_RGB_joint_calls=1,
        cached_detector_and_upstream_initial_pose=True, new_SubPix_calls=0,
        new_neural_training=0, severe_frames=0, GT_reads=0, seconds=time.perf_counter()-start))
    reference = next(r for r in C.rows(COHORT / "region_completion_v1/CAL_N3/PREDICTIONS.jsonl.gz")
                     if r["id"] == protocol["frame_id"] and r["method"] == "FINITE_REGION")
    delta = {field: float(np.max(np.abs(np.asarray(result["actual_pose"][field])-np.asarray(reference["actual_pose"][field]))))
             for field in ("R_cf", "R_physical", "centroid")}
    readout_delta = float(np.max(np.abs(np.asarray(result["reported_points"])-np.asarray(reference["keypoint_readout"]["reported_points"]))))
    C.write(HERE / "PARITY.json", dict(pose_max_abs_delta=delta, reported_points_max_abs_delta_px=readout_delta,
        selected_pose_source=result["solution"]["selected_pose_source"],
        reference_pose_source=reference["solution"]["selected_pose_source"],
        parity_pass=max(delta.values()) < 1e-6 and readout_delta < 1e-5,
        predictions_compared_after_live_seal=True, GT_reads=0))
    assert max(delta.values()) < 1e-6 and readout_delta < 1e-5
print("LIVE_STATEWISE_PARITY_PASS", delta, readout_delta, flush=True)
