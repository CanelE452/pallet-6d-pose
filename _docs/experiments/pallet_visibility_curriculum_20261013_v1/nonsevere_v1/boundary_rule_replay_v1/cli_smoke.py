"""One real CAL image CLI check selected by count, without reading targets."""
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent


def main():
    for name in ("EXAMPLE_INPUT.json", "EXAMPLE_OUTPUT.json", "CLI_SMOKE.json"):
        if (HERE / name).exists():
            raise FileExistsError(name)
    seal = C.read(COHORT / "boundary_count_rules_v1/PREDICTION_SEAL.json")
    predictions = COHORT / "boundary_count_rules_v1/PREDICTIONS.jsonl.gz"
    assert seal["predictions_sha256"] == C.sha(predictions)
    chosen = next(row for row in C.rows(predictions)
                  if row["arm"] == "FUSED_SIGMA2_BOUNDARY_COUNT_RULE"
                  and row["solution"]["accepted"] and row["solution"]["estimated_dof"] == 3)
    frame = next(f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"]
                 if f["id"] == chosen["id"])
    assert frame["partition"] == "CAL"
    image_path = ROOT / frame["image"]
    assert C.sha(image_path) == frame["image_sha256"]
    inputs = dict(points=frame["points"]["N3_SUBPIX"], K=frame["K"],
                  initial_pose=frame["initial_poses"]["N3_SUBPIX"], registered_dimensions=frame["xyz"])
    C.write(HERE / "EXAMPLE_INPUT.json", inputs)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.argv = ["boundary_rule_api", "--image", str(image_path), "--input", str(HERE / "EXAMPLE_INPUT.json"),
                "--config", str(COHORT / "BOUNDARY_RULE_CONFIG.json"), "--output", str(HERE / "EXAMPLE_OUTPUT.json"),
                "--device", "cuda"]
    with torch.inference_mode():
        runpy.run_module("scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_rule_api", run_name="__main__")
    output = C.read(HERE / "EXAMPLE_OUTPUT.json")
    assert output["decision"]["accepted"] and output["decision"]["estimated_dof"] == 3
    assert output["actual_pose"]["R_cf"] == inputs["initial_pose"]["R_cf"]
    assert output["actual_pose"]["R_physical"] == inputs["initial_pose"]["R_physical"]
    assert len(output["projected_completion_ids"]) == 5
    assert output["decision"]["used_ids"] == chosen["solution"]["used_ids"]
    for field in ("R_cf", "R_physical", "centroid"):
        np.testing.assert_allclose(output["actual_pose"][field], chosen["actual_pose"][field], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(output["completed_points"][8:], inputs["points"][8:])
    C.write(HERE / "CLI_SMOKE.json", dict(passed=True, frame_id=frame["id"], partition="CAL",
        selection="First sealed CAL FUSED count-rule prediction with accepted three-DOF branch; no outcome selection",
        image=str(image_path), image_sha256=C.sha(image_path), fresh_RGB_forward=True,
        invocation=sys.argv, estimated_dof=3, exact_initial_both_rotations=True,
        completion_slots=5, completed_slots_are_output_only=True, frozen_CAL_pose_parity_atol=1e-10,
        GT_inputs=False, severe_frames=0, FINAL_reads=0, new_training=0,
        input_sha256=C.sha(HERE / "EXAMPLE_INPUT.json"), output_sha256=C.sha(HERE / "EXAMPLE_OUTPUT.json")))
    print("COUNT_RULE_REAL_RGB_CLI_PASS", frame["id"], flush=True)


if __name__ == "__main__":
    main()
