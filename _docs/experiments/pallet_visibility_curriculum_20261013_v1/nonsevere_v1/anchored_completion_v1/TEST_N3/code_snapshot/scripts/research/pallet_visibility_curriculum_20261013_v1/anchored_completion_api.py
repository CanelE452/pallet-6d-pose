"""Protected RGB observations and separate rigid completion of all eight slots."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .joint_api import _validate


class AnchoredCompletion:
    """Optional state-aware correction; no hidden projection enters fitting."""

    def __init__(self, observation_provider, rules=None, *, interior_provider=None):
        self.observer = observation_provider
        self.rules = deepcopy(rules or {})
        self.interior_provider = interior_provider
        self.config_binding = None

    @classmethod
    def from_config(cls, path, device=None):
        from .boundary_observation import BoundaryObservation
        path = Path(path).resolve()
        config = C.read(path)
        if config.get("schema") != "anchored_completion_config_v1":
            raise ValueError("Explicit anchored_completion_config_v1 required")
        observation_path = (path.parent / config["observation_config"]).resolve()
        if C.sha(observation_path) != config["observation_config_sha256"]:
            raise ValueError("Frozen observation configuration changed")
        interior_provider = None
        if config.get("interior_image_refinement", False):
            from .interior_observation import infer_interior_observations
            settings = deepcopy(config.get("interior_observation", {}))
            interior_provider = lambda image, q, K, pose, obs: infer_interior_observations(
                image, q, K, pose, obs, settings)
        result = cls(BoundaryObservation.from_config(observation_path, device),
                     config.get("rules"), interior_provider=interior_provider)
        result.config_binding = dict(path=str(path), sha256=C.sha(path))
        return result

    def refine(self, image_bgr, points, K, initial_pose, registered_dimensions,
               *, occluding_evidence=None):
        from .anchored_completion import solve_anchored_completion
        started = time.perf_counter()
        image, raw, camera, registry, H = _validate(
            image_bgr, points, K, initial_pose, registered_dimensions)
        initial = deepcopy(initial_pose)
        query, provider_camera = raw.copy(), camera.copy()
        provider_pose = deepcopy(initial)
        readonly_image = image.view()
        readonly_image.flags.writeable = False
        observation = self.observer.assess(readonly_image, query, provider_camera, provider_pose)
        if (not np.array_equal(query, raw, equal_nan=True)
                or not np.array_equal(provider_camera, camera, equal_nan=True)
                or any(field not in provider_pose or not np.array_equal(
                    np.asarray(provider_pose[field], float), np.asarray(initial[field], float), equal_nan=True)
                    for field in ("R_cf", "R_physical", "centroid", "cf_extents"))):
            raise ValueError("Observation provider mutated query, camera or pose")
        if (not np.array_equal(np.asarray(observation["raw_points"], float), raw, equal_nan=True)
                or not np.array_equal(np.asarray(observation["self_hidden"], bool), H)):
            raise ValueError("Observation provider changed original coordinates or H")
        observation = deepcopy(observation)
        observation["image_hw"] = list(image.shape[:2])
        if occluding_evidence is not None:
            observation["occluding_evidence"] = deepcopy(occluding_evidence)
        if self.interior_provider is not None:
            observation["interior_observations"] = self.interior_provider(
                readonly_image, raw.copy(), camera.copy(), deepcopy(initial), deepcopy(observation))
        rules = deepcopy(self.rules)
        rules["image_hw"] = list(image.shape[:2])
        if observation.get("interior_observations") is not None:
            from .interior_observation import image_pixel_sha256
            actual_pixel_hash = image_pixel_sha256(readonly_image)
            observation["image_pixel_sha256"] = actual_pixel_hash
            rules["image_pixel_sha256"] = actual_pixel_hash
        xyz = C.cuboid(*initial["cf_extents"])
        R0 = np.asarray(initial["R_cf"], float)
        solution = solve_anchored_completion(observation, xyz, camera,
            R0, np.asarray(initial["centroid"], float), rules)
        actual = deepcopy(initial)
        if solution["accepted"]:
            physical_basis = R0.T @ np.asarray(initial["R_physical"], float)
            actual.update(R_cf=solution["R"], centroid=solution["t"],
                          R_physical=(np.asarray(solution["R"]) @ physical_basis).tolist())
            if np.array_equal(np.asarray(solution["R"], float), R0):
                actual["R_cf"] = deepcopy(initial["R_cf"])
                actual["R_physical"] = deepcopy(initial["R_physical"])
        projected = C.project(xyz, actual["R_cf"], actual["centroid"], camera)
        rigid_points = raw.copy()
        rigid_points[:8] = projected
        anchors = list(solution["anchor_ids"])
        used = list(solution["used_ids"])
        # The model completion is one rigid pose, including measured slots.
        # Independent measurement coordinates remain in a separate array.
        if any(H[k] for k in used):
            raise AssertionError("Self-hidden model projection entered fitting")
        if not np.array_equal(rigid_points[8:], raw[8:], equal_nan=True):
            raise AssertionError("Rigid completion changed native centre")
        return C.finite(dict(schema="anchored_completion_inference_v1", experimental=True,
            actual_pose=actual, registered_dimensions=registry.tolist(),
            self_hidden=H.tolist(), solution=solution, observation_result=observation,
            original_predictions=raw.tolist(), measured_points=solution["measured_points"],
            keypoint_status=solution["keypoint_status"],
            observed_measurements_by_id={str(k): solution["measured_points"][k] for k in used},
            anchor_ids=anchors, used_ids=used, projected_points=projected.tolist(),
            completed_points=rigid_points.tolist(), rigid_model_corners=projected.tolist(),
            measurement_residuals={str(k): (projected[k]-np.asarray(solution["measured_points"], float)[k]).tolist()
                                   for k in used},
            provenance=dict(ground_truth_inputs=False, original_native_order_and_centre_preserved=True,
                measurements_separate_from_rigid_model=True, hidden_points_are_output_completion_only=True,
                projected_coordinates_used_as_measurements=False, rigid_completion_is_observed=False,
                anchor_observations_held_immutable=True, anchor_projection_exact_lock=False,
                observation_based_guards_are_ground_truth_accuracy_guarantees=False,
                initial_geometric_flags_are_conditional=True, config_binding=self.config_binding,
                default_model_promoted=False),
            timing=dict(total_postprocessing_seconds=time.perf_counter()-started)))


def main():
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous inference output")
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    data = C.read(args.input)
    result = AnchoredCompletion.from_config(args.config, args.device).refine(
        image, data["points"], data["K"], data["initial_pose"], data["registered_dimensions"],
        occluding_evidence=data.get("occluding_evidence"))
    C.write(args.output, result)
    print("ANCHORED_COMPLETION_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
