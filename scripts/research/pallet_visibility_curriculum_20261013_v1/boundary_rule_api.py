"""Optional image-supported 5/4/3/2-corner correction with explicit decisions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .joint_api import _validate


class BoundaryRuleCorrector:
    def __init__(self, observation_provider, rules=None):
        self.observer = observation_provider
        self.rules = deepcopy(rules or {})
        self.config_binding = None

    @classmethod
    def from_config(cls, path, device=None):
        from .boundary_observation import BoundaryObservation
        path = Path(path).resolve()
        config = C.read(path)
        if config.get("schema") != "boundary_rule_corrector_v1":
            raise ValueError("Expected explicit boundary_rule_corrector_v1 configuration")
        observation_path = (path.parent / config["observation_config"]).resolve()
        expected = config.get("observation_config_sha256")
        if expected is not None and C.sha(observation_path) != expected:
            raise ValueError("Frozen observation configuration changed")
        result = cls(BoundaryObservation.from_config(observation_path, device), config.get("rules"))
        result.config_binding = dict(path=str(path), sha256=C.sha(path))
        return result

    def refine(self, image_bgr, points, K, initial_pose, registered_dimensions,
               *, occluding_evidence=None):
        """Fit measured strong corners; complete others as separate projections.

        A supplied independent occluder record can exclude an otherwise strong
        query. Palette negatives alone never establish another object's identity.
        Three-point corrections retain the initial rotation exactly. No point
        receives a new observed coordinate just because it is missing or hidden.
        """
        from .boundary_rules import solve_boundary_rules
        started = time.perf_counter()
        image, raw, camera, registry, H = _validate(
            image_bgr, points, K, initial_pose, registered_dimensions)
        initial = deepcopy(initial_pose)
        query, observer_camera = raw.copy(), camera.copy()
        observer_initial = deepcopy(initial)
        observer_image = image.view()
        observer_image.flags.writeable = False
        observations = self.observer.assess(observer_image, query, observer_camera, observer_initial)
        if (not np.array_equal(query, raw, equal_nan=True)
                or not np.array_equal(observer_camera, camera, equal_nan=True)
                or any(field not in observer_initial or not np.array_equal(
                       np.asarray(observer_initial[field], float), np.asarray(initial[field], float), equal_nan=True)
                       for field in ("R_cf", "R_physical", "centroid", "cf_extents"))):
            raise ValueError("Observation provider mutated queries, camera or initial pose")
        if (not np.array_equal(np.asarray(observations["raw_points"], float), raw, equal_nan=True)
                or not np.array_equal(np.asarray(observations["self_hidden"], bool), H)):
            raise ValueError("Observation provider changed raw native coordinates or initial H")
        fused = np.asarray(observations["fused_points"], float)
        valid = np.isfinite(raw[:8]).all(1) & ~np.all(raw[:8] == -1, axis=1)
        height, width = image.shape[:2]
        inside = valid & (raw[:8, 0] >= 0) & (raw[:8, 0] < width) & (raw[:8, 1] >= 0) & (raw[:8, 1] < height)
        excluded = H | ~inside
        if (fused.shape != raw.shape
                or not np.array_equal(fused[:8][excluded], raw[:8][excluded], equal_nan=True)
                or not np.array_equal(fused[8:], raw[8:], equal_nan=True)):
            raise ValueError("Observation provider moved excluded points or centre")
        supported = fused[:8][inside]
        if (not np.isfinite(supported).all()
                or np.any(supported[:, 0] < 0) or np.any(supported[:, 0] >= width)
                or np.any(supported[:, 1] < 0) or np.any(supported[:, 1] >= height)
                or np.any(np.linalg.norm(supported - raw[:8][inside], axis=1) > 6. + 1e-8)):
            raise ValueError("Observation provider exceeded canvas or 6px fusion bound")
        observations = deepcopy(observations)
        observations["image_hw"] = list(image.shape[:2])
        if occluding_evidence is not None:
            observations["occluding_evidence"] = deepcopy(occluding_evidence)
        cfg = deepcopy(self.rules)
        cfg["image_hw"] = list(image.shape[:2])
        xyz = C.cuboid(*initial["cf_extents"])
        R0, t0 = np.asarray(initial["R_cf"], float), np.asarray(initial["centroid"], float)
        tick = time.perf_counter()
        solution = solve_boundary_rules(observations, xyz, camera, R0, t0, cfg)
        physical_basis = R0.T @ np.asarray(initial["R_physical"], float)
        actual = deepcopy(initial)
        if solution["accepted"]:
            actual.update(R_cf=solution["R"], centroid=solution["t"],
                          R_physical=(np.asarray(solution["R"]) @ physical_basis).tolist())
        if solution["accepted"] and solution["estimated_dof"] == 3:
            # Keep both rotational matrices exact, including rounding in the
            # caller's physical basis; translation is the only changed field.
            actual["R_cf"] = deepcopy(initial["R_cf"])
            actual["R_physical"] = deepcopy(initial["R_physical"])
        projected = C.project(xyz, actual["R_cf"], actual["centroid"], camera)
        measured = np.asarray(observations["raw_points"] if solution["coordinate_source"] == "RAW"
                              else observations["fused_points"], float)
        completed = raw.copy()
        completion_ids = []
        if solution["accepted"]:
            used = set(solution["used_ids"])
            completion_ids = sorted(set(range(8)) - used)
            completed[:8] = measured[:8]
            completed[completion_ids] = projected[completion_ids]
        if any(H[k] or not inside[k] for k in solution["used_ids"]):
            raise AssertionError("Excluded corner entered rule-based fitting")
        if not np.array_equal(completed[8:], raw[8:], equal_nan=True):
            raise AssertionError("Rule completion changed centre")
        return C.finite(dict(schema="boundary_rule_corrector_inference_v1", experimental=True,
            actual_pose=actual, registered_dimensions=registry.tolist(), self_hidden=H.tolist(),
            keypoint_status=solution["evidence"]["records"], decision=dict(
                branch=solution["branch"], estimated_dof=solution["estimated_dof"],
                geometric_boundary_count=solution["geometric_boundary_count"],
                RGB_material_boundary_count=solution["RGB_material_boundary_count"],
                image_supported_boundary_count=solution["reliable_boundary_count"],
                image_supported_boundary_ids=solution["reliable_boundary_ids"],
                used_ids=solution["used_ids"], accepted=solution["accepted"],
                fallback_reason=solution["fallback_reason"]),
            solution=solution, observation_result=observations,
            observed_points=raw.tolist(), corrected_measurements=measured.tolist(),
            projected_points=projected.tolist(), completed_points=completed.tolist(),
            projected_completion_ids=completion_ids,
            provenance=dict(ground_truth_inputs=False, native_order_and_center_preserved=True,
                initial_model_and_physical_basis_preserved=True, other_object_requires_independent_evidence=True,
                missing_or_excluded_points_are_output_completion_only=True,
                projections_used_as_measurements=False, projected_completion_used_for_fit=False,
                low_RGB_support_is_unknown_or_occlusion=True, count_is_pose_rank_proof=False,
                three_point_rotation_is_exactly_initial=True, default_model_promoted=False,
                config_binding=self.config_binding),
            timing=dict(pose_rule_seconds=time.perf_counter() - tick,
                        total_postprocessing_seconds=time.perf_counter() - started)))


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
        raise FileExistsError("Preserve previous inference")
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(str(args.image))
    inputs = C.read(args.input)
    result = BoundaryRuleCorrector.from_config(args.config, args.device).refine(
        image, inputs["points"], inputs["K"], inputs["initial_pose"], inputs["registered_dimensions"],
        occluding_evidence=inputs.get("occluding_evidence"))
    C.write(args.output, result)
    print("BOUNDARY_RULE_CORRECTION_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
