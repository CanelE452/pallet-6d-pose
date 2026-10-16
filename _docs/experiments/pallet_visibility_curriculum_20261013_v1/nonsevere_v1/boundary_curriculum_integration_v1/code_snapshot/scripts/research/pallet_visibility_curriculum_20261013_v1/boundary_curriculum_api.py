"""Explicit research API for RGB visibility, outline measurements and curriculum.

The supplied initial cuboid determines conditional self-occlusion and outline
difficulty. Raw RGB and an independent image-mask prediction supply all fit
coordinates. Pose projections are returned separately as hidden completion.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .curriculum import CurriculumConfig, solve_curriculum
from .joint_api import _validate


class BoundaryCurriculum:
    """Optional fixed-family method; no detector or current default is changed."""

    def __init__(self, observation_provider, curriculum=None):
        self.observer = observation_provider
        profile = getattr(observation_provider, "boundary_profile", {})
        fusion = getattr(observation_provider, "fusion", {})
        self.maximum_measurement_shift_px = float(profile.get("maximum_corner_shift_px", 12.)) * float(fusion.get("maximum_alpha", .5))
        if not np.isfinite(self.maximum_measurement_shift_px) or self.maximum_measurement_shift_px <= 0:
            raise ValueError("Expected a finite positive observation-shift bound")
        self.config = CurriculumConfig(**dict(curriculum or {
            "visible_threshold": .2, "geometric_outline_priority": True}))
        self.config_binding = None

    @classmethod
    def from_config(cls, path, device=None):
        from .boundary_observation import BoundaryObservation
        path = Path(path).resolve()
        config = C.read(path)
        if config.get("schema") != "boundary_curriculum_v1":
            raise ValueError("Expected explicit boundary_curriculum_v1 configuration")
        observation_path = (path.parent / config["observation_config"]).resolve()
        expected = config.get("observation_config_sha256")
        if expected is not None and C.sha(observation_path) != expected:
            raise ValueError("Frozen observation configuration changed")
        observer = BoundaryObservation.from_config(observation_path, device)
        result = cls(observer, config.get("curriculum"))
        result.config_binding = dict(path=str(path), sha256=C.sha(path))
        return result

    def refine(self, image_bgr, points, K, initial_pose, registered_dimensions):
        """Keep original channel identity, dimensions and physical rotation basis."""
        started = time.perf_counter()
        image, raw, camera, registry, H = _validate(
            image_bgr, points, K, initial_pose, registered_dimensions)
        initial = deepcopy(initial_pose)
        # Protect the fit contract even with an injected or future observer.
        # It receives detached inputs, and mutations are rejected explicitly.
        query, observer_camera = raw.copy(), camera.copy()
        observer_initial = deepcopy(initial)
        observations = self.observer.assess(image, query, observer_camera, observer_initial)
        pose_fields = ("R_cf", "R_physical", "centroid", "cf_extents")
        if (not np.array_equal(query, raw, equal_nan=True)
                or not np.array_equal(observer_camera, camera, equal_nan=True)
                or any(field not in observer_initial or not np.array_equal(np.asarray(observer_initial[field], float), np.asarray(initial[field], float), equal_nan=True)
                       for field in pose_fields)):
            raise ValueError("Observation provider mutated raw queries, camera or initial pose")
        fused = np.asarray(observations["fused_points"], float)
        queried = np.asarray(observations["raw_points"], float)
        appearance = deepcopy(observations["visibility_appearance"])
        if (fused.shape != raw.shape or queried.shape != raw.shape
                or not np.array_equal(queried, raw, equal_nan=True)
                or not np.array_equal(np.asarray(observations["self_hidden"], bool), H)):
            raise ValueError("Observation provider changed raw inputs or fixed self-occlusion")
        valid = np.isfinite(raw[:8]).all(1) & ~np.all(raw[:8] == -1, axis=1)
        height, width = image.shape[:2]
        inside = valid & (raw[:8, 0] >= 0) & (raw[:8, 0] < width) & (raw[:8, 1] >= 0) & (raw[:8, 1] < height)
        excluded = H | ~inside
        if (not np.array_equal(fused[:8][excluded], raw[:8][excluded], equal_nan=True)
                or not np.array_equal(fused[8:], raw[8:], equal_nan=True)):
            raise ValueError("Observation provider moved hidden/missing/outside points or centre")
        if not np.isfinite(fused[:8][inside]).all():
            raise ValueError("Observation provider produced invalid image measurements")
        supported = fused[:8][inside]
        if (np.any(supported[:, 0] < 0) or np.any(supported[:, 0] >= width)
                or np.any(supported[:, 1] < 0) or np.any(supported[:, 1] >= height)
                or np.any(np.linalg.norm(fused[:8][inside] - raw[:8][inside], axis=1) > self.maximum_measurement_shift_px + 1e-8)):
            raise ValueError("Observation provider exceeded image bounds or configured shift bound")
        xyz = C.cuboid(*initial["cf_extents"])
        R0, t0 = np.asarray(initial["R_cf"], float), np.asarray(initial["centroid"], float)
        physical_basis = R0.T @ np.asarray(initial["R_physical"], float)
        tick = time.perf_counter()
        solution = solve_curriculum(fused[:8], xyz, camera, R0, t0, H, appearance, self.config)
        actual = dict(initial, R_cf=solution["R"], centroid=solution["t"],
                      R_physical=(np.asarray(solution["R"]) @ physical_basis).tolist())
        if not solution["accepted"]:
            actual = deepcopy(initial)
        projected = C.project(xyz, actual["R_cf"], actual["centroid"], camera)
        completed = raw.copy()
        if solution["accepted"]:
            completed[:8] = fused[:8]
            completed[np.flatnonzero(H)] = projected[H]
        if not np.array_equal(completed[8:], raw[8:], equal_nan=True):
            raise AssertionError("Hidden completion changed centre")
        if any(H[k] for stage in solution["stages"] for k in stage["used_ids"]):
            raise AssertionError("Self-hidden projection entered pose fit")
        return C.finite(dict(schema="boundary_curriculum_inference_v1", experimental=True,
            actual_pose=actual, registered_dimensions=registry.tolist(), self_hidden=H.tolist(),
            boundary_visibility=observations["boundary_visibility"],
            visibility_assessment=appearance, observation_result=observations, solution=solution,
            observed_points=raw.tolist(), corrected_measurements=fused.tolist(),
            projected_points=projected.tolist(), completed_points=completed.tolist(),
            provenance=dict(ground_truth_inputs=False, native_corner_ids_preserved=True,
                initial_model_and_physical_basis_preserved=True, original_points_preserved=True,
                image_measurement_source="raw native predictions with bounded two-image-line outline fusion",
                self_hidden_source="fixed original initial cuboid pose",
                projections_used_for_fit=False, hidden_completion_used_for_fit=False,
                geometric_outline_used_for="difficulty only; independent RGB determines visibility and measurements",
                visibility_and_coordinate_precision_separated=True,
                line_covariance_is_calibrated_semantic_probability=False,
                default_model_promoted=False, centre_preserved=True,
                configuration=self.config_binding),
            timing=dict(pose_solve_seconds=time.perf_counter() - tick,
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
        raise FileExistsError("Preserve previous inference: " + str(args.output))
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(str(args.image))
    inputs = C.read(args.input)
    engine = BoundaryCurriculum.from_config(args.config, args.device)
    output = engine.refine(image, inputs["points"], inputs["K"], inputs["initial_pose"],
                           inputs["registered_dimensions"])
    C.write(args.output, output)
    print("BOUNDARY_CURRICULUM_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
