"""Image-only visibility curriculum after an upstream keypoint/pose estimate.

The caller supplies native cuboid corner order, raw-image pixel coordinates,
raw-image K, and the initial camera-facing pose in metres. Evaluation caches,
annotations and camera-facing/physical correspondence searches are not inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .appearance import infer_appearance
from .calibration import predict
from .curriculum import solve_curriculum


class VisibilityCurriculum:
    """Reusable postprocessor; the frozen model parameters load once."""

    def __init__(self, visibility_model=None, segmentation_weights=None,
                 curriculum_config=None, appearance_config=None, device=None):
        self.visibility_model = (C.read(visibility_model) if isinstance(visibility_model, (str, Path))
                                 else deepcopy(visibility_model))
        self.segmentation_weights = segmentation_weights
        self.curriculum_config = dict(curriculum_config or {})
        if self.visibility_model is not None:
            self.curriculum_config.setdefault("visible_threshold", self.visibility_model["visible_threshold"])
        self.appearance_config = dict(appearance_config or {})
        self.device = device

    @classmethod
    def from_config(cls, path, device=None):
        """Resolve weight paths relative to a portable configuration file."""
        path = Path(path).resolve()
        config = C.read(path)
        paths = {}
        for key in ("visibility_model", "segmentation_weights"):
            value = config.get(key)
            paths[key] = None if value is None else str((path.parent / value).resolve())
        return cls(**paths, curriculum_config=config.get("curriculum"),
                   appearance_config=config.get("appearance"), device=device)

    def refine(self, image_bgr, points, K, initial_pose):
        """Refine a single already-selected pallet without mutating inputs.

        initial_pose: available, R_cf, centroid, cf_extents (width,height,depth),
        and optionally R_physical/selected_hypothesis. Corner order is
        common.cuboid(); the optional ninth centre point is preserved.
        """
        started = time.perf_counter()
        image = np.asarray(image_bgr)
        q = np.asarray(points, dtype=float)
        camera = np.asarray(K, dtype=float)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty uint8 BGR image [H,W,3]")
        if q.shape not in ((8, 2), (9, 2)):
            raise ValueError("Expected native cuboid points [8,2] or [9,2]")
        if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
            raise ValueError("Expected finite raw-image camera matrix with positive focal lengths")
        if not initial_pose.get("available", False):
            raise ValueError("An available upstream initial pose is required")
        R = np.asarray(initial_pose["R_cf"], dtype=float)
        t = np.asarray(initial_pose["centroid"], dtype=float)
        dimensions = np.asarray(initial_pose["cf_extents"], dtype=float)
        if (R.shape != (3, 3) or t.shape != (3,) or dimensions.shape != (3,)
                or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.isfinite(dimensions).all() or (dimensions <= 0).any()
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(R), 1., atol=1e-5)):
            raise ValueError("Initial pose must use a proper rotation and positive metre dimensions")
        xyz = C.cuboid(*dimensions)
        if np.any((xyz @ R.T + t)[:, 2] <= 0):
            raise ValueError("Initial cuboid must lie in front of the camera")
        hidden = C.self_hidden(xyz, R, t)
        tick = time.perf_counter()
        if self.segmentation_weights is None:
            appearance = infer_appearance(image, q[:8], hidden, self.appearance_config)
        else:
            from .segmentation import segment
            from .segmented_appearance import infer_segmented_appearance
            probability = segment(image, self.segmentation_weights, self.device)
            appearance = infer_segmented_appearance(image, q[:8], hidden, probability, self.appearance_config)
        if self.visibility_model is not None:
            appearance = predict(appearance, self.visibility_model)
        evidence_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        solution = solve_curriculum(q[:8], xyz, camera, R, t, hidden, appearance, self.curriculum_config)
        solve_seconds = time.perf_counter() - tick
        result_R = np.asarray(solution["R"])
        projected = C.project(xyz, result_R, solution["t"], camera)
        completed = q.copy()
        if solution["accepted"]:
            completed[np.flatnonzero(hidden)] = projected[hidden]
        pose = dict(available=True, R_cf=solution["R"], centroid=solution["t"],
                    cf_extents=dimensions.tolist(), selected_hypothesis=initial_pose.get("selected_hypothesis"))
        if "R_physical" in initial_pose:
            physical = np.asarray(initial_pose["R_physical"], dtype=float)
            if (physical.shape != (3, 3) or not np.isfinite(physical).all()
                    or not np.allclose(physical.T @ physical, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(physical), 1., atol=1e-5)):
                raise ValueError("R_physical must be a proper rotation")
            pose["R_physical"] = (result_R @ (R.T @ physical)).tolist()
        return C.finite(dict(schema="pallet_visibility_curriculum_live_v1", actual_pose=pose,
            self_hidden=hidden.tolist(), appearance=appearance, solution=solution,
            observed_points=q.tolist(), projected_points=projected.tolist(), completed_points=completed.tolist(),
            provenance=dict(ground_truth_inputs=False, projected_points_used_for_fit=False,
                self_hidden_source="fixed_initial_pose", completed_points_source="observations_with_self_hidden_projection_after_accepted_fit",
                center_preserved=len(q) != 9 or np.array_equal(completed[8], q[8], equal_nan=True)),
            timing=dict(evidence_seconds=evidence_seconds, solve_seconds=solve_seconds,
                        total_postprocessing_seconds=time.perf_counter() - started)))


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON containing points, K and initial_pose")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous inference: " + str(args.output))
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(str(args.image))
    values = C.read(args.input)
    result = VisibilityCurriculum.from_config(args.config, args.device).refine(
        image, values["points"], values["K"], values["initial_pose"])
    C.write(args.output, result)
    print("accepted", result["solution"]["accepted"], "used", result["solution"]["used_ids"], args.output)


if __name__ == "__main__":
    main()
