"""Preserve image measurements while testing protected rigid pose proposals.

The reported 2D keypoints can mix measured and model-derived coordinates.
The separate rigid model always comes from one pose. Neither output is fed
back as an additional image observation.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .anchored_completion_api import AnchoredCompletion


class _CachedObservation:
    def __init__(self, observation):
        self.observation = observation

    def assess(self, *args):
        return deepcopy(self.observation)


def pose_proposal(name, pose, initial):
    """Convert a known physical basis without relabelling native keypoints.

    Alternate dimension families require an explicitly registered native
    correspondence and are rejected here. No annotation or best permutation
    is consulted.
    """
    extents = np.asarray(initial["cf_extents"], float)
    candidate_extents = np.asarray(pose["cf_extents"], float)
    if extents.shape != (3,) or not np.allclose(extents, candidate_extents, atol=1e-9, rtol=0):
        raise ValueError("Pose proposal changed the original dimension family")
    R0 = np.asarray(initial["R_cf"], float)
    physical0 = np.asarray(initial["R_physical"], float)
    Q0 = R0.T @ physical0
    candidate_R = np.asarray(pose["R_cf"], float)
    candidate_physical = np.asarray(pose["R_physical"], float)
    Qcandidate = candidate_R.T @ candidate_physical
    if not np.allclose(Qcandidate, Q0, atol=1e-7, rtol=0):
        raise ValueError("Pose proposal changed the native model-to-physical basis")
    # The explicit conversion also checks the source's paired rotations.
    native_R = candidate_physical @ Q0.T
    if not np.allclose(native_R, candidate_R, atol=1e-7, rtol=0):
        raise ValueError("Pose proposal has inconsistent rotations")
    return dict(name=name, R=native_R.tolist(), t=deepcopy(pose["centroid"]),
                cf_extents=extents.tolist(), coordinate_frame="ORIGINAL_NATIVE_CF",
                native_order_preserved=True)


class ProtectedCompletion:
    """Opt-in completion using fixed image evidence and finite pose proposals."""

    def __init__(self, anchored, *, progressive_config=None, joint_engine=None,
                 proposal_provider=None, proposal_config=None, region_model=None,
                 region_model_sha256=None, region_config=None):
        self.anchored = anchored
        self.progressive_config = deepcopy(progressive_config)
        self.joint_engine = joint_engine
        self.proposal_provider = proposal_provider
        self.proposal_config = deepcopy(proposal_config or {})
        self.region_model = deepcopy(region_model)
        self.region_model_sha256 = region_model_sha256
        self.region_config = deepcopy(region_config or {})
        self.config_binding = None

    @classmethod
    def from_config(cls, path, device=None):
        from .joint_api import JointSpatialBranch
        path = Path(path).resolve()
        config = C.read(path)
        if config.get("schema") != "protected_completion_config_v2":
            raise ValueError("Explicit protected_completion_config_v2 required")
        paths = {}
        for key in ("anchored_config", "progressive_config", "joint_config", "region_model"):
            if config.get(key):
                bound = (path.parent / config[key]).resolve()
                if C.sha(bound) != config["source_sha256"][key]:
                    raise ValueError("Frozen configuration changed: " + key)
                paths[key] = bound
        anchored = AnchoredCompletion.from_config(paths["anchored_config"], device)
        progressive = C.read(paths["progressive_config"])["curriculum"] if "progressive_config" in paths else None
        joint = JointSpatialBranch.from_config(paths["joint_config"], device) if "joint_config" in paths else None
        if joint is not None and joint.branch_policy != "preserve_original_family":
            raise ValueError("Pose proposals must preserve the original native dimension family")
        result = cls(anchored, progressive_config=progressive, joint_engine=joint,
                     proposal_config=config.get("proposal_rules"),
                     region_model=C.read(paths["region_model"]) if "region_model" in paths else None,
                     region_model_sha256=C.sha(paths["region_model"]) if "region_model" in paths else None,
                     region_config=config.get("region_rules"))
        result.config_binding = dict(path=str(path), sha256=C.sha(path))
        return result

    def refine(self, image_bgr, points, K, initial_pose, registered_dimensions,
               *, occluding_evidence=None):
        from .boundary_curriculum_api import BoundaryCurriculum
        from .protected_proposals import select_protected_pose, state_keypoint_readout
        started = time.perf_counter()
        base = self.anchored.refine(image_bgr, points, K, initial_pose, registered_dimensions,
                                   occluding_evidence=occluding_evidence)
        proposals, failures = [], []
        if self.progressive_config is not None:
            result = BoundaryCurriculum(_CachedObservation(base["observation_result"]),
                                        self.progressive_config).refine(
                image_bgr, points, K, initial_pose, registered_dimensions)
            proposals.append(pose_proposal("LEGACY_PROGRESSIVE", result["actual_pose"], initial_pose))
        if self.joint_engine is not None:
            result = self.joint_engine.refine(image_bgr, points, K, initial_pose, registered_dimensions)
            proposals.append(pose_proposal("LEARNED_JOINT", result["actual_pose"], initial_pose))
        if self.proposal_provider is not None:
            original_q = np.asarray(points, float).copy()
            original_K = np.asarray(K, float).copy()
            query, camera, pose = original_q.copy(), original_K.copy(), deepcopy(initial_pose)
            image = np.asarray(image_bgr).view()
            image.flags.writeable = False
            records = self.proposal_provider(image, query, camera, pose,
                                             deepcopy(registered_dimensions), deepcopy(base))
            if (not np.array_equal(query, original_q, equal_nan=True)
                    or not np.array_equal(camera, original_K, equal_nan=True)
                    or any(field not in pose or not np.array_equal(np.asarray(pose[field], float),
                        np.asarray(initial_pose[field], float), equal_nan=True)
                        for field in ("R_cf", "R_physical", "centroid", "cf_extents"))):
                raise ValueError("Pose proposal provider changed its immutable inputs")
            for record in records:
                try:
                    proposals.append(pose_proposal(record["name"], record["actual_pose"], initial_pose))
                except (ValueError, KeyError, TypeError) as error:
                    failures.append(dict(name=record.get("name"), reason=str(error)))
        xyz = C.cuboid(*initial_pose["cf_extents"])
        regions = []
        if self.region_model is None:
            solution = select_protected_pose(base["solution"], xyz, K, initial_pose["R_cf"],
                                             initial_pose["centroid"], proposals, self.proposal_config)
        else:
            from .empirical_regions import build_empirical_regions
            from .region_completion import select_region_pose
            from .keypoint_regions import image_pixel_sha256
            regions = build_empirical_regions(base["solution"], image_bgr, K, xyz,
                                              self.region_model, model_sha256=self.region_model_sha256)
            solution = select_region_pose(base["solution"], xyz, K, initial_pose["R_cf"],
                                          initial_pose["centroid"], regions, proposals, self.region_config,
                                          image_sha256=image_pixel_sha256(image_bgr))
        readout = state_keypoint_readout(solution)
        actual = deepcopy(initial_pose)
        if solution["accepted"]:
            R0 = np.asarray(initial_pose["R_cf"], float)
            R = np.asarray(solution["R"], float)
            actual.update(R_cf=solution["R"], centroid=solution["t"],
                          R_physical=(R @ (R0.T @ np.asarray(initial_pose["R_physical"], float))).tolist())
            if np.array_equal(R, R0):
                actual["R_cf"] = deepcopy(initial_pose["R_cf"])
                actual["R_physical"] = deepcopy(initial_pose["R_physical"])
        output = deepcopy(base)
        output.update(schema="protected_completion_inference_v2", actual_pose=actual,
                      solution=solution, reported_points=readout["reported_points"],
                      completed_points=readout["reported_points"],
                      rigid_model_corners=readout["rigid_model_corners"],
                      projected_points=readout["rigid_model_corners"], keypoint_readout=readout,
                      coarse_keypoint_regions=regions,
                      proposal_conversion_failures=failures, config_binding=self.config_binding)
        projected = np.asarray(readout["rigid_model_corners"], float)
        output["measurement_residuals"] = {
            str(k): (projected[k] - np.asarray(solution["measured_points"][k], float)).tolist()
            for k in solution["used_ids"]}
        output["provenance"].update(reported_points_are_one_rigid_pose=False,
            reported_image_measurements_preserved=True, hidden_interval_constraints_used=bool(regions),
            region_constraints_are_calibrated_probabilities=False,
            additional_proposals_are_measurements=False,
            proposal_choice_source="same immutable visible-image objective and fixed anchor guards",
            original_anchored_runtime_retained=True)
        output["timing"]["total_postprocessing_seconds"] = time.perf_counter() - started
        return C.finite(output)


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
    inputs = C.read(args.input)
    output = ProtectedCompletion.from_config(args.config, args.device).refine(
        image, inputs["points"], inputs["K"], inputs["initial_pose"], inputs["registered_dimensions"],
        occluding_evidence=inputs.get("occluding_evidence"))
    C.write(args.output, output)
    print("PROTECTED_COMPLETION_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
