"""Separate geometric visibility, RGB evidence, and observation usability.

This module consumes already computed predictions. It reads no files, loads no
models, samples no image pixels, and estimates no pose. Initial projections are
conditional geometric diagnostics, never independent measurements. The optional
ninth centre observation is neither interpreted as a corner nor changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KeypointStatus:
    """Immutable evidence record; geometry and RGB flags may coexist."""

    corner: int
    raw_xy: Tuple[Optional[float], Optional[float]]
    prediction_missing: bool
    prediction_out_of_frame: Optional[bool]
    geometric_self_occluded: bool
    initial_projection_xy: Tuple[Optional[float], Optional[float]]
    initial_projection_valid: bool
    initial_projection_out_of_frame: Optional[bool]
    conditional_truncation: Optional[bool]
    rgb_query_eligible: bool
    source_visibility_state: str
    rgb_visibility_state: str
    rgb_visibility_score: Optional[float]
    rgb_boundary_role: str
    rgb_boundary_score: Optional[float]
    low_RGB_support_hypothesis: Optional[str]
    joint_usable_score: float
    joint_admission_threshold: float
    joint_eligible: bool
    joint_admitted: bool
    observation_usability_state: str
    geometry_source: str = "fixed_original_initial_pose"
    RGB_query_source: str = "original_independent_predicted_xy"
    truncation_is_conditional_on_initial_pose: bool = True
    RGB_3D_visibility_certified: bool = False
    external_occlusion_confirmed: bool = False
    projected_coordinates_used_as_measurements: bool = False

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible copy."""
        result = asdict(self)
        result["raw_xy"] = list(self.raw_xy)
        result["initial_projection_xy"] = list(self.initial_projection_xy)
        return result


def status_records_to_json(records) -> list:
    """Serialize records without exposing mutable arrays or shared dictionaries."""
    return [record.to_dict() for record in records]


def _scores(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (8,) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError(name + " must contain eight finite scores in [0,1]")
    return result


def _booleans(value, name):
    result = np.asarray(value)
    if result.shape != (8,) or not np.isin(result, [False, True]).all():
        raise ValueError(name + " must contain eight boolean flags")
    return result.astype(bool)


def _xy(value):
    return tuple(float(v) if np.isfinite(v) else None for v in value)


def assess_keypoint_status(points, K, initial_pose, self_hidden,
                          visibility_appearance, joint_prediction, *, image_hw):
    """Return eight immutable records without changing admission or observations.

    ``visibility_appearance`` must be the calibrated RGB/segmentation appearance
    saved BEFORE replacing its visibility states/scores with joint P_usable.
    A visible RGB query with low usability remains visible in this assessment.
    Low RGB support is only an occlusion/localization hypothesis. Neither a
    foreground-mask pixel nor P_usable proves the requested physical3D corner.

    ``image_hw`` is the actual (height,width) canvas. Camera principal points
    cannot supply it. A projected initial corner outside this canvas produces a
    conditional truncation flag even when its raw detector query is inside.
    This flag does not modify the predictor's original H/missing/outside gates.
    """
    q = np.asarray(points, dtype=float)
    camera = np.asarray(K, dtype=float)
    hidden = _booleans(self_hidden, "self_hidden")
    hw = np.asarray(image_hw)
    if hw.shape != (2,) or not np.isfinite(hw).all() or np.any(hw <= 0) or np.any(hw != np.floor(hw)):
        raise ValueError("image_hw must be actual positive integer (height,width)")
    height, width = map(int, hw)
    if q.shape not in ((8, 2), (9, 2)):
        raise ValueError("points must be eight native corners or eight corners plus centre")
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or min(camera[0, 0], camera[1, 1]) <= 0:
        raise ValueError("K must be finite with positive focal lengths")
    R = np.asarray(initial_pose["R_cf"], dtype=float)
    t = np.asarray(initial_pose["centroid"], dtype=float)
    dimensions = np.asarray(initial_pose["cf_extents"], dtype=float)
    if (R.shape != (3, 3) or t.shape != (3,) or dimensions.shape != (3,)
            or not np.isfinite(R).all() or not np.isfinite(t).all()
            or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-6)):
        raise ValueError("Initial R_cf/centroid/extents must define a proper finite cuboid")
    if visibility_appearance.get("diagnostics", {}).get("joint_observation") is not None:
        raise ValueError("Save independent visibility_appearance before joint usability overwrite")
    states = tuple(str(s).upper() for s in visibility_appearance["states"])
    if len(states) != 8:
        raise ValueError("visibility_appearance.states must have eight native entries")
    visibility = _scores(visibility_appearance["visibility_score"], "RGB visibility")
    boundary = _scores(visibility_appearance["boundary_score"], "RGB boundary")
    probability = _scores(joint_prediction["P_usable"], "joint P_usable")
    joint_eligible = _booleans(joint_prediction["eligible"], "joint eligible")
    threshold = float(joint_prediction["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Joint admission threshold must be finite in [0,1]")

    valid = np.isfinite(q[:8]).all(axis=1) & ~np.all(q[:8] == -1, axis=1)
    inside = valid & (q[:8, 0] >= 0) & (q[:8, 0] < width) & (q[:8, 1] >= 0) & (q[:8, 1] < height)
    query_eligible = inside & ~hidden
    if not np.array_equal(joint_eligible, query_eligible):
        raise ValueError("Joint eligibility must retain original missing/outside/self gates")
    allowed_RGB = {"VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "UNKNOWN", "OCCLUSION_SUSPECTED"}
    if any(query_eligible[k] and states[k] not in allowed_RGB for k in range(8)):
        raise ValueError("Eligible RGB query has an incompatible visibility state")

    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    camera_points = (signs * (dimensions / 2)) @ R.T + t
    homogeneous = camera_points @ camera.T
    projectable = (camera_points[:, 2] > 1e-6) & np.isfinite(homogeneous).all(axis=1) & (homogeneous[:, 2] > 1e-6)
    projected = np.full((8, 2), np.nan)
    projected[projectable] = homogeneous[projectable, :2] / homogeneous[projectable, 2:3]

    result = []
    for k in range(8):
        projected_outside = (not (0 <= projected[k, 0] < width and 0 <= projected[k, 1] < height)) if projectable[k] else None
        raw_outside = bool(not inside[k]) if valid[k] else None
        RGB_available = bool(query_eligible[k])
        RGB_state = states[k] if RGB_available else "NOT_ASSESSED"
        role = "BOUNDARY" if RGB_state == "VISIBLE_BOUNDARY" else "INTERIOR" if RGB_state == "VISIBLE_INTERIOR" else "UNKNOWN"
        hypothesis = "POSSIBLE_EXTERNAL_OCCLUSION_OR_LOCALIZATION_ERROR" if RGB_available and RGB_state in ("UNKNOWN", "OCCLUSION_SUSPECTED") else None
        admitted = bool(joint_eligible[k] and probability[k] >= threshold)
        usability = "ADMITTED" if admitted else "LOW_USABILITY" if joint_eligible[k] else "GEOMETRICALLY_EXCLUDED"
        result.append(KeypointStatus(
            corner=k, raw_xy=_xy(q[k]), prediction_missing=bool(not valid[k]),
            prediction_out_of_frame=raw_outside, geometric_self_occluded=bool(hidden[k]),
            initial_projection_xy=_xy(projected[k]), initial_projection_valid=bool(projectable[k]),
            initial_projection_out_of_frame=projected_outside, conditional_truncation=projected_outside,
            rgb_query_eligible=RGB_available, source_visibility_state=states[k], rgb_visibility_state=RGB_state,
            rgb_visibility_score=float(visibility[k]) if RGB_available else None,
            rgb_boundary_role=role, rgb_boundary_score=float(boundary[k]) if RGB_available else None,
            low_RGB_support_hypothesis=hypothesis, joint_usable_score=float(probability[k]),
            joint_admission_threshold=threshold, joint_eligible=bool(joint_eligible[k]),
            joint_admitted=admitted, observation_usability_state=usability))
    return tuple(result)
