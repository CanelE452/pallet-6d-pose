"""Pose evaluation paths, with prediction and scoring separated at the type level.

The separation is the point.  `predict_pose_without_gt` has no ground-truth
parameter, so a reviewer can verify by reading the signature that the axis decision
cannot have been made by looking at the answer.

    main path      predicted keypoints -> prediction-only selector -> SQPnP ->
                   RefineLM -> (R_pred, t_pred),  and only then the reviewed GT

    oracle path    the same keypoints, but the human-reviewed long axis picks the
                   hypothesis.  Diagnostic only, always tagged, never merged into
                   the main table.

Why both exist:

    main poor / oracle good     the axis selector is the bottleneck
    main poor / oracle poor     keypoint localisation or PnP geometry is also a
                                bottleneck
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ORACLE_TAG = "oracle"
MAIN_TAG = "main"

CF_WIDTH = "CF_WIDTH"
CF_DEPTH = "CF_DEPTH"


def _extents_for_long_axis(long_axis: str, long_m: float, short_m: float,
                           height_m: float) -> tuple[float, float, float]:
    """(across, height, along) for the camera-facing cuboid.

    `across` is the Axis A (WIDTH) extent, `along` is the Axis B (DEPTH) extent.
    """

    if long_axis == CF_WIDTH:
        return (long_m, height_m, short_m)
    if long_axis == CF_DEPTH:
        return (short_m, height_m, long_m)
    raise ValueError(f"long_axis must be {CF_WIDTH} or {CF_DEPTH}, got {long_axis!r}")


def predict_pose_without_gt(
    predicted_keypoints: Sequence[Sequence[float]] | np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]] | np.ndarray,
    physical_long_m: float,
    physical_short_m: float,
    physical_height_m: float,
    selector_config=None,
):
    """Choose a W/D hypothesis and solve for the pose, using no ground truth.

    **This function must never gain a ground-truth parameter.** A contract test
    asserts the signature of the underlying selector for exactly that reason.

    Returns a dict with the selected hypothesis and the resulting pose, or a
    POSE_UNRESOLVED marker when the selector declines to choose.
    """

    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    result = select_pnp_hypotheses(
        predicted_keypoints,
        camera_intrinsics,
        {"x": physical_long_m, "y": physical_height_m, "z": physical_short_m},
        selector_config,
    )
    return {
        "mode": MAIN_TAG,
        "is_oracle": False,
        "selector_result": result,
        "gt_consulted": False,
    }


def predict_pose_with_oracle_axis(
    predicted_keypoints: Sequence[Sequence[float]] | np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]] | np.ndarray,
    reviewed_long_axis: str,
    physical_long_m: float,
    physical_short_m: float,
    physical_height_m: float,
):
    """Upper bound: the human-reviewed axis picks the hypothesis.

    Diagnostic only.  Every record it produces carries `is_oracle = True`, and the
    main tables never contain a value produced here.
    """

    import cv2

    across, height, along = _extents_for_long_axis(
        reviewed_long_axis, physical_long_m, physical_short_m, physical_height_m)
    model = cuboid_model_points((across, height, along))
    points = np.asarray(predicted_keypoints, dtype=np.float64)[:8]
    camera = np.asarray(camera_intrinsics, dtype=np.float64)
    usable = np.isfinite(points).all(axis=1)
    if usable.sum() < 6:
        return {"mode": ORACLE_TAG, "is_oracle": True, "status": "POSE_UNRESOLVED",
                "reason": "fewer than six usable keypoints"}
    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return {"mode": ORACLE_TAG, "is_oracle": True, "status": "POSE_UNRESOLVED",
                "reason": "SQPnP failed"}
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "mode": ORACLE_TAG,
        "is_oracle": True,
        "status": "OK",
        "rotation": rotation,
        "translation": tvec.reshape(-1),
        "long_axis": reviewed_long_axis,
        "gt_consulted": True,
        "gt_consulted_for": "axis hypothesis selection (this is what makes it an oracle)",
    }


def score_pose_against_gt(
    model_points,
    predicted_rotation, predicted_translation,
    target_rotation, target_translation,
    extents,
) -> dict:
    """Compare a solved pose with the reviewed ground truth.

    Called only after the prediction path has finished.
    """

    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from symmetry_aware_pose_metrics import (
        model_diameter_m,
        rotation_error_degrees,
        symmetry_aware_add_m,
        translation_components_m,
        yaw_error_degrees,
    )

    parts = translation_components_m(predicted_translation, target_translation)
    add = symmetry_aware_add_m(model_points, predicted_rotation, predicted_translation,
                               target_rotation, target_translation)
    diameter = model_diameter_m(model_points)
    return {
        "rotation_error_deg": rotation_error_degrees(predicted_rotation, target_rotation),
        "yaw_error_deg": yaw_error_degrees(predicted_rotation, target_rotation),
        "translation_error_m": parts["total_m"],
        "translation_error_cm": parts["total_m"] * 100.0,
        "lateral_error_cm": parts["lateral_m"] * 100.0,
        "depth_error_cm": parts["depth_m"] * 100.0,
        "iou3d": oriented_iou_3d(predicted_rotation, predicted_translation, extents,
                                 target_rotation, target_translation, extents),
        "symmetry_aware_add_m": add,
        "symmetry_aware_add_normalized": add / diameter,
        "model_diameter_m": diameter,
    }


def evaluate_frame(model_points, predicted_rotation, predicted_translation,
                   target_rotation, target_translation, extents, *,
                   mode: str = MAIN_TAG) -> dict:
    """Score one frame and stamp which path produced it."""

    if mode not in (MAIN_TAG, ORACLE_TAG):
        raise ValueError(f"mode must be {MAIN_TAG!r} or {ORACLE_TAG!r}")
    record = score_pose_against_gt(model_points, predicted_rotation,
                                   predicted_translation, target_rotation,
                                   target_translation, extents)
    record["mode"] = mode
    record["is_oracle"] = mode == ORACLE_TAG
    return record


def cuboid_model_points(extents: Sequence[float]) -> np.ndarray:
    from symmetry_aware_pose_metrics import cuboid_model_points as build

    return build(extents)
