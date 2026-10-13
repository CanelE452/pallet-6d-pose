"""Inference-only coarse circles fitted on a separately recorded CAL study.

An empirical quantile is an exploratory tolerance. It is not a joint conformal
guarantee or an image-conditional probability. The source neural predictions
share RGB and may have already contributed to the upstream initial pose.
"""
from copy import deepcopy

import numpy as np


def prediction_scale(points):
    q = np.asarray(points, float)[:8]
    valid = np.isfinite(q).all(1) & ~np.all(q == -1, axis=1)
    return max(float(np.linalg.norm(np.ptp(q[valid], axis=0))) if valid.any() else 0., 16.)


def region_queries(solution):
    raw = np.asarray(solution["raw_points"], float)[:8]
    valid = np.isfinite(raw).all(1) & ~np.all(raw == -1, axis=1)
    pool = set(solution["immutable_pool_ids"])
    hidden = np.asarray(solution["self_hidden"], bool)
    return {k: "INITIAL_MODEL_SELF" if hidden[k] else "OTHER_UNOBSERVED"
            for k in range(8) if valid[k] and k not in pool}


def build_empirical_regions(solution, image_bgr, K, xyz, model, *, model_sha256=None):
    from .keypoint_regions import build_keypoint_regions
    if model.get("schema") != "empirical_CAL_coarse_keypoint_regions_v1":
        raise ValueError("Explicit empirical coarse-region model required")
    if model.get("calibrated_joint_coverage") is not False:
        raise ValueError("This model provides no calibrated joint probability")
    queries = region_queries(solution)
    scale = prediction_scale(solution["raw_points"])
    scales = np.full(8, scale*float(model["pooled_normalized_radius"]))
    for k, category in queries.items():
        scales[k] = scale*float(model["categories"][category]["normalized_radius"])
    states = [queries.get(k, "PRECISE_POOL_NO_COARSE_REGION") for k in range(8)]
    result = build_keypoint_regions(solution["raw_points"], image_bgr,
        scales_px=scales, states=states, native_ids=sorted(queries), K=K, xyz=xyz,
        radius_rule=deepcopy(model["radius_rule"]), calibration_model_sha256=model_sha256)
    return result
