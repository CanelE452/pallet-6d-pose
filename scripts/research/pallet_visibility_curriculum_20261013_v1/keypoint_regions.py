"""Typed coarse 2D regions for scoring an immutable finite pose pool.

These regions are not point correspondences. IMAGE_REGRESSED regions share
image/query evidence and have no default coverage guarantee. POSE_PRIOR regions
are diagnostic only: evaluating a pose against its own propagated prior cannot
create new independent evidence. This module performs no data/model/file I/O.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np


SCHEMA = "keypoint_region_v1"
IMAGE_REGRESSED = "IMAGE_REGRESSED"
POSE_PRIOR = "POSE_PRIOR"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_sha(value, kind):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(kind.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def native_query_sha256(points):
    """Bind native order, missing values and the optional ninth centre carrier."""
    q = np.asarray(points, float)
    if q.shape not in ((8, 2), (9, 2)):
        raise ValueError("points must have native shape (8,2) or (9,2)")
    return _array_sha(q, "native_query_v1")


def image_pixel_sha256(image_bgr):
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image_bgr must be a nonempty uint8 BGR image")
    return hashlib.sha256(b"BGR_PIXEL_SHA256_v1" + str(image.shape).encode()
                          + np.ascontiguousarray(image).tobytes()).hexdigest()


def _sha(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(field + " must be a lowercase SHA256 hex string")
    return value


def _canvas(value):
    a = np.asarray(value, float)
    if a.shape != (2,) or not np.isfinite(a).all() or (a <= 0).any() or not np.equal(a, np.floor(a)).all():
        raise ValueError("image_hw must contain positive integer height and width")
    return a.astype(int).tolist()


def _geometry(K, xyz):
    result = {}
    if K is not None:
        camera = np.asarray(K, float)
        if camera.shape != (3, 3) or not np.isfinite(camera).all() or abs(np.linalg.det(camera)) < 1e-12:
            raise ValueError("K must be a finite invertible 3x3 matrix")
        result["K_sha256"] = _array_sha(camera, "K_v1")
    if xyz is not None:
        model = np.asarray(xyz, float)
        if model.shape != (8, 3) or not np.isfinite(model).all():
            raise ValueError("xyz must be a finite native (8,3) model")
        result["xyz_sha256"] = _array_sha(model, "native_xyz_v1")
    return result


def _axes(scales_px):
    scales = np.asarray(scales_px, float)
    if scales.ndim == 0:
        scales = np.full((8, 2), float(scales))
    elif scales.shape == (8,):
        scales = np.repeat(scales[:, None], 2, axis=1)
    if scales.shape != (8, 2) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("scales_px must be positive scalar, (8,) radii or (8,2) half axes")
    return scales


def _hard_contract(policy):
    if policy is None:
        return None
    policy = deepcopy(policy)
    kind = policy.get("kind")
    if kind == "CALIBRATED_JOINT_COVERAGE":
        coverage = policy.get("coverage")
        if (not isinstance(coverage, (int, float)) or not np.isfinite(coverage)
                or not 0 < coverage < 1 or policy.get("joint_coverage") is not True
                or policy.get("exchangeability_assumed") is not True
                or not policy.get("calibration_reference") or not policy.get("assumptions_reference")):
            raise ValueError("hard pruning requires an explicit joint coverage and exchangeability contract")
        _sha(policy.get("calibration_model_sha256"), "calibration_model_sha256")
    elif kind == "INDEPENDENTLY_SPECIFIED_SEMANTIC_REGION":
        if (policy.get("ownership_verified") is not True
                or policy.get("region_geometry_verified") is not True
                or policy.get("independent_of_initial_projection") is not True
                or not policy.get("source_reference")):
            raise ValueError("semantic hard regions require ownership and independent geometry; a mask alone is insufficient")
    else:
        raise ValueError("empirical radii/point confidence do not authorize hard pruning")
    # The caller supplies this scientific contract. Its validity is not proved
    # by accepting metadata, nor does it guarantee the finite pool contains truth.
    _digest(policy)
    return policy


def build_keypoint_regions(points, image_bgr=None, *, scales_px, states=None,
                           native_ids=None, image_sha256=None, image_hw=None,
                           K=None, xyz=None, radius_rule=None,
                           calibration_model_sha256=None, shape="ellipse",
                           hard_policy=None):
    """Return available raw-query-centred native regions, never a centre region.

    Cached prediction-only drivers may pass a sealed file SHA + canvas instead
    of pixels. That binding is explicitly not a decoded-pixel verification.
    OOB raw predictions remain coarse predictions, not physical truncation facts.
    """
    q = np.asarray(points, float)
    query_sha = native_query_sha256(q)
    axes = _axes(scales_px)
    if shape not in ("ellipse", "axis_aligned_box"):
        raise ValueError("shape must be ellipse or axis_aligned_box")
    if image_bgr is not None:
        pixel_sha = image_pixel_sha256(image_bgr)
        canvas = list(np.asarray(image_bgr).shape[:2])
        if image_hw is not None and _canvas(image_hw) != canvas:
            raise ValueError("image canvas binding mismatch")
        if image_sha256 is not None and _sha(image_sha256, "image_sha256") != pixel_sha:
            raise ValueError("provided image_sha256 does not match decoded BGR pixels")
        binding = dict(sha256=pixel_sha, kind="BGR_PIXEL_SHA256", verified_pixels=True,
                       image_hw=canvas)
    else:
        binding = dict(sha256=_sha(image_sha256, "image_sha256"), kind="SEALED_FILE_SHA256",
                       verified_pixels=False, image_hw=_canvas(image_hw))
    if states is not None and len(states) != 8:
        raise ValueError("states must have eight native entries")
    ids = list(range(8)) if native_ids is None else list(native_ids)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k < 8 for k in ids) or len(ids) != len(set(ids)):
        raise ValueError("native_ids must be unique integers in 0..7")
    geometry = _geometry(K, xyz)
    policy = _hard_contract(hard_policy)
    if calibration_model_sha256 is not None:
        _sha(calibration_model_sha256, "calibration_model_sha256")
    if policy and policy["kind"] == "CALIBRATED_JOINT_COVERAGE" and policy["calibration_model_sha256"] != calibration_model_sha256:
        raise ValueError("hard coverage model must match the region radius calibration binding")
    records = []
    for k in ids:
        if not np.isfinite(q[k]).all() or np.array_equal(q[k], [-1., -1.]):
            continue
        h, w = binding["image_hw"]
        record = dict(schema=SCHEMA, native_corner=int(k), source=IMAGE_REGRESSED,
                      center_xy=q[k].tolist(), raw_query_xy=q[k].tolist(),
                      half_axes_px=axes[k].tolist(), shape=shape,
                      state=None if states is None else str(states[k]),
                      query_sha256=query_sha, image_binding=deepcopy(binding),
                      geometry_binding=deepcopy(geometry), radius_rule=deepcopy(radius_rule),
                      calibration_model_sha256=calibration_model_sha256,
                      hard_policy=deepcopy(policy), raw_query_out_of_frame=bool(not (0 <= q[k, 0] < w and 0 <= q[k, 1] < h)),
                      physical_truncation_certified=False,
                      independently_specified_image_observation=False,
                      independent_precise_support_count=0,
                      default_coverage_guaranteed=False)
        record["region_sha256"] = _digest(record)
        records.append(record)
    return records


def build_pose_prior_regions(projected_points, *, scales_px, native_ids=None,
                             K=None, xyz=None, pose_reference=None, shape="ellipse"):
    """Typed pose propagation diagnostics; never image likelihood or pruning."""
    q = np.asarray(projected_points, float)
    native_query_sha256(q)
    axes = _axes(scales_px)
    if shape not in ("ellipse", "axis_aligned_box"):
        raise ValueError("unsupported prior region shape")
    ids = list(range(8)) if native_ids is None else list(native_ids)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k < 8 for k in ids) or len(ids) != len(set(ids)):
        raise ValueError("invalid native prior IDs")
    geometry = _geometry(K, xyz)
    result = []
    for k in ids:
        if not np.isfinite(q[k]).all() or np.array_equal(q[k], [-1., -1.]):
            continue
        record = dict(schema=SCHEMA, native_corner=int(k), source=POSE_PRIOR,
                      center_xy=q[k].tolist(), half_axes_px=axes[k].tolist(), shape=shape,
                      geometry_binding=deepcopy(geometry), pose_reference=deepcopy(pose_reference),
                      hard_policy=None, independent_precise_support_count=0,
                      contributes_image_likelihood=False, default_coverage_guaranteed=False)
        record["region_sha256"] = _digest(record)
        result.append(record)
    return result


def _validate_region(record, geometry, raw_points, image_sha):
    body = {k: v for k, v in record.items() if k != "region_sha256"}
    if record.get("schema") != SCHEMA or _digest(body) != record.get("region_sha256"):
        raise ValueError("region content/provenance binding mismatch")
    k = record.get("native_corner")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < 8:
        raise ValueError("invalid native region ID")
    center, axes = np.asarray(record.get("center_xy"), float), np.asarray(record.get("half_axes_px"), float)
    if center.shape != (2,) or axes.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(axes).all() or (axes <= 0).any():
        raise ValueError("invalid region coordinates")
    if record.get("shape") not in ("ellipse", "axis_aligned_box"):
        raise ValueError("invalid region shape")
    for name, value in record.get("geometry_binding", {}).items():
        if name not in geometry or value != geometry[name]:
            raise ValueError("region native model/camera binding mismatch")
    source = record.get("source")
    if source == IMAGE_REGRESSED:
        binding = record.get("image_binding", {})
        _sha(binding.get("sha256"), "region image SHA")
        _canvas(binding.get("image_hw"))
        if binding.get("kind") not in ("BGR_PIXEL_SHA256", "SEALED_FILE_SHA256"):
            raise ValueError("invalid image binding namespace")
        if image_sha is not None and binding["sha256"] != _sha(image_sha, "image_sha256"):
            raise ValueError("region image binding mismatch")
        if raw_points is not None:
            q = np.asarray(raw_points, float)
            if native_query_sha256(q) != record.get("query_sha256") or not np.array_equal(center, q[k]):
                raise ValueError("region native raw-query binding mismatch")
        policy = _hard_contract(record.get("hard_policy"))
        if policy and policy["kind"] == "CALIBRATED_JOINT_COVERAGE" and policy["calibration_model_sha256"] != record.get("calibration_model_sha256"):
            raise ValueError("hard region calibration model mismatch")
    elif source == POSE_PRIOR:
        if record.get("hard_policy") is not None:
            raise ValueError("pose prior cannot authorize independent hard pruning")
    else:
        raise ValueError("unknown region source")
    return center, axes


def _distance(delta, axes, shape):
    if shape == "axis_aligned_box":
        excess = np.maximum(np.abs(delta)-axes, 0.)
        return float(excess @ excess), float(np.sum((excess/axes)**2))
    norm = float(np.linalg.norm(delta/axes))
    normalized = max(norm-1., 0.)**2
    if norm <= 1:
        return 0., 0.
    # Exact Euclidean nearest point on an axis-aligned ellipse, using the
    # unique nonnegative multiplier for an outside query (including axis cases).
    square = axes**2
    def constraint(lam):
        return float(np.sum((axes*delta/(lam+square))**2))
    lo, hi = 0., max(float(np.linalg.norm(axes*delta)), 1.)
    while constraint(hi) > 1:
        hi *= 2
    for _ in range(64):
        middle = (lo+hi)/2
        if constraint(middle) > 1:
            lo = middle
        else:
            hi = middle
    closest = square*delta/(hi+square)
    difference = delta-closest
    return float(difference @ difference), float(normalized)


def _pose(candidate):
    R = np.asarray(candidate.get("R", candidate.get("R_cf")), float)
    t = np.asarray(candidate.get("t", candidate.get("centroid")), float).reshape(-1)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0.) or abs(np.linalg.det(R)-1.) > 1e-6):
        raise ValueError("candidate must contain a finite proper R/t pose")
    return R, t


def evaluate_pose_regions(candidate_poses, xyz, K, regions, *, weights=None,
                          raw_points=None, image_sha256=None):
    """Score supplied finite poses in order; never fit, select or add support.

    normalized_penalty is the weighted mean of IMAGE_REGRESSED outside losses.
    Pixel penalty is nearest-region Euclidean gap squared in native pixels.
    hard_valid is meaningful only under an explicit caller hard-region contract.
    The caller must separately enforce depth, precise support and anchor guards.
    """
    if K is None or xyz is None:
        raise ValueError("region evaluation requires the native model and camera")
    geometry = _geometry(K, xyz)
    model, camera = np.asarray(xyz, float), np.asarray(K, float)
    records = list(regions)
    keys = [(r.get("source"), r.get("native_corner")) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate source/native regions cannot multiply evidence")
    validated = [_validate_region(r, geometry, raw_points, image_sha256) for r in records]
    image_records = [r for r in records if r["source"] == IMAGE_REGRESSED]
    if image_records:
        identity = [(r["query_sha256"], r["image_binding"]["sha256"],
                     r["image_binding"]["kind"], tuple(r["image_binding"]["image_hw"]))
                    for r in image_records]
        if len(set(identity)) != 1:
            raise ValueError("mixed frame/query region binding mismatch")
    if weights is None:
        region_weights = np.ones(len(records))
    elif isinstance(weights, dict):
        region_weights = np.array([weights.get(r["native_corner"], weights.get(str(r["native_corner"]), 1.)) for r in records], float)
    else:
        supplied = np.asarray(weights, float)
        if supplied.shape != (8,):
            raise ValueError("weights must be a native eight-vector or native-ID mapping")
        region_weights = np.array([supplied[r["native_corner"]] for r in records])
    if not np.isfinite(region_weights).all() or (region_weights < 0).any():
        raise ValueError("region weights must be finite and nonnegative")
    results = []
    for index, candidate in enumerate(candidate_poses):
        R, t = _pose(candidate)
        homogeneous = (model @ R.T+t) @ camera.T
        pixel_sum, normalized_sum, weight_sum, hard_valid = 0., 0., 0., True
        details = []
        for record, (center, axes), weight in zip(records, validated, region_weights):
            k, source = record["native_corner"], record["source"]
            finite = bool(np.isfinite(homogeneous[k]).all() and abs(homogeneous[k, 2]) > 1e-12)
            projected = homogeneous[k, :2]/homogeneous[k, 2] if finite else None
            pixel_loss, normalized_loss = _distance(projected-center, axes, record["shape"]) if finite else (1e12, 1e12)
            contributes = bool(source == IMAGE_REGRESSED and weight > 0)
            if contributes:
                pixel_sum += weight*pixel_loss
                normalized_sum += weight*normalized_loss
                weight_sum += weight
            hard = source == IMAGE_REGRESSED and record.get("hard_policy") is not None
            if hard and (not finite or normalized_loss > 1e-12):
                hard_valid = False
            details.append(dict(native_corner=k, source=source, projected_xy=None if projected is None else projected.tolist(),
                                pixel_gap_squared_px2=pixel_loss, normalized_outside_loss=normalized_loss,
                                inside=bool(finite and normalized_loss <= 1e-12),
                                contributes_to_penalty=contributes, hard_constraint=hard,
                                weight=float(weight), region_sha256=record["region_sha256"]))
        results.append(dict(name=str(candidate.get("name", index)), candidate_index=index,
                            normalized_penalty=float(normalized_sum/weight_sum) if weight_sum else 0.,
                            pixel_penalty_px2=float(pixel_sum/weight_sum) if weight_sum else 0.,
                            hard_valid=bool(hard_valid), image_region_weight_sum=float(weight_sum),
                            image_region_count=sum(d["contributes_to_penalty"] for d in details),
                            independent_precise_support_count=0, regions=details))
    return dict(schema="finite_pose_region_assessment_v1", candidates=results,
                regions_are_point_correspondences=False, independent_precise_support_count=0,
                image_regions_statistically_independent=False, calibrated_pose_probability=False,
                finite_pool_contains_true_pose_guaranteed=False,
                hard_policy_contract_verified_statistically=False,
                pose_prior_contributes_to_penalty=False)
