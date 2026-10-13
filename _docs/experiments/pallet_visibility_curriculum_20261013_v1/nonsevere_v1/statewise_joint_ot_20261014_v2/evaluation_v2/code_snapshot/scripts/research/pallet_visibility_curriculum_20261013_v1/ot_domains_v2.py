"""External-occlusion uncertainty and real-crop truncation domains.

This new experimental module never treats self-occlusion as image evidence.
O radii use actual external-O CAL errors only. They are empirical tolerances,
not image-conditional probabilities or a 95% joint coverage guarantee. T has
no learned finite location here: known outside-frame halfplanes are retained.
Prediction functions have no ground-truth-coordinate argument.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

import numpy as np


MODEL_SCHEMA = "OT_EXTERNAL_CAL_ERROR_V2"
DOMAIN_SCHEMA = "STATEWISE_OT_DOMAINS_V2"
STATE_ALIASES = {
    "B": "V", "I": "V",
    "DIRECT_VISIBLE": "V", "VISIBLE_BOUNDARY": "V", "VISIBLE_INTERIOR": "V",
    "EXTERNAL_OCCLUDED": "O", "EXTERNALLY_OCCLUDED": "O",
    "SELF_OCCLUDED": "S", "TRUNCATED": "T", "OUT_OF_FRAME": "T",
    "UNANNOTATED": "U", "UNKNOWN": "U",
}
OUTSIDE_SIDES = {"left", "right", "top", "bottom", "unknown"}


def canonical_states(states, self_hidden=None):
    """Resolve S before any image-feature or native-geometry indexing."""
    result = [STATE_ALIASES.get(str(s).upper(), str(s).upper()) for s in states]
    if len(result) != 8 or any(s not in "VOTSU" or len(s) != 1 for s in result):
        raise ValueError("states must contain eight native B/I/V/O/T/S/U states")
    if self_hidden is not None:
        hidden = np.asarray(self_hidden)
        if hidden.shape != (8,) or not np.isin(hidden, [False, True]).all():
            raise ValueError("self_hidden must contain eight booleans")
        result = ["S" if hidden[k] else s for k, s in enumerate(result)]
    return result


def ot_native_ids(states, self_hidden=None):
    return [k for k, state in enumerate(canonical_states(states, self_hidden)) if state in ("O", "T")]


def _points(points):
    q = np.asarray(points, dtype=float)
    if q.shape not in ((8, 2), (9, 2)):
        raise ValueError("raw_points must be eight native corners, optionally plus centre")
    return q


def _hw(image_hw):
    value = np.asarray(image_hw)
    if value.shape != (2,) or not np.isfinite(value).all() or np.any(value <= 0) or np.any(value != np.floor(value)):
        raise ValueError("image_hw must be actual positive integer height,width")
    return tuple(map(int, value))


def _valid_xy(xy):
    return bool(np.isfinite(xy).all() and not np.all(np.asarray(xy) == -1))


def nonself_prediction_scale(raw_points, states, self_hidden=None):
    """Raw-query bbox diagonal; S is removed before coordinate indexing.

    Translation-only real crops preserve this scale, including surviving raw
    outside predictions. Neither pose projections nor dimensions define it.
    """
    effective = canonical_states(states, self_hidden)
    ids = [k for k, state in enumerate(effective) if state != "S"]
    q = _points(raw_points)[ids]
    valid = np.isfinite(q).all(1) & ~np.all(q == -1, axis=1)
    return max(float(np.linalg.norm(np.ptp(q[valid], axis=0))) if valid.any() else 0., 16.)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def image_pixel_sha256(image):
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8 or min(pixels.shape[:2]) <= 0:
        raise ValueError("image_bgr must be nonempty uint8 HxWx3")
    digest = hashlib.sha256(str(pixels.shape).encode())
    digest.update(np.ascontiguousarray(pixels).tobytes())
    return digest.hexdigest()


def fit_external_o_error_model(rows, *, quantile=1.0, minimum_radius_px=8.,
                               prediction_source="raw_backbone"):
    """Fit a tiny-data uncertainty model on genuine, in-frame CAL external O.

    Each row must explicitly supply CAL, HUMAN external_label_confirmed, an
    original image prediction ``raw_xy``, its actual ``target_xy``, and a scale
    computed by ``nonself_prediction_scale``. Non-O/S rows are skipped before
    accessing coordinates. TEST/FINAL/SEVERE rows are rejected before labels.
    No pooled self-hidden, pseudo-label, or all-unobserved error is accepted.
    """
    quantile, floor = float(quantile), float(minimum_radius_px)
    if not np.isfinite(quantile) or not 0 < quantile <= 1 or not np.isfinite(floor) or floor <= 0:
        raise ValueError("quantile in (0,1] and positive minimum_radius_px are required")
    samples, excluded = [], Counter()
    for row in rows:
        if str(row.get("split", row.get("partition", ""))).upper() != "CAL":
            raise ValueError("Only CAL rows may train external-O domains")
        if str(row.get("severity", "CLEAN")).upper() not in ("CLEAN", "MODERATE"):
            raise ValueError("Only nonsevere CLEAN/MODERATE rows may train domains")
        state = STATE_ALIASES.get(str(row.get("state", "")).upper(), str(row.get("state", "")).upper())
        if state != "O" or bool(row.get("self_hidden", False)):
            excluded["not_actual_external_O_or_fixed_S"] += 1
            continue
        if row.get("external_label_confirmed") is not True or str(row.get("label_source", "")).upper() not in ("HUMAN", "HUMAN_EXTERNAL_OCCLUDED"):
            raise ValueError("Actual HUMAN external-O labels are required")
        if row.get("prediction_source", prediction_source) != prediction_source:
            raise ValueError("CAL and inference must use the same image prediction source")
        raw, target = np.asarray(row["raw_xy"], float), np.asarray(row["target_xy"], float)
        if raw.shape != (2,) or target.shape != (2,):
            raise ValueError("CAL raw_xy and actual target_xy must have shape[2]")
        height, width = _hw(row["image_hw"])
        if not _valid_xy(raw) or not (0 <= raw[0] < width and 0 <= raw[1] < height):
            excluded["raw_missing_or_outside"] += 1
            continue
        if not _valid_xy(target) or not (0 <= target[0] < width and 0 <= target[1] < height):
            raise ValueError("O calibration must have a real in-frame external-O target")
        scale = float(row["scale_px"])
        if not np.isfinite(scale) or scale < 16:
            raise ValueError("scale_px must be a finite nonself raw-query diagonal >=16")
        corner = int(row.get("native_corner", row.get("native_id", -1)))
        if corner not in range(8):
            raise ValueError("Calibration native corner must lie in [0,7]")
        error = float(np.linalg.norm(target - raw))
        samples.append(dict(record_id=str(row["record_id"]), session=str(row.get("session", "unknown")),
                            native_corner=corner, error_px=error, scale_px=scale,
                            normalized_error=error / scale,
                            source_image_sha256=row.get("image_sha256")))
    if not samples:
        raise ValueError("No eligible actual external-O CAL errors")
    radius = float(np.quantile([s["normalized_error"] for s in samples], quantile, method="higher"))
    model = dict(schema=MODEL_SCHEMA, prediction_source=str(prediction_source),
                 normalized_radius=radius, minimum_radius_px=floor, quantile=quantile,
                 sample_count=len(samples), recording_count=len({s["session"] for s in samples}),
                 training_record_ids=sorted({s["record_id"] for s in samples}), samples=samples,
                 excluded=dict(excluded), trained_states=["O"], calibrated_joint_coverage=False,
                 image_conditional_probability=False, confidence_guarantee=None,
                 source="actual_HUMAN_external_O_CAL_image_prediction_errors",
                 center_rule="original_current_RGB_image_prediction",
                 radius_rule="max(minimum_radius_px,nonself_raw_query_diagonal*empirical_CAL_quantile)",
                 self_hidden_samples_used=False, TEST_inputs_used=False,
                 candidate_pose_inputs_used=False, note="Small-data empirical tolerance; no 95% claim")
    model["model_sha256"] = _digest(model)
    return model


def _validate_model(model, prediction_source):
    if model is None:
        return
    if model.get("schema") != MODEL_SCHEMA or model.get("trained_states") != ["O"] or model.get("calibrated_joint_coverage") is not False:
        raise ValueError("An explicit external-O-only empirical model is required")
    if model.get("prediction_source") != prediction_source:
        raise ValueError("Current image predictor differs from CAL source")
    body = {k: v for k, v in model.items() if k != "model_sha256"}
    if model.get("model_sha256") != _digest(body):
        raise ValueError("External-O model binding changed")


def outside_halfplane(side, image_hw):
    height, width = _hw(image_hw)
    # Keypoint coordinates are continuous. Use the closed halfplane containing
    # every truly outside point, including -0.3px; do not manufacture a 1px gap.
    rules = {"left": [1., 0., 0.], "right": [-1., 0., -float(width)],
             "top": [0., 1., 0.], "bottom": [0., -1., -float(height)]}
    if side not in rules:
        raise ValueError("A specific outside side is required")
    return rules[side]


def _canvas_halfplanes(image_hw):
    height, width = _hw(image_hw)
    return [[-1., 0., 0.], [1., 0., float(width)],
            [0., -1., 0.], [0., 1., float(height)]]


def _crop_validity(crop_transform, image, camera, native_corner):
    if crop_transform is None:
        return []
    if crop_transform.get("schema") != "REAL_RGB_TRANSLATION_CROP_V2":
        raise ValueError("Known actual translation-only RGB crop transform required")
    if tuple(crop_transform["canvas_hw"]) != tuple(image.shape[:2]) or crop_transform["crop_image_sha256"] != image_pixel_sha256(image):
        raise ValueError("Real crop transform is not bound to current RGB canvas")
    if crop_transform["crop_K_sha256"] != _array_digest(camera):
        raise ValueError("Real crop transform is not bound to current K")
    source_h, source_w = _hw(crop_transform["source_hw"])
    rect = np.asarray(crop_transform["crop_xyxy"])
    if rect.shape != (4,) or not np.isfinite(rect).all() or np.any(rect != np.floor(rect)):
        raise ValueError("Invalid real crop rectangle")
    x0, y0, x1, y1 = map(int, rect)
    if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h) or image.shape[:2] != (y1-y0, x1-x0):
        raise ValueError("Real crop rectangle and source/current canvases disagree")
    expected = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
    supplied = np.asarray(crop_transform["source_to_crop"], float)
    if supplied.shape != (3, 3) or not np.array_equal(supplied, expected):
        raise ValueError("Crop coordinate transform does not match actual rectangle")
    source_camera = np.asarray(crop_transform.get("source_K", np.linalg.inv(expected) @ camera), float)
    if (source_camera.shape != (3, 3) or crop_transform["source_K_sha256"] != _array_digest(source_camera)
            or not np.allclose(expected @ source_camera, camera, atol=1e-10, rtol=1e-12)):
        raise ValueError("Source K and crop coordinate transform binding changed")
    if native_corner not in crop_transform.get("source_inframe_native_ids", []):
        return []
    return [[-1., 0., float(x0)], [1., 0., float(source_w-x0)],
            [0., -1., float(y0)], [0., 1., float(source_h-y0)]]


def build_ot_domains(image_bgr, raw_points, states, K, native_xyz, model=None, *,
                     prediction_source="raw_backbone", record_id="current", self_hidden=None,
                     truncation_sides=None, crop_transform=None):
    """Build O/T-only domains at original independent image-prediction queries.

    A supplied T side must be known from truncation/crop metadata, rather than
    inferred from a candidate pose. With no known side the outside-canvas union
    is represented by ``outside_canvas_side='unknown'``. Missing O predictions
    remain broad image-canvas boxes. No missing T location is fabricated.
    """
    effective = canonical_states(states, self_hidden)
    ids = [k for k, state in enumerate(effective) if state in ("O", "T")]
    image = np.asarray(image_bgr)
    image_hash = image_pixel_sha256(image)
    q = _points(raw_points)
    camera = np.asarray(K, float)
    xyz = np.asarray(native_xyz, float)
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or min(camera[0, 0], camera[1, 1]) <= 0:
        raise ValueError("K must be finite[3,3] with positive focal lengths")
    if xyz.shape != (8, 3) or not np.isfinite(xyz[ids]).all():
        raise ValueError("native_xyz must be[8,3] with finite eligible O/T geometry")
    _validate_model(model, prediction_source)
    scale = nonself_prediction_scale(q, effective)
    height, width = image.shape[:2]
    bindings = dict(record_id=str(record_id), image_sha256=image_hash, image_hw=[height, width],
                    K_sha256=_array_digest(camera), native_ids=ids,
                    native_xyz_sha256=_array_digest(xyz[ids]),
                    original_queries_sha256=_array_digest(q[ids]),
                    prediction_source=str(prediction_source), states=effective,
                    calibration_model_sha256=None if model is None else model["model_sha256"])
    domains = []
    for k in ids:
        state = effective[k]
        common = dict(native_corner=k, native_id=k, state=state, canvas_hw=[height, width],
                      binding=deepcopy(bindings), independent_precise_support=False,
                      candidate_pose_inputs_used=False, GT_coordinates_used=False,
                      image_source_statistically_independent_of_visible_source=False)
        if state == "O":
            localizable = (_valid_xy(q[k]) and 0 <= q[k, 0] < width and 0 <= q[k, 1] < height and model is not None)
            if localizable:
                radius = max(float(model["minimum_radius_px"]), scale*float(model["normalized_radius"]))
                domain = dict(common, shape="ellipse", center_xy=q[k].tolist(),
                              half_axes_px=[radius, radius], angle_deg=0.,
                              halfplanes=_canvas_halfplanes((height, width)), localization_available=True,
                              source="current_RGB_raw_image_predictor_plus_actual_external_O_CAL_uncertainty")
            else:
                domain = dict(common, shape="box", bounds_xyxy=[0., 0., float(width), float(height)],
                              halfplanes=[], localization_available=False,
                              source="known_inframe_external_O_broad_canvas",
                              unavailable_reason="missing_or_outside_image_prediction_or_missing_CAL_model")
        else:
            side = str((truncation_sides or {}).get(k, (truncation_sides or {}).get(str(k), "unknown"))).lower()
            if side not in OUTSIDE_SIDES:
                raise ValueError("truncation side must be left/right/top/bottom/unknown")
            valid_halfplanes = _crop_validity(crop_transform, image, camera, k)
            planes = ([] if side == "unknown" else [outside_halfplane(side, (height, width))]) + valid_halfplanes
            domain = dict(common, shape="halfplane", outside_canvas_side=side, outside_margin_px=0., halfplanes=planes,
                          localization_available=False, source="known_outside_frame_condition",
                          source_crop_validity_intersection=bool(valid_halfplanes),
                          coordinate_predictor_available=False, uses_O_model=False,
                          independent_image_localization_evidence=False,
                          note="Broad geometric condition; no finite T location or confidence guarantee")
        domains.append(domain)
    return dict(schema=DOMAIN_SCHEMA, domains=domains, domains_sha256=_digest(domains),
                binding=bindings, effective_states=effective,
                excluded_S_ids=[k for k, state in enumerate(effective) if state == "S"],
                nonself_prediction_scale_px=scale, precise_image_anchor_count=0,
                S_image_features_queried=False, S_native_geometry_queried=False,
                GT_coordinates_used=False, candidate_pose_inputs_used=False,
                calibrated_joint_coverage=False, confidence_guarantee=None)


def validate_domain_bindings(bundle, image_bgr, raw_points, states, K, native_xyz, *,
                             prediction_source="raw_backbone", record_id="current", self_hidden=None):
    """Reject replay against changed current image/query/K/native correspondence."""
    if bundle.get("schema") != DOMAIN_SCHEMA:
        raise ValueError("Unsupported OT domain schema")
    if bundle.get("domains_sha256") != _digest(bundle["domains"]):
        raise ValueError("OT domain geometry/provenance binding changed")
    effective = canonical_states(states, self_hidden)
    ids = [k for k, state in enumerate(effective) if state in ("O", "T")]
    q, xyz = _points(raw_points), np.asarray(native_xyz, float)
    expected = dict(record_id=str(record_id), image_sha256=image_pixel_sha256(image_bgr),
                    image_hw=list(np.asarray(image_bgr).shape[:2]), K_sha256=_array_digest(K),
                    native_ids=ids, native_xyz_sha256=_array_digest(xyz[ids]),
                    original_queries_sha256=_array_digest(q[ids]),
                    prediction_source=str(prediction_source), states=effective)
    for key, value in expected.items():
        if bundle["binding"].get(key) != value:
            raise ValueError("OT domain binding changed: "+key)
    for domain in bundle["domains"]:
        k = domain["native_corner"]
        if k not in ids or domain["state"] != effective[k] or domain.get("binding") != bundle["binding"]:
            raise ValueError("OT domain native/state binding changed")
    return True


def apply_real_crop(image_bgr, K, raw_points, crop_xyxy, *, source_inframe_native_ids=()):
    """Actual RGB slice plus exact translation of K and raw native queries.

    No resize, reordering, object-dimension change, or GT point manufacture is
    performed. Source-inframe IDs must be established by source image evidence.
    Their source-canvas validity can subsequently restrict a T halfplane.
    """
    image = np.asarray(image_bgr)
    source_hash = image_pixel_sha256(image)
    source_h, source_w = image.shape[:2]
    rect = np.asarray(crop_xyxy)
    if rect.shape != (4,) or not np.isfinite(rect).all() or np.any(rect != np.floor(rect)):
        raise ValueError("crop_xyxy must be four finite integers")
    x0, y0, x1, y1 = map(int, rect)
    if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h):
        raise ValueError("Real crop must lie within the source RGB canvas")
    known = sorted(set(map(int, source_inframe_native_ids)))
    if any(k not in range(8) for k in known):
        raise ValueError("source_inframe_native_ids must be native corners")
    camera, q = np.asarray(K, float), _points(raw_points).copy()
    source_queries_sha = _array_digest(q)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("K must be finite[3,3]")
    translation = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
    valid = np.isfinite(q).all(1) & ~np.all(q == -1, axis=1)
    q[valid] -= np.array([x0, y0])
    crop = image[y0:y1, x0:x1].copy()
    transformed_K = translation @ camera
    transform = dict(schema="REAL_RGB_TRANSLATION_CROP_V2", crop_xyxy=[x0, y0, x1, y1],
                     source_hw=[source_h, source_w], canvas_hw=list(crop.shape[:2]),
                     source_to_crop=translation.tolist(), crop_to_source=np.linalg.inv(translation).tolist(),
                     source_image_sha256=source_hash, crop_image_sha256=image_pixel_sha256(crop),
                     source_K_sha256=_array_digest(camera), crop_K_sha256=_array_digest(transformed_K),
                     source_K=camera.tolist(),
                     source_queries_sha256=source_queries_sha, crop_queries_sha256=_array_digest(q),
                     source_inframe_native_ids=known, raw_queries_transformed=True,
                     GT_coordinates_used=False, dimensions_changed=False, native_order_changed=False,
                     resized=False, actual_RGB_crop=True)
    return dict(image_bgr=crop, K=transformed_K, raw_points=q, transform=transform)


def inventory_human_state_metadata(records):
    """Read only state metadata for CAL/TEST inventory; never inspect targets.

    ``records`` is a mapping fid -> metadata or fid -> {'metadata': metadata}.
    This supports isolated scorer containers without accessing their targets.
    Severity is inherited from the already isolated nonsevere cohort if absent.
    """
    counts, by_split, excluded = Counter(), {}, []
    for fid, container in records.items():
        metadata = container.get("metadata", container)
        split = str(metadata.get("partition", metadata.get("split", ""))).upper()
        severity = str(metadata.get("severity", "NONSEVERE_COHORT")).upper()
        if split not in ("CAL", "TEST") or severity not in ("CLEAN", "MODERATE", "NONSEVERE_COHORT"):
            excluded.append(str(fid))
            continue
        states = canonical_states(metadata["human_states_native"])
        ids = {state: [k for k, value in enumerate(states) if value == state] for state in "VOTSU"}
        counts.update(states)
        row = dict(record_id=str(fid), session=str(metadata.get("session", "unknown")),
                   split=split, severity=severity, human_states=states,
                   O_ids=ids["O"], T_ids=ids["T"], S_ids=ids["S"], joint_external_O=len(ids["O"]) >= 2)
        by_split.setdefault(split, []).append(row)
    return dict(schema="OT_STATE_METADATA_INVENTORY_V2", state_counts=dict(counts), by_split=by_split,
                joint_external_O_records={split: [r["record_id"] for r in rows if r["joint_external_O"]]
                                          for split, rows in by_split.items()},
                excluded_record_ids=excluded, coordinate_targets_accessed=False,
                state_inventory_does_not_create_image_measurements=True)
