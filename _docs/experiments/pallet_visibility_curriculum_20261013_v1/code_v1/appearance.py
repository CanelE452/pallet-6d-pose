"""Conservative, training-free RGB evidence for predicted cuboid keypoints.

Only the current BGR image, predicted points, and supplied geometric
self-occlusion flags are consumed. A material palette is estimated from
eroded projected face interiors and contrasted with a surrounding ring.
These are appearance *evidence* scores, not calibrated probabilities or a
proof that an observed pixel belongs to the requested 3D point. In particular,
same-coloured occluders and virtual bounding-box corners remain ambiguous.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
DEFAULT_CONFIG = {
    "patch_radius_fraction": .025,
    "patch_radius_min": 6,
    "patch_radius_max": 24,
    "face_shrink": .65,
    "minimum_palette_samples": 32,
    "maximum_palette_samples": 12000,
    "palette_modes": 3,
    "minimum_material_fraction": .06,
    "minimum_ring_contrast": .12,
    "lab_tolerance": [48., 20., 20.],
    "palette_match_radius": 1.0,
    "visible_support_threshold": .55,
    "boundary_threshold": .45,
    "suspected_occlusion_threshold": .25,
}


def _config(config: dict | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown appearance config fields: {sorted(unknown)}")
        cfg.update(config)
    tolerance = np.asarray(cfg["lab_tolerance"], float)
    if tolerance.shape != (3,) or not np.isfinite(tolerance).all() or (tolerance <= 0).any():
        raise ValueError("lab_tolerance must contain three positive finite numbers")
    if not (0 < float(cfg["face_shrink"]) < 1):
        raise ValueError("face_shrink must lie strictly between zero and one")
    if int(cfg["patch_radius_min"]) < 2 or int(cfg["patch_radius_max"]) < int(cfg["patch_radius_min"]):
        raise ValueError("Invalid patch radius range")
    if float(cfg["patch_radius_fraction"]) <= 0:
        raise ValueError("patch_radius_fraction must be positive")
    if int(cfg["minimum_palette_samples"]) < 1 or int(cfg["maximum_palette_samples"]) < int(cfg["minimum_palette_samples"]):
        raise ValueError("Invalid palette sample range")
    if not 1 <= int(cfg["palette_modes"]) <= 8:
        raise ValueError("palette_modes must be between one and eight")
    return cfg


def _inputs(image_bgr, points, self_hidden):
    image = np.asarray(image_bgr)
    q = np.asarray(points, dtype=np.float64)
    hidden = np.asarray(self_hidden, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError("Expected a nonempty uint8 BGR image [H,W,3]")
    if q.shape not in ((8, 2), (9, 2)) or hidden.shape != (8,):
        raise ValueError("Expected points [8,2] or [9,2] and self_hidden [8]")
    q = q[:8].copy()
    valid = np.isfinite(q).all(axis=1) & ~np.all(q == -1, axis=1)
    h, w = image.shape[:2]
    inside = valid & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    return image, q, hidden, valid, inside


def _fill_polygon(mask, polygon):
    # Bound extreme off-screen predictions before integer conversion. Polygon
    # clipping is ultimately performed by OpenCV against the image canvas.
    extent = max(mask.shape) * 4
    p = np.rint(np.clip(polygon, -extent, extent)).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(p), 255)


def _sample(values, mask, maximum):
    chosen = values[mask > 0]
    if len(chosen) > maximum:
        # Fixed stride sampling keeps results deterministic and avoids RNG state.
        chosen = chosen[np.linspace(0, len(chosen) - 1, maximum, dtype=np.int64)]
    return chosen.astype(np.float32)


def _match(lab, centers, tolerance, match_radius):
    if not centers:
        return np.zeros(lab.shape[:2], dtype=np.uint8)
    matched = np.zeros(lab.shape[:2], dtype=bool)
    for center in centers:
        d = (lab.astype(np.float32) - center) / tolerance
        matched |= np.einsum("...i,...i->...", d, d) <= match_radius ** 2
    return matched.astype(np.uint8) * 255


def _palette(lab, seed, ring, cfg):
    tolerance = np.asarray(cfg["lab_tolerance"], np.float32)
    inside = _sample(lab, seed, int(cfg["maximum_palette_samples"]))
    outside = _sample(lab, ring, int(cfg["maximum_palette_samples"]))
    diag: dict[str, Any] = {
        "seed_samples": int(len(inside)), "ring_samples": int(len(outside)),
        "centers_lab": [], "mode_support": [], "confidence": 0.,
        "status": "insufficient_palette_samples",
    }
    if len(inside) < int(cfg["minimum_palette_samples"]):
        return [], diag
    # Dominant bins propose modes, robust medians represent each mode. Similar
    # bins are combined by evaluating their entire tolerance neighbourhood.
    bins = np.floor(inside / 16).astype(np.int32)
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:24]
    centers, supports = [], []
    for index in order:
        candidate = np.median(inside[keys == unique[index]], axis=0)
        if any(np.linalg.norm((candidate - old) / tolerance) < 1. for old in centers):
            continue
        distance = np.linalg.norm((inside - candidate) / tolerance, axis=1)
        neighbours = distance <= float(cfg["palette_match_radius"])
        if not neighbours.any():
            continue
        candidate = np.median(inside[neighbours], axis=0)
        within = np.linalg.norm((inside - candidate) / tolerance, axis=1) <= float(cfg["palette_match_radius"])
        support = float(np.mean(within))
        if support < float(cfg["minimum_material_fraction"]):
            continue
        external = float(np.mean(np.linalg.norm((outside - candidate) / tolerance, axis=1)
                                 <= float(cfg["palette_match_radius"]))) if len(outside) else None
        # Background visible through cutouts should not become a material mode.
        # Without a usable surrounding ring we retain a provisional low-confidence
        # mode rather than claiming foreground/background identity.
        contrast = support - external if external is not None else 0.
        if external is not None and contrast < float(cfg["minimum_ring_contrast"]):
            continue
        centers.append(candidate)
        supports.append({"interior_fraction": support, "ring_fraction": external,
                         "contrast": contrast})
        if len(centers) >= int(cfg["palette_modes"]):
            break
    if not centers:
        diag["status"] = "no_material_background_separation"
        return [], diag
    best_contrast = max(s["contrast"] for s in supports)
    confidence = float(np.clip(.45 + .55 * best_contrast, .45, 1.)) if len(outside) else .35
    diag.update(centers_lab=[c.astype(float).tolist() for c in centers],
                mode_support=supports, confidence=confidence,
                status="contrast_supported" if len(outside) else "no_background_reference")
    return centers, diag


def _scene(image, q, hidden, valid, cfg):
    h, w = image.shape[:2]
    hull = np.zeros((h, w), np.uint8)
    faces = np.zeros_like(hull)
    finite = q[valid]
    extent = float(np.hypot(*(np.ptp(finite, axis=0)))) if len(finite) else 0.
    radius = int(np.clip(round(extent * float(cfg["patch_radius_fraction"])),
                         cfg["patch_radius_min"], cfg["patch_radius_max"]))
    if len(finite) >= 3:
        _fill_polygon(hull, finite)
    face_count = 0
    for ids in FACES:
        ids = np.asarray(ids)
        # With vertex self visibility alone, all-visible corners are sufficient
        # evidence to seed a face. Three visible corners cannot prove that its
        # fourth, hidden corner's face is front-facing.
        if not valid[ids].all() or hidden[ids].any():
            continue
        polygon = q[ids]
        if abs(cv2.contourArea(polygon.astype(np.float32))) < 32:
            continue
        center = np.mean(polygon, axis=0)
        _fill_polygon(faces, center + float(cfg["face_shrink"]) * (polygon - center))
        face_count += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    eroded = cv2.erode(hull, kernel)
    seed = cv2.bitwise_and(faces, eroded) if face_count else eroded
    if np.count_nonzero(seed) < int(cfg["minimum_palette_samples"]):
        seed = eroded
    ring = cv2.subtract(cv2.dilate(hull, kernel), hull)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    centers, palette = _palette(lab, seed, ring, cfg)
    material = _match(lab, centers, np.asarray(cfg["lab_tolerance"], np.float32),
                      float(cfg["palette_match_radius"]))
    # A colour boundary is a local cue only: texture and other objects can
    # generate equally strong edges. Distance provides tolerance to xy error.
    edge = cv2.morphologyEx(material, cv2.MORPH_GRADIENT,
                          np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3) if edge.any() else None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    return {"hull": hull, "material": material, "edge": edge,
            "distance": distance, "gx": gx, "gy": gy,
            "radius": radius, "palette": palette, "face_seed_count": face_count}


def _mean(binary, selection):
    return float(np.mean(binary[selection])) if np.any(selection) else None


def _local(q, scene, cfg):
    material = scene["material"] > 0
    h, w = material.shape
    r = scene["radius"]
    x, y = q
    x0, x1 = max(0, int(math.floor(x - 2 * r))), min(w, int(math.ceil(x + 2 * r)) + 1)
    y0, y1 = max(0, int(math.floor(y - 2 * r))), min(h, int(math.ceil(y + 2 * r)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    disk, neighbourhood = d2 <= r ** 2, d2 <= (2 * r) ** 2
    expected = scene["hull"][y0:y1, x0:x1] > 0
    match = material[y0:y1, x0:x1]
    inner = disk & expected
    outer = disk & ~expected
    inward_occupancy = _mean(match, inner)
    whole_occupancy = _mean(match, disk)
    outward_occupancy = _mean(match, outer)
    nearby_occupancy = _mean(match, neighbourhood & expected)
    if inward_occupancy is None:
        inward_occupancy = whole_occupancy or 0.
    if nearby_occupancy is None:
        nearby_occupancy = inward_occupancy
    ix, iy = min(w - 1, int(round(x))), min(h - 1, int(round(y)))
    edge_distance = float(scene["distance"][iy, ix]) if scene["distance"] is not None else None
    edge_proximity = math.exp(-edge_distance / max(2., .3 * r)) if edge_distance is not None else 0.
    gx, gy = scene["gx"][y0:y1, x0:x1][disk], scene["gy"][y0:y1, x0:x1][disk]
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)],
                       [np.mean(gx * gy), np.mean(gy * gy)]])
    eigenvalues = np.linalg.eigvalsh(tensor)
    corner = float(np.clip(2 * eigenvalues[0] / max(25., eigenvalues.sum()), 0., 1.))
    edge_energy = float(np.clip(math.sqrt(max(0., float(eigenvalues.sum()))) / 20., 0., 1.))
    if outward_occupancy is not None:
        separation = max(0., inward_occupancy - outward_occupancy)
    else:
        # A hole background can be an actual local boundary inside the hull.
        separation = float(np.clip(4 * whole_occupancy * (1 - whole_occupancy), 0., 1.))
    boundary = float(np.clip(edge_proximity * (.3 + .7 * separation), 0., 1.))
    confidence = float(scene["palette"]["confidence"])
    visibility = float(np.clip((.12 + .78 * inward_occupancy) * confidence
                               + .10 * min(inward_occupancy, nearby_occupancy), 0., 1.))
    # Interior material is weak evidence for an exact semantic corner. Both
    # local edge distance and two-direction corner structure improve support.
    accuracy = float(np.clip(confidence * inward_occupancy
                            * (.18 + .48 * edge_proximity + .34 * corner * edge_energy), 0., 1.))
    row = {"inward_material_fraction": inward_occupancy,
           "outward_material_fraction": outward_occupancy,
           "patch_material_fraction": whole_occupancy,
           "nearby_inward_material_fraction": nearby_occupancy,
           "edge_distance_px": edge_distance, "corner_structure": corner,
           "edge_energy": edge_energy, "patch_samples": int(disk.sum()),
           "expected_inward_samples": int(inner.sum())}
    if confidence < .4:
        state = "UNKNOWN"
    elif inward_occupancy < float(cfg["suspected_occlusion_threshold"]) and nearby_occupancy >= .4:
        state = "OCCLUSION_SUSPECTED"
    elif inward_occupancy < float(cfg["visible_support_threshold"]):
        state = "UNKNOWN"
    elif boundary >= float(cfg["boundary_threshold"]):
        state = "VISIBLE_BOUNDARY"
    elif inward_occupancy >= .7 and whole_occupancy >= .7:
        state = "VISIBLE_INTERIOR"
    else:
        state = "UNKNOWN"
    return state, visibility, boundary, accuracy, row


def infer_appearance(image_bgr, points, self_hidden, config=None) -> dict:
    """Return JSON-serializable eight-corner states and continuous RGB evidence.

    OUT_OF_FRAME precedes SELF_OCCLUDED; invalid/missing coordinates remain
    UNKNOWN. An external colour mismatch is not labelled a proven occlusion.
    No coordinates are changed and no geometrically reprojected points are
    produced. ``local_accuracy_score`` is image support, not measured xy error.
    """
    cfg = _config(config)
    image, q, hidden, valid, inside = _inputs(image_bgr, points, self_hidden)
    scene = _scene(image, q, hidden, valid, cfg)
    states, visibility, boundary, accuracy, rows = [], [], [], [], []
    for k in range(8):
        if not valid[k]:
            state, vis, bound, acc, row = "UNKNOWN", 0., 0., 0., {"reason": "missing_or_nonfinite_prediction"}
        elif not inside[k]:
            state, vis, bound, acc, row = "OUT_OF_FRAME", 0., 0., 0., {"reason": "predicted_coordinate_outside_frame"}
        elif hidden[k]:
            state, vis, bound, acc, row = "SELF_OCCLUDED", 0., 0., 0., {"reason": "supplied_initial_pose_self_occlusion"}
        else:
            state, vis, bound, acc, row = _local(q[k], scene, cfg)
            row["reason"] = "rgb_evidence_only"
        states.append(state)
        visibility.append(vis)
        boundary.append(bound)
        accuracy.append(acc)
        rows.append({"corner": k, "state": state, **row})
    return {"states": states, "visibility_score": visibility,
            "boundary_score": boundary, "local_accuracy_score": accuracy,
            "diagnostics": {"palette": scene["palette"], "corners": rows,
                            "patch_radius_px": scene["radius"],
                            "face_seed_count": scene["face_seed_count"],
                            "config": cfg,
                            "inference_inputs": ["current_image_bgr", "predicted_points", "supplied_self_hidden"],
                            "ground_truth_inputs": False,
                            "scores_calibrated": False,
                            "limits": ["same_coloured_occluders", "incorrect_initial_pose_or_xy",
                                       "virtual_bbox_corners", "palette_contamination", "material_background_similarity"]}}


def build_appearance_mask(image_bgr, points, self_hidden, config=None) -> np.ndarray:
    """Return the uint8 (0/255) material-colour mask for optional visualization.

    This is deliberately called a material mask rather than a target instance
    segmentation: matching RGB cannot establish object instance identity.
    """
    cfg = _config(config)
    image, q, hidden, valid, _ = _inputs(image_bgr, points, self_hidden)
    return _scene(image, q, hidden, valid, cfg)["material"]
