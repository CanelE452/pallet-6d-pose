"""Fuse a frozen RGB CNN mask with the existing local appearance features.

The learned mask is independent image evidence, not ground truth. It is
limited to a dilation of the predicted hull, and cannot establish visibility
of a specific 3-D corner or same-coloured occluder ownership. Raw Lab and
soft-colour features retain their original meanings for CAL classifiers.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from . import appearance as A


def _probability(value, hw=None):
    probability = np.asarray(value, dtype=np.float32)
    if probability.ndim != 2 or min(probability.shape) < 1:
        raise ValueError("Expected nonempty CNN foreground probability [H,W]")
    if hw is not None and probability.shape != tuple(hw):
        raise ValueError("CNN mask must already have the original image shape")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("CNN foreground scores must be finite and in[0,1]")
    return probability


def _hull_roi(probability, points, config=None):
    cfg = A._config(config)
    q = np.asarray(points, float)
    if q.shape not in ((8, 2), (9, 2)):
        raise ValueError("Expected points[8,2] or points[9,2]")
    q = q[:8]
    valid = np.isfinite(q).all(1) & ~(q == -1).all(1)
    finite = q[valid]
    extent = float(np.hypot(*(np.ptp(finite, axis=0)))) if len(finite) else 0.
    radius = int(np.clip(round(extent * cfg["patch_radius_fraction"]),
                         cfg["patch_radius_min"], cfg["patch_radius_max"]))
    hull = np.zeros(probability.shape, np.uint8)
    if len(finite) >= 3:
        A._fill_polygon(hull, finite)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    roi = cv2.dilate(hull, kernel)
    return hull, roi, radius


def mask_from_probability(probability_mask, points, config=None):
    """Return uint8 0/255 CNN>=.5 within dilated predicted hull; no GT input."""
    probability = _probability(probability_mask)
    _, roi, _ = _hull_roi(probability, points, config)
    return ((probability >= .5) & (roi > 0)).astype(np.uint8) * 255


def _patch_probability(point, scene, probability, roi):
    h, w = probability.shape
    x, y = point
    radius = scene["radius"]
    x0, x1 = max(0, int(math.floor(x - 2 * radius))), min(w, int(math.ceil(x + 2 * radius)) + 1)
    y0, y1 = max(0, int(math.floor(y - 2 * radius))), min(h, int(math.ceil(y + 2 * radius)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    disk, nearby = d2 <= radius ** 2, d2 <= (2 * radius) ** 2
    hull = scene["hull"][y0:y1, x0:x1] > 0
    raw = probability[y0:y1, x0:x1]
    restricted = raw * (roi[y0:y1, x0:x1] > 0)
    inner, outer = disk & hull, disk & ~hull

    def mean(values, selected):
        return float(np.mean(values[selected])) if selected.any() else None

    return dict(segmentation_probability_patch_mean=mean(raw, disk),
                segmentation_probability_inward_mean=mean(raw, inner),
                segmentation_probability_outward_mean=mean(raw, outer),
                segmentation_probability_nearby_inward_mean=mean(raw, nearby & hull),
                segmentation_effective_probability_patch_mean=mean(restricted, disk),
                segmentation_effective_probability_inward_mean=mean(restricted, inner),
                segmentation_effective_probability_outward_mean=mean(restricted, outer),
                segmentation_effective_probability_nearby_inward_mean=mean(restricted, nearby & hull))


def infer_segmented_appearance(image_bgr, points, self_hidden, probability_mask, config=None):
    """Return the original eight-corner JSON contract plus frozen-CNN evidence.

    Missing/out-of-frame/self-hidden priorities remain unchanged. The raw RGB
    palette is retained in diagnostics, but its absence does not disable the
    separately supplied CNN mask. The CNN>=.5 threshold is fixed, not tuned.
    """
    cfg = A._config(config)
    image, q, hidden, valid, inside = A._inputs(image_bgr, points, self_hidden)
    probability = _probability(probability_mask, image.shape[:2])
    scene = A._scene(image, q, hidden, valid, cfg)
    raw_palette = scene["palette"]
    raw_material = scene["material"]
    hull, roi, radius = _hull_roi(probability, q, cfg)
    if radius != scene["radius"] or not np.array_equal(hull, scene["hull"]):
        raise AssertionError("CNN mask ROI and original appearance geometry differ")
    material = ((probability >= .5) & (roi > 0)).astype(np.uint8) * 255
    scene["material"] = material
    scene["edge"] = cv2.morphologyEx(material, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    scene["distance"] = cv2.distanceTransform((scene["edge"] == 0).astype(np.uint8), cv2.DIST_L2, 3) if scene["edge"].any() else None
    # _local's legacy palette gate is a colour-presence gate. CNN evidence must
    # not disappear merely because the background and pallet share that colour.
    # This placeholder is used only for that gate; no Lab colour is fabricated
    # in returned features or in the original raw palette diagnostics.
    scene["palette"] = dict(raw_palette, confidence=1., centers_lab=raw_palette["centers_lab"] or [[0., 0., 0.]])
    states, visibility, boundary, accuracy, corners = [], [], [], [], []
    for k in range(8):
        if not valid[k]:
            state, vis, bound, acc, row = "UNKNOWN", 0., 0., 0., dict(reason="missing_or_nonfinite_prediction")
        elif not inside[k]:
            state, vis, bound, acc, row = "OUT_OF_FRAME", 0., 0., 0., dict(reason="predicted_coordinate_outside_frame")
        elif hidden[k]:
            state, vis, bound, acc, row = "SELF_OCCLUDED", 0., 0., 0., dict(reason="supplied_initial_pose_self_occlusion")
        else:
            state, vis, bound, acc, row = A._local(q[k], scene, cfg)
            row["reason"] = "frozen_source_CNN_mask_and_raw_RGB_features"
        if valid[k] and inside[k]:
            row.update(_patch_probability(q[k], scene, probability, roi))
        row["binary_material_feature_source"] = "frozen_source_RGB_CNN_with_predicted_hull_ROI"
        row["soft_material_feature_source"] = "original_RGB_Lab_palette_support"
        row["appearance_version"] = "v3_source_CNN_mask_raw_v2_Lab_features"
        states.append(state)
        visibility.append(vis)
        boundary.append(bound)
        accuracy.append(acc)
        corners.append(dict(corner=k, state=state, **row))
    return dict(states=states, visibility_score=visibility, boundary_score=boundary,
        local_accuracy_score=accuracy, diagnostics=dict(palette=raw_palette,
            corners=corners, patch_radius_px=radius, face_seed_count=scene["face_seed_count"], config=cfg,
            mask_source="frozen_source_RGB_CNN", mask_threshold=.5,
            mask_ROI="predicted_cuboid_hull_dilated_by_patch_radius",
            mask_ROI_pixels=int(np.count_nonzero(roi)), mask_pixels=int(np.count_nonzero(material)),
            original_colour_material_pixels=int(np.count_nonzero(raw_material)),
            CNN_colour_material_intersection_pixels=int(np.count_nonzero((raw_material > 0) & (material > 0))),
            effective_palette_gate=dict(raw_colour_gate_applied_to_CNN=False,
                                        calibrated_CNN_confidence_claimed=False),
            inference_inputs=["current_image_bgr", "predicted_points", "supplied_self_hidden", "frozen_CNN_probability_mask"],
            ground_truth_inputs=False, scores_calibrated=False,
            limits=["source_graphite_single_shape_domain", "CNN_background_false_positives",
                    "incorrect_initial_pose_or_xy", "virtual_bbox_corners", "same_coloured_occluder_ownership",
                    "a_mask_support_pixel_is_not_proof_of_the_requested_3D_corner"]))
