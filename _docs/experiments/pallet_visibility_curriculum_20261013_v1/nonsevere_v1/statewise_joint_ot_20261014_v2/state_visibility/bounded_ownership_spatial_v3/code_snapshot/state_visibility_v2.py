"""Actual DIRECT/EXTERNAL visibility with a pre-RGB geometric S gate.

Initial boundary/interior roles are geometry inputs, not visibility labels.
This module never changes coordinates and never uses observation usability.
T from an initial projection is explicitly conditional; an OOB detector query
alone is not evidence of physical truncation. U is an internal abstention.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .joint_observation import JointObservationModel, crop_context, spatial_features


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "STATE_VISIBILITY_DIRECT_EXTERNAL_V2"
LOCAL_FEATURE_NAMES = (
    ["lab_mean_" + c for c in "Lab"]
    + ["lab_std_" + c for c in "Lab"]
    + ["lab_center_minus_context_" + c for c in "Lab"]
    + ["gradient_mean", "gradient_std", "gradient_q90", "corner_eigen_ratio"]
    + ["gradient_orientation_bin_" + str(i) for i in range(8)]
    + ["mask_center_mean", "mask_context_mean", "mask_context_std"]
    + ["mask_grid_" + str(i) for i in range(16)]
)
RGB26_SCALARS = ("inward_material_fraction", "outward_material_fraction", "patch_material_fraction",
    "nearby_inward_material_fraction", "soft_inward_material_support", "soft_nearby_inward_material_support",
    "corner_structure", "edge_energy", "patch_chroma_mean", "patch_chroma_std")
RGB26_VECTORS = ("patch_lab_mean", "patch_lab_median", "patch_lab_std", "inward_lab_mean", "inward_lab_std")
RGB26_FEATURE_NAMES = list(RGB26_SCALARS) + [key + "_" + channel for key in RGB26_VECTORS for channel in "Lab"] + ["edge_distance_over_patch_radius"]
OWNERSHIP_FEATURE_NAMES = [source + "_" + name for source in ("raw", "initial_projected") for name in (
    "center_palette_distance", "center_chroma", "inward_soft_1", "inward_soft_2",
    "outward_soft_1", "outward_soft_2", "near_edge_normal_energy", "far_edge_normal_energy",
    "near_edge_tangent_energy", "far_edge_tangent_energy", "wedge_soft_mean", "wedge_missing_fraction")]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class StateVisibilityModel:
    descriptor: dict
    arrays: dict
    frozen_backbone: JointObservationModel


@lru_cache(maxsize=4)
def load_state_model(path, device="cpu"):
    path = Path(path).resolve()
    descriptor = json.loads(path.read_text())
    if descriptor.get("schema") != SCHEMA:
        raise ValueError("Unsupported actual-visibility descriptor")
    array_path = path.parent / descriptor["arrays_file"]
    if sha256(array_path) != descriptor["arrays_sha256"]:
        raise ValueError("State classifier parameter binding changed")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    for array in arrays.values():
        array.setflags(write=False)
    # Reuse only the frozen network loader; no old learned class/regression
    # parameters or old appearance function participate in state inference.
    network = None
    if descriptor.get("feature_family") != "NON_S_RGB26" or descriptor.get("backbone"):
        backbone = descriptor["backbone"]
        weight_path = ROOT / backbone["weights_repo_relative"]
        if sha256(weight_path) != backbone["weights_sha256"]:
            raise ValueError("Frozen RGB backbone binding changed")
        network = JointObservationModel({"backbone": backbone}, {}, weight_path, device)
    return StateVisibilityModel(descriptor, arrays, network)


def _geometry_inputs(image_bgr, raw_points, initial_self_hidden, initial_roles,
                     initial_projection_xy):
    image = np.asarray(image_bgr)
    q = np.asarray(raw_points, float)
    hidden = np.asarray(initial_self_hidden)
    roles = tuple(str(role).upper() for role in initial_roles)
    projected = np.asarray(initial_projection_xy, float)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image_bgr must be nonempty uint8 BGR[H,W,3]")
    if q.shape not in ((8, 2), (9, 2)) or projected.shape != (8, 2):
        raise ValueError("Expected raw native 8/9 points and initial projection[8,2]")
    if hidden.shape != (8,) or not np.isin(hidden, [False, True]).all():
        raise ValueError("initial_self_hidden must contain eight booleans")
    if len(roles) != 8 or any(roles[k] not in ("B", "I") for k in range(8) if not hidden[k]):
        raise ValueError("Every non-S point needs its initial B/I geometry role")
    return image, q[:8], hidden.astype(bool), roles, projected


def _outside_side(point, hw):
    h, w = hw
    sides = []
    if point[0] < 0:
        sides.append("left")
    if point[0] >= w:
        sides.append("right")
    if point[1] < 0:
        sides.append("top")
    if point[1] >= h:
        sides.append("bottom")
    return sides


def geometry_state_gates(image_bgr, raw_points, initial_self_hidden, initial_roles,
                         initial_projection_xy, *, independent_truncation=None):
    """Decide S/T/U gates before any pixel sampling or classifier invocation.

    Independent T evidence is optional per-native-ID metadata with
    ``independent_of_initial_pose=True``, a ``source`` and nonempty valid
    ``outside_canvas_side``. Passing this contract asserts its provenance;
    metadata alone does not verify the external evidence.
    """
    image, q, hidden, roles, projected = _geometry_inputs(
        image_bgr, raw_points, initial_self_hidden, initial_roles, initial_projection_xy)
    h, w = image.shape[:2]
    records, eligible = [], []
    independent = independent_truncation or {}
    for k in range(8):
        raw_valid = bool(np.isfinite(q[k]).all() and not np.array_equal(q[k], [-1., -1.]))
        raw_oob = bool(_outside_side(q[k], (h, w))) if raw_valid else None
        projection_valid = bool(np.isfinite(projected[k]).all())
        sides = _outside_side(projected[k], (h, w)) if projection_valid else []
        row = dict(native_id=k, state="U", initial_role=None if hidden[k] else roles[k],
                   initial_self_occluded=bool(hidden[k]), rgb_evaluated=False,
                   p_visible=None, raw_query_out_of_frame=raw_oob,
                   conditional_truncation=bool(sides) if projection_valid else None,
                   physical_truncation_certified=False, outside_canvas_side=sides,
                   truncation_source=None, abstained=True, reason=None)
        # S has precedence even when its raw query/projection is OOB. No S
        # projected coordinate is exposed as an inferred keypoint observation.
        if hidden[k]:
            row.update(state="S", abstained=False, reason="fixed_initial_geometric_self_occlusion",
                       conditional_truncation=None, outside_canvas_side=[])
        elif k in independent or str(k) in independent:
            proof = independent.get(k, independent.get(str(k)))
            proof_sides = proof.get("outside_canvas_side", [])
            if isinstance(proof_sides, str):
                proof_sides = [proof_sides]
            if (proof.get("independent_of_initial_pose") is not True or not proof.get("source")
                    or not proof_sides or any(s not in ("left", "right", "top", "bottom") for s in proof_sides)):
                raise ValueError("Independent T requires explicit provenance and outside-canvas sides")
            row.update(state="T", abstained=False, physical_truncation_certified=True,
                       outside_canvas_side=list(proof_sides), truncation_source=proof["source"],
                       reason="independently_supplied_physical_truncation")
        elif not projection_valid:
            row["reason"] = "initial_projection_unavailable"
        elif sides:
            row.update(state="T", abstained=False, truncation_source="initial_pose_projection",
                       reason="conditional_initial_pose_truncation")
        elif not raw_valid or raw_oob:
            row["reason"] = "RGB_query_missing_or_OOB_not_physical_T"
        else:
            eligible.append(k)
            row["reason"] = "awaiting_actual_visibility_classifier"
        records.append(row)
    return records, np.asarray(eligible, int)


def patch_radius(raw_points, eligible_ids):
    """Choose context scale using eligible non-S queries only."""
    ids = np.asarray(eligible_ids, int)
    if len(ids) < 2:
        return 8
    q = np.asarray(raw_points, float)[:8][ids]
    return int(np.clip(round(.04 * np.linalg.norm(np.ptp(q, axis=0))), 5, 24))


def extract_state_features(image_bgr, raw_points, eligible_ids, radius, frozen_backbone,
                           *, pallet_mask):
    """Extract rows only for explicitly supplied already-gated native IDs.

    Mask probabilities are features, never a DIRECT/EXTERNAL label. The frozen
    embedding uses native detector-query patches, not initial projections.
    No keypoint IDs, absolute positions, session IDs or coordinate errors are
    included as learned feature columns.
    """
    image = np.asarray(image_bgr)
    q = np.asarray(raw_points, float)[:8]
    ids = np.asarray(eligible_ids, int)
    mask = np.asarray(pallet_mask, np.float32)
    if (mask.shape != image.shape[:2] or not np.isfinite(mask).all()
            or np.any((mask < 0) | (mask > 1))):
        raise ValueError("pallet_mask must be finite canvas-sized probabilities in [0,1]")
    if len(ids) != len(set(ids.tolist())) or np.any((ids < 0) | (ids >= 8)):
        raise ValueError("eligible_ids must be unique native corner IDs")
    local = []
    for k in ids:
        tile = cv2.resize(crop_context(image, q[k], 4 * radius), (64, 64))
        mask_tile = cv2.getRectSubPix(mask, (4 * radius, 4 * radius), tuple(map(float, q[k])))
        mask_tile = cv2.resize(mask_tile, (64, 64), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB).astype(float) / 255.
        flat = lab.reshape(-1, 3)
        center = lab[28:36, 28:36].mean((0, 1))
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
        magnitude = np.hypot(gx, gy)
        eig = np.linalg.eigvalsh([[np.mean(gx * gx), np.mean(gx * gy)],
                                [np.mean(gx * gy), np.mean(gy * gy)]])
        orientation = np.arctan2(gy, gx)
        hist = np.histogram(orientation, bins=8, range=(-np.pi, np.pi), weights=magnitude)[0]
        hist = hist / max(float(hist.sum()), 1e-8)
        mask_grid = cv2.resize(mask_tile, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)
        values = np.r_[flat.mean(0), flat.std(0), center - flat.mean(0),
                       magnitude.mean(), magnitude.std(), np.quantile(magnitude, .9),
                       2 * eig[0] / max(float(eig.sum()), 1e-8), hist,
                       mask_tile[28:36, 28:36].mean(), mask_tile.mean(), mask_tile.std(), mask_grid]
        local.append(values)
    values = np.asarray(local, float).reshape(len(ids), len(LOCAL_FEATURE_NAMES))
    spatial = spatial_features(image, q, ids, radius, frozen_backbone)
    if spatial.shape != (len(ids), 2560) or not np.isfinite(values).all() or not np.isfinite(spatial).all():
        raise ValueError("Actual visibility features must be finite and schema-matched")
    return values, spatial


def predict_actual_visibility(local_features, frozen_spatial, model):
    """Pure forward of a human DIRECT/EXTERNAL classifier; no P_usable input."""
    a = model.arrays
    family = model.descriptor.get("feature_family", "")
    if family.startswith("NON_S_RGB26"):
        x = np.array(local_features, float, copy=True)
        if model.descriptor.get("rgb26_version") == "legacy_zero_soft":
            x[:, 4:6] = 0.
        if "RAW_SPATIAL" in family:
            spatial = (np.asarray(frozen_spatial, float) - a["spatial_mean"]) / a["spatial_scale"]
            x = np.column_stack([x, spatial])
    else:
        spatial = np.asarray(frozen_spatial, float)
        z = (spatial - a["spatial_mean"]) / a["spatial_scale"]
        compressed = (z - a["pca_mean"]) @ a["pca_components"].T
        x = np.column_stack([np.asarray(local_features, float), compressed])
    class_z = (x - a["class_mean"]) / a["class_scale"]
    logits = class_z @ a["class_coefficients"].reshape(-1) + float(a["class_intercept"].reshape(-1)[0])
    return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))


def extract_nonS_rgb26(image_bgr, raw_points, eligible_ids):
    """RGB26 with noneligible coordinates erased before global scene features.

    Old `_scene/_local` helpers provide pixel descriptors only. Their states,
    visibility score and coordinate-quality score are discarded. Only a new
    human DIRECT/EXTERNAL classifier can turn these descriptors into O.
    S coordinates cannot affect hull, radius, palette or local pixel queries.
    Initial pose projections are never used as query positions.
    """
    from . import appearance as A
    ids = np.asarray(eligible_ids, int)
    if len(ids) != len(set(ids.tolist())) or np.any((ids < 0) | (ids >= 8)):
        raise ValueError("eligible_ids must be unique native corners")
    # Do not pass the original coordinate array to any image helper.
    masked_q = np.full((8, 2), np.nan)
    masked_q[ids] = np.asarray(raw_points, float)[:8][ids]
    valid = np.zeros(8, bool)
    valid[ids] = True
    hidden = ~valid
    cfg = A._config(None)
    scene = A._scene(np.asarray(image_bgr), masked_q, hidden, valid, cfg)
    radius = max(1., float(scene["radius"]))
    features, diagnostics = [], []
    for k in ids:
        _, _, _, _, row = A._local(masked_q[k], scene, cfg)
        values = [float(row.get(key) or 0.) for key in RGB26_SCALARS]
        for key in RGB26_VECTORS:
            value = row.get(key)
            values.extend([0., 0., 0.] if value is None else value)
        values.append(min(4., float(row.get("edge_distance_px") or 0.) / radius))
        features.append(values)
        diagnostics.append(dict(native_id=int(k), **row))
    x = np.asarray(features, float).reshape(len(ids), 26)
    if not np.isfinite(x).all():
        raise ValueError("Non-S RGB26 features must be finite")
    return x, dict(patch_radius_px=radius, queried_ids=ids.tolist(),
                   scene_coordinates_are_masked=True, palette=scene["palette"], rows=diagnostics)


def extract_geometry_ownership(image_bgr, raw_points, eligible_ids, initial_projection_xy):
    """Bounded image cues at raw/non-S initial-model guidance locations.

    Palette mismatch remains a feature. Edge continuation compares image
    gradients along initially registered non-S edges near and away from the
    query. Initial projections provide directions/context only, never a
    corrected observation or a visibility target. No S coordinate is passed
    to scene construction or sampled as an RGB keypoint query.
    """
    from . import appearance as A
    ids = np.asarray(eligible_ids, int)
    q, projected = np.full((8, 2), np.nan), np.full((8, 2), np.nan)
    q[ids] = np.asarray(raw_points, float)[:8][ids]
    projected[ids] = np.asarray(initial_projection_xy, float)[ids]
    valid = np.zeros(8, bool)
    valid[ids] = True
    scene = A._scene(np.asarray(image_bgr), q, ~valid, valid, A._config(None))
    radius = float(scene["radius"])
    center = np.mean(projected[ids], axis=0) if len(ids) else np.zeros(2)
    # Native cuboid edges; neighbours that are S or otherwise excluded are
    # removed BEFORE any image samples or direction computations.
    neighbours = ((1, 3, 4), (0, 2, 5), (1, 3, 6), (0, 2, 7),
                  (0, 5, 7), (1, 4, 6), (2, 5, 7), (3, 4, 6))
    def sample(array, point):
        h, w = array.shape[:2]
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            return None
        return float(cv2.getRectSubPix(np.asarray(array, np.float32), (1, 1), tuple(map(float, point)))[0, 0])
    def average(array, points, absolute=False):
        values = [sample(array, point) for point in points]
        values = [abs(value) if absolute else value for value in values if value is not None]
        return float(np.mean(values)) if values else 0.
    rows = []
    for k in ids:
        direction = center - projected[k]
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        tangent = np.array([-direction[1], direction[0]])
        edge_dirs = []
        for j in neighbours[k]:
            if not valid[j]:
                continue
            edge = projected[j] - projected[k]
            if np.linalg.norm(edge) > 1e-6:
                edge_dirs.append(edge / np.linalg.norm(edge))
        values = []
        for source in (q[k], projected[k]):
            patch = cv2.getRectSubPix(scene["lab"], (5, 5), tuple(map(float, source))).astype(float)
            lab = np.median(patch.reshape(-1, 3), axis=0)
            palette = np.asarray(scene["palette"]["centers_lab"], float)
            distance = min(4., float(np.min(np.linalg.norm((palette-lab)/[48., 20., 20.], axis=1)))) if len(palette) else 4.
            chroma = float(np.linalg.norm(lab[1:]-128.) / 128.)
            inward = [average(scene["soft_material"], [source + direction*radius*d + tangent*radius*t for t in (-.25, 0, .25)]) for d in (1., 2.)]
            outward = [average(scene["soft_material"], [source - direction*radius*d + tangent*radius*t for t in (-.25, 0, .25)]) for d in (1., 2.)]
            normal_near, normal_far, tangent_near, tangent_far = [], [], [], []
            for edge in edge_dirs:
                normal = np.array([-edge[1], edge[0]])
                gnormal = scene["gx"]*normal[0] + scene["gy"]*normal[1]
                gtangent = scene["gx"]*edge[0] + scene["gy"]*edge[1]
                near = [source+edge*radius*d for d in (.5, 1., 1.5)]
                far = [source+edge*radius*d for d in (2., 3., 4.)]
                normal_near.append(average(gnormal, near, True)/32.)
                normal_far.append(average(gnormal, far, True)/32.)
                tangent_near.append(average(gtangent, near, True)/32.)
                tangent_far.append(average(gtangent, far, True)/32.)
            wedge = [source+direction*radius*d+tangent*radius*t for d in (.5, 1., 1.5, 2.) for t in (-.5, 0., .5)]
            soft = [sample(scene["soft_material"], point) for point in wedge]
            soft = [s for s in soft if s is not None]
            values.extend([distance, chroma, *inward, *outward,
                float(np.mean(normal_near)) if normal_near else 0., float(np.mean(normal_far)) if normal_far else 0.,
                float(np.mean(tangent_near)) if tangent_near else 0., float(np.mean(tangent_far)) if tangent_far else 0.,
                float(np.mean(soft)) if soft else 0., float(np.mean(np.asarray(soft)<.25)) if soft else 1.])
        rows.append(values)
    result = np.asarray(rows, float).reshape(len(ids), len(OWNERSHIP_FEATURE_NAMES))
    if not np.isfinite(result).all():
        raise ValueError("Geometry ownership features must be finite")
    return result


def infer_state_visibility(image_bgr, raw_points, initial_self_hidden, initial_roles,
                           initial_projection_xy, model, *, pallet_mask=None,
                           independent_truncation=None, feature_extractor=None,
                           probability_predictor=None):
    """Return native S/B/I/O/T states, or U for an internal abstention.

    Feature/predictor injection supports contract tests and alternate frozen
    descriptors. Neither callable is invoked when no non-S RGB query exists.
    The caller supplies a frozen pallet mask, or the descriptor binds its
    frozen source segmenter. S/T/U gates are resolved before even that load.
    """
    records, ids = geometry_state_gates(image_bgr, raw_points, initial_self_hidden,
                                      initial_roles, initial_projection_xy,
                                      independent_truncation=independent_truncation)
    radius = patch_radius(raw_points, ids)
    if len(ids):
        if not isinstance(model, StateVisibilityModel):
            model = load_state_model(str(model))
        family = model.descriptor.get("feature_family", "")
        rgb26_family = family.startswith("NON_S_RGB26")
        if rgb26_family:
            local, feature_details = extract_nonS_rgb26(image_bgr, raw_points, ids)
            spatial = None
            radius = feature_details["patch_radius_px"]
            if "GEOMETRY" in family:
                geometry = extract_geometry_ownership(image_bgr, raw_points, ids, initial_projection_xy)
                local = np.column_stack([local, geometry])
            if "RAW_SPATIAL" in family:
                spatial = spatial_features(image_bgr, np.asarray(raw_points, float)[:8], ids,
                                           patch_radius(raw_points, ids), model.frozen_backbone)
        elif pallet_mask is None:
            from .segmentation import segment
            mask_binding = model.descriptor["pallet_segmenter"]
            weights = ROOT / mask_binding["weights_repo_relative"]
            if sha256(weights) != mask_binding["weights_sha256"]:
                raise ValueError("Frozen pallet segmenter binding changed")
            pallet_mask = segment(image_bgr, weights, model.frozen_backbone.device)
        predict = probability_predictor or predict_actual_visibility
        if not rgb26_family:
            extract = feature_extractor or extract_state_features
            local, spatial = extract(image_bgr, raw_points, ids, radius, model.frozen_backbone,
                                     pallet_mask=pallet_mask)
        probability = np.asarray(predict(local, spatial, model), float)
        if probability.shape != (len(ids),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Visibility predictor must return one probability per gated query")
        threshold = float(model.descriptor["visible_threshold"])
        margin = float(model.descriptor.get("abstention_margin", 0.))
        if not (0 <= margin <= threshold <= 1 and threshold + margin <= 1):
            raise ValueError("Invalid visible threshold/abstention margin")
        for k, p in zip(ids, probability):
            state = records[k]["initial_role"] if p >= threshold + margin else "O" if p < threshold - margin else "U"
            if state == "O" and model.descriptor.get("external_O_enabled", True) is not True:
                state = "U"
            records[k].update(state=state, p_visible=float(p), rgb_evaluated=True,
                              abstained=state == "U", reason="human_DIRECT_vs_EXTERNAL_RGB_classifier",
                              external_O_enabled=model.descriptor.get("external_O_enabled", True))
    return dict(states=[r["state"] for r in records], records=records,
                rgb_queried_ids=ids.tolist(), p_visible=[r["p_visible"] for r in records],
                diagnostics=dict(schema=SCHEMA, patch_radius_px=radius,
                    self_excluded_before_RGB=True, coordinate_quality_is_visibility=False,
                    projected_points_sampled=False, coordinates_changed=False,
                    state_target="human_DIRECT_VISIBLE_vs_EXTERNAL_OCCLUDED",
                    conditional_T_is_independent_physical_fact=False,
                    U_is_internal_abstention=True))
