"""Independent RGB transitions between two visible cuboid face palettes.

The internal three-face junction is not a material/background outline corner.
Initial pose determines visible faces and edge identities; all fitted line
observations come from current image pixels. Intersections can still represent
a virtual cuboid vertex rather than a physical surface point on rounded pallets.
"""
from __future__ import annotations

import math
from itertools import combinations

import cv2
import numpy as np

from . import appearance as A
from . import boundary_refine as B
from . import common as C


FACES = A.FACES
NORMALS = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0],
                    [1, 0, 0], [0, 1, 0], [-1, 0, 0]], float)
EDGES = sorted({tuple(sorted((face[k], face[(k + 1) % 4]))) for face in FACES for k in range(4)})
DEFAULT_CONFIG = dict(minimum_face_contrast=8., normal_search_radius_px=12.,
    maximum_corner_shift_px=6., face_seed_scale=.6, minimum_face_seed_area=64,
    minimum_material_seed_purity=.55, minimum_palette_mode_purity=.55,
    palette_mode_radius_Lab=16., maximum_seed_samples=4000,
    face_front_margin_deg=2., maximum_pose_median_reprojection_px=20.,
    maximum_pose_P90_reprojection_px=40., minimum_face_area_px=100.,
    face_likelihood_confidence=.65, likelihood_sigma_floor=4., likelihood_sigma_ceiling=12.,
    maximum_absolute_face_distance_sigma=3., minimum_material_probe_support=.55,
    normal_step_px=.5, transition_probe_px=2.5, context_probe_px=5.,
    sample_spacing_px=4., minimum_samples=9, maximum_samples=49,
    along_interval=[.15, .85], minimum_edge_length_px=20.,
    minimum_inlier_count=6, minimum_inlier_fraction=.6, minimum_inlier_span_fraction=.55,
    huber_delta_px=1.25, maximum_line_rms_px=1.25, maximum_angle_change_deg=10.,
    minimum_intersection_sine=.25, maximum_intersection_disagreement_px=3.)


def _config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        if set(config) - set(cfg):
            raise ValueError("Unknown internal-face configuration fields")
        cfg.update(config)
    return cfg


def _sample_lab(lab, positions):
    return cv2.remap(lab, positions[..., 0].astype(np.float32), positions[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(float)


def _face_palette(lab, material, polygon, cfg):
    seed = np.zeros(material.shape, np.uint8)
    center = polygon.mean(0)
    A._fill_polygon(seed, center + cfg["face_seed_scale"] * (polygon - center))
    all_pixels = int(np.count_nonzero(seed))
    support = (seed > 0) & (material > 127)
    pixels = lab[support].astype(float)
    diag = dict(seed_area_px=all_pixels, RGB_material_supported_seed_pixels=len(pixels),
                RGB_material_seed_purity=len(pixels) / max(1, all_pixels), palette_valid=False)
    if all_pixels < cfg["minimum_face_seed_area"] or len(pixels) < cfg["minimum_face_seed_area"]:
        return None, dict(diag, reason="insufficient_face_seed_area")
    if diag["RGB_material_seed_purity"] < cfg["minimum_material_seed_purity"]:
        return None, dict(diag, reason="low_RGB_material_seed_purity")
    if len(pixels) > cfg["maximum_seed_samples"]:
        pixels = pixels[np.linspace(0, len(pixels) - 1, cfg["maximum_seed_samples"], dtype=int)]
    bins = np.floor(pixels / 8).astype(int)
    keys = bins[:, 0] * 1024 + bins[:, 1] * 32 + bins[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    proposals = np.argsort(-counts, kind="stable")[:12]
    best = None
    for position in proposals:
        centroid = np.median(pixels[keys == unique[position]], axis=0)
        members = np.linalg.norm(pixels - centroid, axis=1) <= cfg["palette_mode_radius_Lab"]
        if not members.any():
            continue
        centroid = np.median(pixels[members], axis=0)
        members = np.linalg.norm(pixels - centroid, axis=1) <= cfg["palette_mode_radius_Lab"]
        purity = float(np.mean(members))
        if best is None or purity > best[0]:
            best = purity, centroid, pixels[members]
    if best is None or best[0] < cfg["minimum_palette_mode_purity"]:
        return None, dict(diag, reason="noncompact_face_palette", palette_mode_purity=best[0] if best else 0.)
    purity, centroid, body = best
    mad = np.median(np.abs(body - centroid), axis=0) * 1.4826
    palette = dict(centroid=centroid, sigma=float(np.clip(np.sqrt(np.mean(mad ** 2)),
                    cfg["likelihood_sigma_floor"], cfg["likelihood_sigma_ceiling"])))
    return palette, dict(diag, reason="compact_RGB_face_palette", palette_valid=True,
        centroid_encoded_Lab=centroid.tolist(), robust_MAD_encoded_Lab=mad.tolist(),
        likelihood_sigma=palette["sigma"], palette_mode_purity=purity,
        seed_polygon=(center + cfg["face_seed_scale"] * (polygon - center)).tolist())


def build_internal_context(image_bgr, points, self_hidden, xyz, initial_R, initial_t, K, config=None):
    cfg = _config(config)
    image, q, hidden, valid, inside = A._inputs(image_bgr, points, self_hidden)
    xyz, R, t, K = np.asarray(xyz, float), np.asarray(initial_R, float), np.asarray(initial_t, float), np.asarray(K, float)
    if xyz.shape != (8, 3) or R.shape != (3, 3) or t.shape != (3,) or K.shape != (3, 3):
        raise ValueError("Expected ordered xyz[8,3], R/K[3,3], t[3]")
    projected = C.project(xyz, R, t, K)
    residual = np.linalg.norm(q[valid] - projected[valid], axis=1)
    coherence = bool(len(residual) >= 4 and np.median(residual) <= cfg["maximum_pose_median_reprojection_px"]
                     and np.quantile(residual, .9) <= cfg["maximum_pose_P90_reprojection_px"])
    camera_object = -R.T @ t
    facing, face_records, palettes = [], [], {}
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    material = A.build_appearance_mask(image, q, hidden)
    hull = np.flatnonzero(valid)
    hull_ids = hull[cv2.convexHull(q[hull].astype(np.float32), returnPoints=False).reshape(-1)].tolist() if len(hull) >= 3 else []
    for face_id, (ids, normal) in enumerate(zip(FACES, NORMALS)):
        ids = np.asarray(ids, int)
        direction = camera_object - xyz[ids].mean(0)
        cosine = float(normal @ direction / max(np.linalg.norm(direction), 1e-12))
        geometrically_visible = bool(cosine > np.sin(np.deg2rad(cfg["face_front_margin_deg"])) and not hidden[ids].any())
        record = dict(face=face_id, ordered_corner_ids=ids.tolist(), initial_outward_normal=normal.tolist(),
                      facing_cosine=cosine, geometrically_visible=geometrically_visible, palette_valid=False)
        if not coherence:
            record["reason"] = "initial_pose_q_incoherent"
        elif not geometrically_visible or not valid[ids].all():
            record["reason"] = "face_not_geometrically_visible_or_invalid_q"
        else:
            poly, reference = q[ids], projected[ids]
            signed = cv2.contourArea(poly.astype(np.float32), oriented=True)
            reference_signed = cv2.contourArea(reference.astype(np.float32), oriented=True)
            convex = cv2.isContourConvex(poly.astype(np.float32))
            if not convex or abs(signed) < cfg["minimum_face_area_px"] or signed * reference_signed <= 0:
                record["reason"] = "face_polygon_pose_orientation_or_area_incoherent"
            else:
                facing.append(face_id)
                palette, detail = _face_palette(lab, material, poly, cfg)
                record.update(detail)
                if palette is not None:
                    palettes[face_id] = palette
        face_records.append(record)
    return dict(q=q, original=np.asarray(points, float).copy(), hidden=hidden.copy(), valid=valid, inside=inside,
        lab=lab, material=material, image_hw=image.shape[:2], hull_ids=hull_ids,
        visible_faces=facing, face_records=face_records, palettes=palettes,
        pose_coherence=dict(usable=coherence, median_q_projection_px=float(np.median(residual)) if len(residual) else None,
                            P90_q_projection_px=float(np.quantile(residual, .9)) if len(residual) else None),
        scene_config=cfg)


def _pair_probability(pixels, first, second):
    sigma = .5 * (first["sigma"] + second["sigma"])
    da = np.sum((pixels - first["centroid"]) ** 2, axis=-1) / sigma ** 2
    db = np.sum((pixels - second["centroid"]) ** 2, axis=-1) / sigma ** 2
    probability = 1. / (1. + np.exp(np.clip(.5 * (da - db), -60, 60)))
    return probability, da, db


def _observe_internal_edge(context, a_id, b_id, face_a, face_b, cfg):
    q, lab, material = context["q"], context["lab"], context["material"]
    a, b = q[a_id], q[b_id]; length = float(np.linalg.norm(b - a))
    if length < cfg["minimum_edge_length_px"]:
        return None, dict(reason="short_internal_cube_edge", length_px=length)
    normal = np.array([-(b-a)[1], (b-a)[0]]) / length
    centers_a, centers_b = q[list(FACES[face_a])].mean(0), q[list(FACES[face_b])].mean(0)
    if normal @ (centers_a - (a+b)/2) < 0:
        normal = -normal
    if normal @ (centers_b - (a+b)/2) >= -.5 or normal @ (centers_a - (a+b)/2) <= .5:
        return None, dict(reason="adjacent_face_sides_incoherent")
    pa, pb = context["palettes"][face_a], context["palettes"][face_b]
    contrast = float(np.linalg.norm(pa["centroid"] - pb["centroid"]))
    diag = dict(length_px=length, positive_side_face=face_a, negative_side_face=face_b,
                face_centroid_contrast_encoded_Lab=contrast, normal_toward_face_a=normal.tolist())
    if contrast < cfg["minimum_face_contrast"]:
        return None, dict(diag, reason="adjacent_face_palettes_not_separable")
    samples = int(np.clip(math.ceil(length / cfg["sample_spacing_px"]) + 1, cfg["minimum_samples"], cfg["maximum_samples"]))
    along = np.linspace(*cfg["along_interval"], samples)
    offsets = np.arange(-cfg["normal_search_radius_px"], cfg["normal_search_radius_px"] + .25, cfg["normal_step_px"])
    pos = a + along[:, None] * (b - a)
    positions = pos[:, None, :] + offsets[None, :, None] * normal
    plus, minus, distance_a, distance_b, mat_a, mat_b = [], [], [], [], [], []
    for factor in (.5, 1., 1.5):
        plus_positions = positions + normal * cfg["transition_probe_px"] * factor
        minus_positions = positions - normal * cfg["transition_probe_px"] * factor
        prob_a, da, _ = _pair_probability(_sample_lab(lab, plus_positions), pa, pb)
        prob_ba, _, db = _pair_probability(_sample_lab(lab, minus_positions), pa, pb)
        plus.append(prob_a); minus.append(1-prob_ba); distance_a.append(da); distance_b.append(db)
        mat_a.append(B._remap(material, plus_positions)); mat_b.append(B._remap(material, minus_positions))
    support_a, support_b = np.mean(plus, axis=0), np.mean(minus, axis=0)
    da, db = np.mean(distance_a, axis=0), np.mean(distance_b, axis=0)
    material_a, material_b = np.mean(mat_a, axis=0), np.mean(mat_b, axis=0)
    ca = _pair_probability(_sample_lab(lab, positions + normal * cfg["context_probe_px"]), pa, pb)[0]
    cb = 1 - _pair_probability(_sample_lab(lab, positions - normal * cfg["context_probe_px"]), pa, pb)[0]
    fine = (_pair_probability(_sample_lab(lab, positions + normal * .5), pa, pb)[0]
            - _pair_probability(_sample_lab(lab, positions - normal * .5), pa, pb)[0])
    h, w = context["image_hw"]; margin = cfg["context_probe_px"]
    pi, po = positions + normal * margin, positions - normal * margin
    in_frame = ((pi[...,0]>=0)&(pi[...,0]<w-1)&(pi[...,1]>=0)&(pi[...,1]<h-1)
                &(po[...,0]>=0)&(po[...,0]<w-1)&(po[...,1]>=0)&(po[...,1]<h-1))
    limit = cfg["maximum_absolute_face_distance_sigma"] ** 2
    available = (in_frame & (support_a>=cfg["face_likelihood_confidence"])
                 & (support_b>=cfg["face_likelihood_confidence"]) & (ca>=cfg["face_likelihood_confidence"])
                 & (cb>=cfg["face_likelihood_confidence"]) & (da<=limit)&(db<=limit)
                 & (material_a>=cfg["minimum_material_probe_support"])
                 & (material_b>=cfg["minimum_material_probe_support"]))
    transition = np.minimum(support_a, support_b)
    score = np.where(available, transition + .35*fine + .10*np.minimum(ca,cb) - .005*np.abs(offsets)[None,:], -np.inf)
    chosen = np.argmax(score, axis=1); usable = np.isfinite(score[np.arange(samples), chosen])
    diag.update(samples=samples, transition_sample_count=int(usable.sum()),
                probe_identity="positive side faceA versus negative side faceB; BOTH require current RGB pallet-material support")
    if usable.sum() < cfg["minimum_inlier_count"]:
        return None, dict(diag, reason="insufficient_two_face_RGB_transitions")
    obs = positions[np.arange(samples)[usable], chosen[usable]]
    confidence = transition[np.arange(samples)[usable], chosen[usable]]
    line, fit = B._fit_line(obs, confidence, along[usable], normal, samples, cfg)
    idx = (np.arange(samples)[usable], chosen[usable])
    fit.update(observed_sideA_likelihood=support_a[idx].tolist(), observed_sideB_likelihood=support_b[idx].tolist(),
               observed_sideA_material_support=material_a[idx].tolist(), observed_sideB_material_support=material_b[idx].tolist())
    return line, dict(diag, **fit)


def infer_from_context(context, config=None):
    cfg = _config(config)
    q, hidden = context["q"], context["hidden"]
    result = context["original"].copy()
    edges, lines = [], {}
    visible = set(context["visible_faces"])
    for edge_id, (a, b) in enumerate(EDGES):
        incident = [face_id for face_id in visible if a in FACES[face_id] and b in FACES[face_id]]
        if len(incident) != 2:
            continue
        row = dict(edge=edge_id, endpoints=[a,b], face_ids=incident, supported=False)
        if not context["inside"][[a,b]].all() or hidden[[a,b]].any():
            row["reason"] = "internal_edge_endpoint_invalid_or_out_of_frame_or_self_hidden"
        elif any(face not in context["palettes"] for face in incident):
            row["reason"] = "adjacent_face_palette_unavailable"
        else:
            line, observation = _observe_internal_edge(context, a, b, *incident, cfg)
            row.update(observation); row["supported"] = line is not None
            if line is not None:
                lines[edge_id] = line
        edges.append(row)
    records, refined = [], []
    for k in range(8):
        faces = [face_id for face_id in visible if k in FACES[face_id]]
        row = dict(corner=k, visible_incident_faces=faces, refined=False,
                   is_hull_corner=k in context["hull_ids"])
        if hidden[k] or not context["inside"][k] or k in context["hull_ids"] or len(faces) != 3:
            row["reason"] = "not_visible_nonhull_three_face_junction"
            records.append(row); continue
        incident_edges = [i for i,(a,b) in enumerate(EDGES) if k in (a,b) and i in lines]
        row["incident_RGB_supported_edges"] = incident_edges
        candidates = []
        for first, second in combinations(incident_edges, 2):
            matrix = np.stack([lines[first]["normal"], lines[second]["normal"]])
            sine = abs(float(np.linalg.det(matrix)))
            if sine >= cfg["minimum_intersection_sine"]:
                candidate = np.linalg.solve(matrix, -np.array([lines[first]["offset"],lines[second]["offset"]]))
                candidates.append((sine, first, second, candidate))
        if not candidates:
            row["reason"] = "fewer_than_two_well_conditioned_RGB_line_families"
            records.append(row); continue
        best = max(candidates,key=lambda x:x[0]); candidate = best[3]
        disagreement = float(np.max(np.linalg.norm(np.stack([item[3] for item in candidates]) - candidate,axis=1)))
        distance = float(np.linalg.norm(candidate-q[k]))
        row.update(candidate_xy=candidate.tolist(), shift_px=distance,
                   intersection_edges=list(best[1:3]), intersection_sine=best[0],
                   multiple_intersection_disagreement_px=disagreement,
                   correlated_with_any_other_corner_using_same_edges=True)
        h,w=context["image_hw"]
        if not np.isfinite(candidate).all() or not (0<=candidate[0]<w and 0<=candidate[1]<h):
            row["reason"]="intersection_out_of_frame"
        elif disagreement>cfg["maximum_intersection_disagreement_px"]:
            row["reason"]="third_RGB_line_disagrees_with_intersection"
        elif distance>cfg["maximum_corner_shift_px"]:
            row["reason"]="RGB_intersection_exceeds_declared_shift_cap"
        else:
            result[k]=candidate;refined.append(k);row.update(refined=True,reason="two_visible_face_RGB_transition_intersection")
        records.append(row)
    return dict(points=result.tolist(),observed_refined_ids=refined,
        observation_source="CURRENT_RGB_TWO_VISIBLE_FACE_PALETTE_TRANSITIONS",
        independent_image_evidence=True,reprojected_points_used=False,
        outer_material_background_edges_used=False,physical3D_corner_certified=False,
        face_records=context["face_records"],edge_records=edges,corner_records=records,
        diagnostics=dict(config=cfg,pose_coherence=context["pose_coherence"],
                         visible_faces=context["visible_faces"],hull_corner_ids=context["hull_ids"],
                         shared_material_mask_source="current_RGB_palette_only_no_GT",input_points_unchanged=True))


def infer_internal_face_points(image_bgr,points,self_hidden,xyz,initial_R,initial_t,K,config=None):
    context=build_internal_context(image_bgr,points,self_hidden,xyz,initial_R,initial_t,K,config)
    return infer_from_context(context,config)
