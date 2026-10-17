"""One automatic, fixed-support RGB region boundary model and edge control.

SRT3D's color-posterior and correspondence-line equations are reimplemented
in NumPy; this is a single-frame conditional energy, not an SRT3D tracker or
a calibrated pose posterior. Geometry only guides the initial RGB queries.
No reference, label, manual-boundary, image-ID or candidate-pose input is
used to produce observations. Candidate evaluation never queries pixels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class BoundaryConfig:
    # alpha/slope/bins: official SRT3D RegionModality header defaults.
    alpha: float = 0.36
    slope_base_px: float = 0.0
    histogram_bins_per_channel: int = 32
    # Original face-aware local RGB reach: along_end36 + context7.
    sample_radius_base_px: int = 43
    offset_radius_base_px: int = 36
    step_base_px: float = 1.0
    along_base_px: tuple = (4.0, 14.666666666666666, 25.333333333333332, 36.0)
    # Official appearance-line defaults: ignore1, consider18 pixels.
    appearance_start_base_px: float = 1.0
    appearance_end_base_px: float = 18.0
    protected_unknown_mate_radius_base_px: float = 7.0
    canonical_image_diagonal_px: float = 800.0


DEFAULT_CONFIG = BoundaryConfig()


def configuration_dict(config=None):
    return asdict(_config(config))


def _config(config):
    if config is None:
        return DEFAULT_CONFIG
    result = config if isinstance(config, BoundaryConfig) else BoundaryConfig(**config)
    if not (0 < result.alpha < 0.5) or result.slope_base_px < 0:
        raise ValueError("Require 0<alpha<.5 and nonnegative fixed slope")
    if result.histogram_bins_per_channel not in (2, 4, 8, 16, 32, 64):
        raise ValueError("Official power-of-two histogram binning required")
    if not (result.sample_radius_base_px > result.offset_radius_base_px > 0 and result.step_base_px > 0):
        raise ValueError("Positive common sample/offset domain with context required")
    if (not result.along_base_px or min(result.along_base_px) <= 0 or
            result.canonical_image_diagonal_px <= 0):
        raise ValueError("Positive fixed along-edge locations and canonical diagonal required")
    return result


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _finite(a, shape, name):
    a = np.asarray(a, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + " has invalid finite shape")
    return a


def _hull_indices(points):
    """Deterministic convex-hull indices; no image evidence is manufactured."""
    unique = sorted((float(p[0]), float(p[1]), i) for i, p in enumerate(points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [p[2] for p in lower[:-1] + upper[:-1]]


def _pixel_indices(coordinates, image_hw):
    rounded = np.floor(np.asarray(coordinates) + 0.5).astype(np.int64)
    h, w = image_hw
    valid = ((rounded[..., 0] >= 0) & (rounded[..., 0] < w) &
             (rounded[..., 1] >= 0) & (rounded[..., 1] < h))
    return rounded, valid


def offset_distribution(foreground_probability, sample_offsets_base_px, offset_grid_base_px, *,
                        alpha=0.36, slope_base_px=0.0):
    """Retain the normalized full discrete Eq14 offset distribution.

    Negative r denotes foreground, positive r background. The uniform prior
    is over the same fixed offset grid for every candidate. No nearest peak
    choice, moment compression, covariance fitting or mode removal occurs.
    """
    pf, r, d = map(lambda x: np.asarray(x, float),
                   (foreground_probability, sample_offsets_base_px, offset_grid_base_px))
    if (pf.ndim != 1 or pf.shape != r.shape or d.ndim != 1 or len(d) < 2 or
            not np.isfinite(np.r_[pf, r, d]).all() or np.any((pf < 0) | (pf > 1)) or
            np.any(np.diff(r) <= 0) or np.any(np.diff(d) <= 0)):
        raise ValueError("Ordered finite fixed line coordinates and color posteriors required")
    if not 0 < alpha < .5 or slope_base_px < 0:
        raise ValueError("Invalid fixed smoothed-step parameters")
    x = r[:, None] - d[None, :]
    transition = np.sign(x) if slope_base_px == 0 else np.tanh(x / (2 * slope_base_px))
    foreground_step = .5 - alpha * transition
    terms = foreground_step * pf[:, None] + (1 - foreground_step) * (1 - pf[:, None])
    log_unnormalized = np.log(terms).sum(axis=0)
    shifted = log_unnormalized - log_unnormalized.max()
    log_probability = shifted - np.log(np.exp(shifted).sum())
    return np.exp(log_probability), log_probability


def _peak_indices(probability):
    p = np.asarray(probability)
    if np.ptp(p) <= 1e-14:
        return []
    # Plateaus remain in the array; peak readout is diagnostic only.
    return np.flatnonzero((p >= np.r_[-np.inf, p[:-1]]) &
                          (p >= np.r_[p[1:], -np.inf]) &
                          ((p > np.r_[-np.inf, p[:-1]]) |
                           (p > np.r_[p[1:], -np.inf]))).tolist()


def _bin_indices(colors, bins):
    q = np.asarray(colors, np.int64) * bins // 256
    return q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]


def _fit_shared_appearance(image, mask, profiles, r, cfg):
    h, w = mask.shape
    foreground, background = set(), set()
    negative = (r <= -cfg.appearance_start_base_px) & (r >= -cfg.appearance_end_base_px)
    positive = (r >= cfg.appearance_start_base_px) & (r <= cfg.appearance_end_base_px)
    for p in profiles:
        xy = np.asarray(p["sample_pixel_xy"], int)
        m = mask[xy[:, 1], xy[:, 0]]
        foreground.update((int(x), int(y)) for x, y in xy[negative & (m >= .5)])
        background.update((int(x), int(y)) for x, y in xy[positive & (m < .5)])
    result = dict(foreground_pixels=len(foreground), background_pixels=len(background),
                  unique_pixel_count=len(foreground | background),
                  scope="single fixed frame; all candidate hypotheses share this appearance",
                  temporal_updates=0, pseudocount=0, equal_foreground_background_priors=True,
                  duplicate_pixels_counted_once=True, bins_per_channel=cfg.histogram_bins_per_channel)
    if not foreground or not background:
        return None, None, {**result, "supported": False, "reason": "missing_foreground_or_background_appearance_pixels"}
    histograms = []
    for positions in (foreground, background):
        xy = np.asarray(sorted(positions), int)
        indices = _bin_indices(image[xy[:, 1], xy[:, 0]], cfg.histogram_bins_per_channel)
        histogram = np.bincount(indices, minlength=cfg.histogram_bins_per_channel ** 3).astype(float)
        histogram /= histogram.sum()
        histograms.append(histogram)
    result.update(supported=True, foreground_histogram_sha256=_array_digest(histograms[0]),
                  background_histogram_sha256=_array_digest(histograms[1]),
                  appearance_support_sha256=_canonical([sorted(foreground), sorted(background)]))
    return *histograms, result


def build_boundary_bundle(image_rgb, target_mask, packet, config=None):
    """Build B-only automatic RGB profiles once, independent of all candidates.

    packet is the original typed packet. Only solver_input states, visible
    observations, registered geometry/coarse pose and activation are read.
    The caller authenticates the original RGB and mask file bindings. Inputs
    are current raw image coordinates, with no resize or crop in this module.
    """
    cfg = _config(config)
    image = np.asarray(image_rgb)
    mask = np.asarray(target_mask, float)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Actual uint8 RGB image required, explicitly convert BGR at caller")
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("Original finite target mask must share the raw RGB canvas")
    spec = packet["solver_input"] if "solver_input" in packet else packet
    states = tuple(spec["states"])
    if len(states) != 8 or any(s not in ("B", "I", "S", "O", "T", "U") for s in states):
        raise ValueError("Eight original authoritative native states required")
    image_hw = tuple(spec.get("image_hw", spec.get("raw_hw", ())))
    if image_hw != mask.shape:
        raise ValueError("Original packet and RGB resolution disagree")
    xyz = _finite(spec.get("xyz", spec.get("native_xyz")), (8, 3), "native cuboid")
    K = _finite(spec["K"], (3, 3), "camera matrix")
    coarse = spec.get("coarse_pose", spec.get("initial_pose"))
    R = _finite(coarse.get("R", coarse.get("R_cf")), (3, 3), "prediction-only initial rotation")
    t = _finite(coarse.get("t", coarse.get("centroid")), (3,), "prediction-only initial center")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError("Initial rotation must be proper")
    camera = xyz @ R.T + t
    if np.min(camera[:, 2]) <= 0:
        raise ValueError("Initial native geometry must have positive depth")
    hom = camera @ K.T
    projection = hom[:, :2] / hom[:, 2:]
    hull = _hull_indices(projection)
    neighbors = {}
    for i, k in enumerate(hull):
        neighbors[k] = (hull[i-1], hull[(i+1) % len(hull)])
    records = {int(m["native_corner"]): m for m in spec.get("visible", spec.get("measurements", []))}
    scale = float(np.hypot(*image_hw) / cfg.canonical_image_diagonal_px)
    r = np.arange(-cfg.sample_radius_base_px, cfg.sample_radius_base_px + cfg.step_base_px * .5,
                  cfg.step_base_px)
    d = np.arange(-cfg.offset_radius_base_px, cfg.offset_radius_base_px + cfg.step_base_px * .5,
                  cfg.step_base_px)
    raw_profiles, skipped = [], []
    for k, state in enumerate(states):
        if state != "B":
            continue  # Gate before reading any observation or querying RGB.
        if k not in records or k not in neighbors:
            skipped.append(dict(native_corner=k, reason="no_B_anchor_or_not_on_initial_cuboid_hull"))
            continue
        anchor = _finite(records[k]["xy"], (2,), "actual B guide coordinate")
        for mate in neighbors[k]:
            # This remains a conditional cuboid edge, never a certified material edge.
            if states[mate] == "S" or np.count_nonzero(np.abs(xyz[k]-xyz[mate]) > 1e-7) != 1:
                skipped.append(dict(native_corner=k, native_neighbor=mate, reason="S_mate_or_non_native_hull_edge"))
                continue
            guide = projection[mate] - projection[k]
            length = float(np.linalg.norm(guide))
            if length <= 1e-8:
                continue
            tangent = guide / length
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (projection[k] + projection[mate]) / 2
            if normal @ (midpoint - projection.mean(axis=0)) < 0:
                normal *= -1
            for along in cfg.along_base_px:
                if along * scale >= length:
                    skipped.append(dict(native_corner=k, native_neighbor=mate, along_base_px=along,
                                        reason="along_query_beyond_initial_native_edge"))
                    continue
                center = anchor + along * scale * tangent
                coordinates = center + r[:, None] * scale * normal
                rounded, valid = _pixel_indices(coordinates, image_hw)
                reason = None
                if not valid.all():
                    reason = "normal_profile_outside_actual_RGB_no_border_padding"
                elif states[mate] in ("O", "T", "U") and np.min(np.linalg.norm(coordinates-projection[mate], axis=1)) < cfg.protected_unknown_mate_radius_base_px * scale:
                    reason = "unknown_mate_neighborhood_excluded_before_RGB_query"
                if reason:
                    skipped.append(dict(native_corner=k, native_neighbor=mate, along_base_px=along, reason=reason))
                    continue
                raw_profiles.append(dict(native_corner=k, native_neighbor=mate,
                    center_xy=center.tolist(), outward_normal_xy=normal.tolist(),
                    sample_xy=coordinates.tolist(), sample_pixel_xy=rounded.tolist(),
                    along_base_px=along, sample_offsets_base_px=r.tolist(),
                    offset_grid_base_px=d.tolist(), initial_geometry_mate_state=states[mate],
                    mate_supplies_direction_only=True, precise_mate_observation_created=False,
                    physical_owner_status="virtual_cuboid_feature_or_unknown_material_correspondence"))
    hist_f, hist_b, appearance = _fit_shared_appearance(image, mask, raw_profiles, r, cfg)
    groups = []
    for k in sorted({p["native_corner"] for p in raw_profiles}):
        queries = []
        for profile in (p for p in raw_profiles if p["native_corner"] == k):
            p = dict(profile)
            xy = np.asarray(p["sample_pixel_xy"], int)
            colors = image[xy[:, 1], xy[:, 0]]
            p["sample_rgb"] = colors.tolist()
            p["sample_rgb_sha256"] = _array_digest(colors)
            p["support_id"] = _canonical({key: p[key] for key in
                ("native_corner", "native_neighbor", "sample_xy", "sample_pixel_xy")})
            if hist_f is None:
                p.update(supported=False, reason="shared_appearance_unsupported")
            else:
                indices = _bin_indices(colors, cfg.histogram_bins_per_channel)
                denominator = hist_f[indices] + hist_b[indices]
                pf = np.divide(hist_f[indices], denominator, out=np.full(len(indices), .5), where=denominator > 0)
                probability, logp = offset_distribution(pf, r, d, alpha=cfg.alpha, slope_base_px=cfg.slope_base_px)
                local_mask = mask[xy[:, 1], xy[:, 0]]
                negative_end = r < -cfg.offset_radius_base_px
                positive_end = r > cfg.offset_radius_base_px
                direction_supported = bool(np.mean(local_mask[negative_end]) >= .5 and
                                           np.mean(local_mask[positive_end]) < .5)
                informative = bool(np.ptp(logp) > 1e-12 and np.any(pf > .5) and np.any(pf < .5))
                signed_drop = pf[:-1] - pf[1:]
                edge_index = int(np.argmax(signed_drop))
                edge_offset = float((r[edge_index] + r[edge_index+1]) / 2)
                supported = direction_supported and informative and float(signed_drop[edge_index]) > 0
                p.update(foreground_probability=pf.tolist(), offset_probability=probability.tolist(),
                    offset_log_probability=logp.tolist(), peak_indices=_peak_indices(probability),
                    exact_flat=bool(np.ptp(logp) <= 1e-12), direction_supported=direction_supported,
                    supported=supported, reason="fixed_RGB_region_distribution" if supported else
                        "flat_or_one_sided_color_support" if direction_supported else "mask_does_not_support_foreground_to_background_direction",
                    edge_location_base_px=edge_offset, edge_tie_rule="first largest signed foreground-to-background posterior drop; no candidate input",
                    edge_signed_drop=float(signed_drop[edge_index]))
            queries.append(p)
        supported_queries = [p for p in queries if p["supported"]]
        reliability = float(records[k].get("fit_reliability_weight", 1.))
        if not np.isfinite(reliability) or reliability < 0:
            raise ValueError("Finite nonnegative original B reliability required")
        groups.append(dict(native_corner=k, supported=bool(supported_queries),
            reliability_weight=reliability, queries=queries,
            supported_query_count=len(supported_queries), aggregation="mean of fixed unique supported profiles; original B reliability once",
            consumes_existing_B_point_and_partial=bool(supported_queries)))
    consumed = [g["native_corner"] for g in groups if g["supported"]]
    activation = packet.get("activation", {})
    complete = set(activation.get("complete_ids", [m["native_corner"] for m in spec.get("visible", [])]))
    partial = set(activation.get("all_partial_ids", [b["native_corner"] for b in spec.get("boundary_observations", [])]))
    return dict(schema="fixed_RGB_boundary_distribution_v1", config=configuration_dict(cfg),
        image_hw=list(image_hw), pixel_scale=scale, image_rgb_pixel_sha256=_array_digest(image),
        target_mask_float64_sha256=_array_digest(mask), initial_projection_xy=projection.tolist(),
        original_states=list(states), groups=groups, skipped=skipped, shared_appearance=appearance,
        consumed_native_ids=consumed,
        consumed_group_ids=sorted([f"full:{k}" for k in consumed if k in complete] +
                                  [f"partial:{k}" for k in consumed if k in partial]),
        replaces_global_outline=bool(consumed), RGB_queries_created_once=True,
        new_precise_O_T_S_U_I_observations=0, new_complete_corner_observations=0,
        candidate_conditioned_appearance=False, calibrated_pose_probability_claimed=False,
        source="automatic original RGB + original clean target mask + prediction-only coarse guide",
        caveat="Registered cuboid silhouette edges can be virtual; actual material correspondence is not independently certified")


def _intersection_offset_and_gradient(query, projected, pixel_scale):
    k, mate = query["native_corner"], query["native_neighbor"]
    c, n = np.asarray(query["center_xy"]), np.asarray(query["outward_normal_xy"])
    v = projected[mate] - projected[k]
    cross = lambda a, b: float(a[0] * b[1] - a[1] * b[0])
    denominator = cross(n, v)
    if np.linalg.norm(v) <= 1e-10 or abs(denominator) <= 1e-10 * max(1., np.linalg.norm(v)):
        return None, np.zeros((8, 2))
    w = projected[k] - c
    numerator = cross(w, v)
    rotate = lambda x: np.array([-x[1], x[0]])
    numerator_a = -rotate(projected[mate] - c)
    numerator_b = rotate(w)
    denominator_a, denominator_b = -rotate(n), rotate(n)
    gradient = np.zeros((8, 2))
    gradient[k] = (numerator_a * denominator - numerator * denominator_a) / denominator**2 / pixel_scale
    gradient[mate] = (numerator_b * denominator - numerator * denominator_b) / denominator**2 / pixel_scale
    return numerator / denominator / pixel_scale, gradient


def _profile_energy_and_derivative(query, offset, model, cfg):
    grid = np.asarray(query["offset_grid_base_px"], float)
    laplace_slope = 2 * np.log((.5 + cfg.alpha) / (.5 - cfg.alpha))
    if offset is None or not np.isfinite(offset):
        # One fixed bad-geometry penalty; observations never disappear per pose.
        return float(laplace_slope * 4 * cfg.sample_radius_base_px), 0.
    if model == "edge":
        delta = offset - query["edge_location_base_px"]
        return float(laplace_slope * abs(delta)), float(laplace_slope * np.sign(delta))
    logp = np.asarray(query["offset_log_probability"], float)
    energy = -2 * (logp - logp.max())
    clipped = np.clip(offset, grid[0], grid[-1])
    value = float(np.interp(clipped, grid, energy) + laplace_slope * abs(offset-clipped))
    if offset < grid[0]:
        derivative = -laplace_slope
    elif offset > grid[-1]:
        derivative = laplace_slope
    else:
        j = min(max(int(np.searchsorted(grid, offset, side="right")) - 1, 0), len(grid)-2)
        derivative = float((energy[j+1]-energy[j]) / (grid[j+1]-grid[j]))
        # At exact knots use the symmetric derivative, matching central FD.
        knots = np.flatnonzero(np.abs(grid-offset) < 1e-10)
        if len(knots):
            j = int(knots[0])
            left = -laplace_slope if j == 0 else (energy[j]-energy[j-1])/(grid[j]-grid[j-1])
            right = laplace_slope if j == len(grid)-1 else (energy[j+1]-energy[j])/(grid[j+1]-grid[j])
            derivative = float((left+right)/2)
    return value, derivative


def candidate_energy(bundle, projected_native_xy, model="region", *, return_audit=False):
    """Evaluate the same fixed observations for every pose/owner hypothesis.

    For AB, pass projected_native_xy[owner_permutation]. No RGB, mask, pose
    fitting, reference or per-candidate support selection is possible here.
    A fixed normal line intersects the candidate's physical projected edge;
    the projected corner alone is not used as the profile displacement.
    """
    value, gradient, audit = energy_and_gradient(bundle, projected_native_xy, model, return_audit=True)
    return (value, audit) if return_audit else value


def energy_and_gradient(bundle, projected_native_xy, model="region", *, return_audit=False):
    """Analytic dE/dq[8,2], including physical-edge/normal-line intersection.

    Piecewise-linear profile interpolation uses the symmetric derivative at
    exact knots; the edge-control absolute value uses zero at its cusp.
    Caller-owned SE(3)/projection Jacobians can apply the chain rule. For AB,
    scatter the returned gradient back through the same owner permutation.
    """
    if model not in ("region", "edge"):
        raise ValueError("One frozen region model or its same-support edge control required")
    projected = _finite(projected_native_xy, (8, 2), "candidate native projections")
    cfg = _config(bundle["config"])
    total, gradient, audits = 0., np.zeros((8, 2)), []
    for group in bundle["groups"]:
        # Duplicate serialized support cannot create extra information.
        unique = {q["support_id"]: q for q in group["queries"] if q["supported"]}
        if not group["supported"] or not unique:
            continue
        per_query, query_gradients = [], []
        for q in unique.values():
            d, d_gradient = _intersection_offset_and_gradient(q, projected, bundle["pixel_scale"])
            energy, derivative = _profile_energy_and_derivative(q, d, model, cfg)
            query_gradients.append(derivative * d_gradient)
            per_query.append(dict(support_id=q["support_id"], native_neighbor=q["native_neighbor"],
                                  offset_base_px=d, energy=energy, candidate_geometry_valid=d is not None))
        unweighted = float(np.mean([q["energy"] for q in per_query]))
        value = float(group["reliability_weight"] * unweighted)
        total += value
        gradient += group["reliability_weight"] * np.mean(query_gradients, axis=0)
        audits.append(dict(native_corner=group["native_corner"], value=value,
                           profile_count=len(per_query), profiles=per_query))
    audit = dict(model=model, value=float(total), groups=audits,
                 consumed_native_ids=bundle["consumed_native_ids"],
                 consumed_group_ids=bundle["consumed_group_ids"],
                 replaces_global_outline=bundle["replaces_global_outline"],
                 candidate_specific_RGB_queries=0, candidate_specific_appearance_updates=0,
                 common_observation_universe=True, independent_sensor_probability_claimed=False)
    audit["gradient_kind"] = "analytic projected-edge intersection plus piecewise-linear log-profile interpolation"
    return (float(total), gradient, audit) if return_audit else (float(total), gradient)


def load_bound_frame(source_root, packet):
    """Authenticate and load the exact existing RGB and pallet_mask_clean.

    Explicit helper only: model construction/evaluation themselves do no file
    IO. Never substitutes a missing file or invokes a new segmentation model.
    """
    from PIL import Image
    root = Path(source_root)
    image_path = root / packet["image"]
    mask_binding = packet["provenance"]["original_mask_binding"]
    mask_path = root / mask_binding["path"]
    for path, expected in ((image_path, packet["image_sha256"]), (mask_path, mask_binding["sha256"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Original bound RGB/mask bytes changed: " + str(path))
    image = np.asarray(Image.open(image_path).convert("RGB"))
    with np.load(mask_path, allow_pickle=False) as arrays:
        mask = arrays["pallet_mask_clean"].copy()
    return image, mask


def cal_probe(source_root, packets_path, output_dir):
    """Inspect all CAL46 RGB supports, without pose/reference scoring."""
    from collections import Counter
    import gzip
    import os
    import resource
    import time
    started, cpu = time.perf_counter(), time.process_time()
    source_root, packets_path, output_dir = map(Path, (source_root, packets_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, preview, inputs, rows = Counter(), [], [], []
    bundle_path = output_dir / "CAL_BOUNDARY_BUNDLES.jsonl.gz"
    with gzip.open(packets_path, "rt") as stream, gzip.open(bundle_path, "wt") as output:
        for line in stream:
            packet = json.loads(line)
            if packet["partition"] != "CAL":
                continue
            load_start = time.perf_counter()
            image, mask = load_bound_frame(source_root, packet)
            counts["image_load_and_SHA_seconds"] += time.perf_counter()-load_start
            build_start = time.perf_counter()
            bundle = build_boundary_bundle(image, mask, packet)
            counts["observation_build_seconds"] += time.perf_counter()-build_start
            output.write(json.dumps(dict(id=packet["id"], bundle=bundle), separators=(",", ":"), allow_nan=False)+"\n")
            counts["frames"] += 1
            counts["supported_frames"] += bool(bundle["consumed_native_ids"])
            counts["B_corners"] += packet["solver_input"]["states"].count("B")
            counts["consumed_B_corners"] += len(bundle["consumed_native_ids"])
            counts["queried_profiles"] += sum(len(g["queries"]) for g in bundle["groups"])
            counts["supported_profiles"] += sum(g["supported_query_count"] for g in bundle["groups"])
            counts.update("query_reason:"+q["reason"] for g in bundle["groups"] for q in g["queries"])
            counts.update("skip_reason:"+s["reason"] for s in bundle["skipped"])
            rows.append(dict(id=packet["id"], B_corners=packet["solver_input"]["states"].count("B"),
                             consumed_native_ids=bundle["consumed_native_ids"],
                             supported_profiles=sum(g["supported_query_count"] for g in bundle["groups"])))
            inputs.append(dict(id=packet["id"], image=packet["image"], image_sha256=packet["image_sha256"],
                               mask_binding=packet["provenance"]["original_mask_binding"]))
            if len(preview) < 3:
                preview.append((packet["id"], image, packet, bundle))
    if counts["frames"] != 46:
        raise ValueError("CAL probe requires the exact original CAL46, never a favorable subset")
    render_start = time.perf_counter()
    preview_path = output_dir / "CAL_BOUNDARY_OVERVIEW.png"
    _cal_preview(preview, preview_path)
    counts["preview_render_seconds"] = time.perf_counter()-render_start
    source_path = Path(__file__)
    receipt = dict(schema="boundary_CAL46_RGB_probe_v1", config=configuration_dict(),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        packet_sha256=hashlib.sha256(packets_path.read_bytes()).hexdigest(),
        source_root=str(source_root), packet_path=str(packets_path), counts=dict(counts),
        input_bindings=inputs, rows=rows,
        preview_selection="first three CAL frames in immutable packet order, irrespective of support/outcome",
        no_pose_or_reference_accuracy_scoring=True, PRIMARY_RGB_queries=0, PRIMARY_reference_reads=0,
        original_manual_six_boundary_inputs=0, new_NN_calls=0, new_PnP_calls=0,
        wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu,
        process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        BLAS_thread_settings={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
        output_files=[dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in (bundle_path,preview_path)])
    (output_dir/"CAL_PROBE_RECEIPT.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    return receipt


def _cal_preview(preview, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(preview), 3, figsize=(16, 4*len(preview)), constrained_layout=True)
    for row, (ident, image, packet, bundle) in enumerate(preview):
        ax, color_ax, distribution_ax = axes[row]
        ax.imshow(image)
        proj = np.asarray(bundle["initial_projection_xy"])
        hull = _hull_indices(proj)
        closed = hull+[hull[0]]
        ax.plot(proj[closed,0],proj[closed,1],"--",color="white",linewidth=1.1,label="virtual initial cuboid guide")
        for m in packet["solver_input"]["visible"]:
            if m["state"] == "B":
                x,y=m["xy"]
                ax.plot(x,y,"o",mfc="none",mec="#FFD700",ms=6)
                ax.annotate(f"B{m['native_corner']}",(x,y),color="#FFD700",fontsize=8)
        # Pick the first existing query, without consulting likelihood, pose or truth.
        profiles=[q for g in bundle["groups"] for q in g["queries"]]
        q=profiles[0] if profiles else None
        if q is not None:
            xy=np.asarray(q["sample_xy"])
            ax.annotate("",xy=xy[-1],xytext=xy[0],arrowprops=dict(arrowstyle="->",color="cyan",lw=1.8))
            ax.plot(*q["center_xy"],"+",color="cyan",ms=8)
        ax.set_title(ident+"\nYellow: upstream B guide; cyan: fixed RGB query F -> B",fontsize=10)
        ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.set_xlabel("Raw image x (px)");ax.set_ylabel("Raw image y (px)")
        ax.legend(fontsize=8,loc="lower left")
        if q is not None and "foreground_probability" in q:
            r=np.asarray(q["sample_offsets_base_px"])
            color_ax.plot(r,q["foreground_probability"],color="#008B45",label="p(foreground | RGB color)")
            color_ax.axvline(0,color="gray",ls="--",lw=.8)
            color_ax.set_ylim(-.03,1.03);color_ax.set_xlabel("Fixed normal coordinate (canonical px)")
            color_ax.set_ylabel("Color foreground probability")
            color_ax.set_title(f"B{q['native_corner']} -> native {q['native_neighbor']}; fixed samples",fontsize=10)
            color_ax.grid(alpha=.25)
            distribution_ax.plot(q["offset_grid_base_px"],q["offset_probability"],color="#7651A8")
            distribution_ax.set_xlabel("F -> B offset d (canonical px)");distribution_ax.set_ylabel("Discrete normalized mass p(d)")
            distribution_ax.set_title(q["reason"],fontsize=9);distribution_ax.grid(alpha=.25)
        else:
            for a in (color_ax,distribution_ax):
                a.text(.5,.5,"Unsupported: no shared appearance or RGB profile",transform=a.transAxes,ha="center",wrap=True)
                a.set_axis_off()
    fig.suptitle("CAL input inspection only: first three frames; no reference poses or success labels\n"
                 "Correspondence is conditional on a virtual cuboid guide; query coordinates are not measured corners.",fontsize=13)
    fig.savefig(output,dpi=135,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--packets",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=cal_probe(args.source_root,args.packets,args.output_dir)
    print(json.dumps({k:result[k] for k in ("counts","wall_seconds","process_cpu_seconds","process_peak_RSS_MiB","source_sha256")},indent=2))
