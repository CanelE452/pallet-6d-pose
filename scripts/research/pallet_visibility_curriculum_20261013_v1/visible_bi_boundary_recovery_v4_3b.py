"""Versioned observed-support guard around immutable boundary recovery v4_3.

A line intersection may extrapolate at most the original along-start distance
plus boundary-context margin beyond each observed inlier interval. Unsupported
intersections leave the original B measurement and its covariance unchanged.
"""
from copy import deepcopy

import numpy as np

from . import common as C
from . import visible_bi_boundary_recovery_v4_3 as ORIGINAL
from . import visible_bi_correction_v2 as LEGACY


DEFAULT_CONFIG = deepcopy(ORIGINAL.DEFAULT_CONFIG)


def _support_interval_guard(intersection_xy, selected_rays, maximum_extrapolation_px):
    """Audit tangent extrapolation from actual inlier support, on both rays."""
    candidate = np.asarray(intersection_xy, float)
    records = []
    valid = candidate.shape == (2,) and np.isfinite(candidate).all() and len(selected_rays) == 2
    for ray in selected_rays:
        normal = np.asarray(ray.get("normal", []), float)
        observed = np.asarray(ray.get("observed_points", []), float)
        inliers = np.asarray(ray.get("inlier_mask", []), bool)
        row = dict(native_neighbor=ray.get("native_neighbor"),
                   edge=ray.get("edge"), passed=False)
        records.append(row)
        if (not ray.get("supported") or normal.shape != (2,)
                or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0
                or observed.ndim != 2 or observed.shape[1] != 2
                or not np.isfinite(observed).all() or inliers.shape != (len(observed),)
                or inliers.sum() < 2 or candidate.shape != (2,) or not np.isfinite(candidate).all()):
            row["reason"] = "missing_finite_observed_inlier_support"
            valid = False
            continue
        normal = normal/np.linalg.norm(normal)
        tangent = np.array([-normal[1], normal[0]])
        support = observed[inliers]@tangent
        low, high = float(support.min()), float(support.max())
        coordinate = float(candidate@tangent)
        gap = max(low-coordinate, 0., coordinate-high)
        passed = gap <= maximum_extrapolation_px
        row.update(tangent_unit=tangent.tolist(), observed_inlier_count=int(inliers.sum()),
                   observed_inlier_interval_px=[low, high], intersection_tangent_coordinate_px=coordinate,
                   extrapolation_px=gap, maximum_extrapolation_px=maximum_extrapolation_px,
                   support_slack_px=maximum_extrapolation_px-gap, passed=bool(passed),
                   reason="within_observed_support_margin" if passed else "intersection_extrapolates_observed_support")
        valid = valid and passed
    return dict(passed=bool(valid), maximum_extrapolation_px=maximum_extrapolation_px,
                intersection_xy=candidate.tolist(), rays=records)


def recover_visible_boundary(image_bgr, raw_points, K, initial_pose, states, *,
                             pallet_mask, base_correction, native_xyz=None, config=None):
    """Same public contract as v4_3, with a fixed observed-segment guard.

Recovery uses v4_3 unchanged. The only added decision concerns excessive
tangent extrapolation; rejection restores the exact supplied base B result.
No annotation, pose fitting, or additional image query is performed here.
"""
    result = ORIGINAL.recover_visible_boundary(image_bgr, raw_points, K, initial_pose, states,
        pallet_mask=pallet_mask, base_correction=base_correction, native_xyz=native_xyz, config=config)
    cfg = base_correction["configuration"]
    maximum = float(cfg["along_start_px"] + cfg["boundary_context_px"])
    old_records = {int(r["native_corner"]): r for r in base_correction["records"]}
    old_measurements = {int(r["native_corner"]): r for r in base_correction["visible_measurements"]}
    record_indices = {int(r["native_corner"]): i for i, r in enumerate(result["records"])}
    measurement_indices = {int(r["native_corner"]): i for i, r in enumerate(result["visible_measurements"])}
    audits = []
    rejected = []
    for row in result["boundary_recovery_records"]:
        if not row["recovered"]:
            continue
        k = int(row["native_corner"])
        audit = _support_interval_guard(row["intersection_xy"], row["selected_rays"], maximum)
        audit["native_corner"] = k
        audit["base_measurement_preserved"] = not audit["passed"]
        row["boundary_recovery_support_guard"] = deepcopy(audit)
        audits.append(audit)
        if audit["passed"]:
            continue
        rejected.append(k)
        audit["rejected_intersection_xy"] = deepcopy(row["intersection_xy"])
        row.update(recovered=False, recovery_before_support_guard=True,
                   reason="intersection_extrapolates_beyond_observed_inlier_support",
                   rejected_intersection_xy=deepcopy(row["intersection_xy"]),
                   rejected_recovered_measurement_xy=deepcopy(row["measured_xy"]),
                   rejected_conditional_covariance=deepcopy(row["conditional_covariance"]),
                   conditional_covariance=None,
                   measured_xy=deepcopy(base_correction["measured_points"][k]),
                   covariance_px2=deepcopy(old_measurements[k]["covariance_px2"]),
                   applied_shift_px=0., blend_alpha=0.)
        # Preserve row/measurement/covariance/source verbatim, including every
        # original fallback diagnostic. Rejection evidence lives separately.
        result["records"][record_indices[k]] = deepcopy(old_records[k])
        result["visible_measurements"][measurement_indices[k]] = deepcopy(old_measurements[k])
        result["measured_points"][k] = deepcopy(base_correction["measured_points"][k])
    result.update(schema="visible_BI_measurements_boundary_recovery_v4_3b",
        recovered_native_ids=[r["native_corner"] for r in result["boundary_recovery_records"] if r["recovered"]],
        boundary_recovery_support_guard=dict(
            rule="each_ray_tangent_interval_extrapolation <= original_along_start_px + original_boundary_context_px",
            along_start_px=cfg["along_start_px"], boundary_context_px=cfg["boundary_context_px"],
            maximum_extrapolation_px=maximum, corner_records=audits, rejected_native_ids=rejected,
            new_image_queries=0, GT_inputs=False, geometry_is_observed_inlier_segment=True))
    result["provenance"].update(recovery_immutable_parent_version="visible_bi_boundary_recovery_v4_3",
        recovery_observed_support_interval_guard=True, recovery_support_margin_derived_from_original_sampling=True,
        recovery_support_guard_new_fitted_parameters=False, recovery_support_guard_GT_inputs=False,
        rejected_recovery_preserves_exact_base_B=True)
    return C.finite(result)


def correct_visible_bi(image_bgr, raw_points, K, initial_pose, states, *,
                       pallet_mask, config=None, native_xyz=None, recovery_config=None):
    """Opt-in RGB adapter; legacy defaults elsewhere are never replaced.

``config`` belongs to the original B/I corrector; ``recovery_config`` belongs
to v4_3 recovery. This composition does not fit or transfer total-error weights.
"""
    base = LEGACY.correct_visible_bi(image_bgr, raw_points, K, initial_pose, states,
        pallet_mask=pallet_mask, config=config, native_xyz=native_xyz)
    return recover_visible_boundary(image_bgr, raw_points, K, initial_pose, states,
        pallet_mask=pallet_mask, base_correction=base, native_xyz=native_xyz, config=recovery_config)
