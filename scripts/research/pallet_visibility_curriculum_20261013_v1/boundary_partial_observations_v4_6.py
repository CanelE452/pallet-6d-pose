"""Pure extraction of supported fallback-B line observations from frozen records.

This performs no RGB query, mask inference, line refit, GT access or calibration.
An observed incident edge is not a complete corner measurement. Its normal
factor replaces that fallback B's raw 2D likelihood in the v4.6 joint solver.
The tangent range is the hull of the actual inlier interval and a raw-centered
interval borrowed from the existing CAL non-S maximum localization model.
It is deliberately broad and is not a calibrated B coverage probability.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

import numpy as np

from . import joint_ot_pnp as V3
from .truncation_interval_v4_2 import load_truncation_interval_model


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' must be a lowercase SHA256')
    return value


def _records_by_id(records):
    result = {}
    for record in records:
        k = V3._id(record)
        if k in result:
            raise ValueError('Duplicate native record')
        result[k] = record
    return result


def _supported_ray(ray, native_id, cfg):
    if not isinstance(ray, Mapping) or ray.get('supported') is not True:
        return None
    # The original TARGET_BACKGROUND record validates only mask occupancy.
    # Use the new already-remeasured exterior + actual RGB support exclusively.
    if ray.get('native_corner') != native_id or ray.get('measurement_source') != 'OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION':
        return None
    rgb_count = ray.get('rgb_supported_sample_count')
    thresholds = ray.get('rgb_support_thresholds', {})
    if (isinstance(rgb_count, bool) or not isinstance(rgb_count, (int, np.integer))
            or rgb_count < cfg['minimum_inlier_count']
            or any(thresholds.get(key) != cfg[key] for key in
                   ('minimum_gradient', 'minimum_gradient_alignment', 'minimum_color_transition'))):
        return None
    mate = ray.get('native_neighbor')
    if isinstance(mate, bool) or not isinstance(mate, (int, np.integer)) or not 0 <= mate < 8 or mate == native_id:
        return None
    try:
        normal = V3._finite_array(ray.get('normal'), (2,), 'observed line normal')
        offset = float(ray.get('offset', np.nan))
        rms = float(ray.get('inlier_rms_px', np.nan))
        angle = float(ray.get('angle_change_deg', np.nan))
        fraction = float(ray.get('inlier_fraction', np.nan))
        span = float(ray.get('inlier_span_fraction', np.nan))
        points = np.asarray(ray.get('observed_points'), dtype=float)
        inliers = np.asarray(ray.get('inlier_mask'), dtype=bool)
    except (TypeError, ValueError):
        return None
    if (not np.isclose(np.linalg.norm(normal), 1., atol=1e-8, rtol=0)
            or not np.isfinite([offset, rms, angle, fraction, span]).all()
            or points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all()
            or inliers.shape != (len(points),)
            or int(inliers.sum()) < cfg['minimum_inlier_count']
            or rms < 0 or rms > cfg['maximum_line_rms_px']
            or abs(angle) > cfg['maximum_direction_change_deg']
            or fraction < cfg['minimum_inlier_fraction']
            or span < cfg['minimum_inlier_span_fraction']):
        return None
    return dict(native_neighbor=int(mate), normal=normal.copy(), offset=offset,
                points=points[inliers].copy(), inlier_count=int(inliers.sum()),
                inlier_fraction=fraction, inlier_span_fraction=span,
                inlier_rms_px=rms, angle_change_deg=angle,
                rgb_supported_sample_count=int(rgb_count), ray=ray)


def observe_single_supported_b_lines(correction, geometry, states, fit, *,
                                     localization_model, nonself_scale_px,
                                     image_binding):
    """Keep one/two supported incident rays, even when no intersection passed.

``fit`` is the immutable, original CAL-effective BI observation list. We do
not read correction.raw_points or any image. State/initial-S/role gates precede
all per-corner measurement, ray or coordinate field reads. Supported rays in
legacy correction, expanded recovery and recovery remeasurements are pooled;
at most the strongest already-supported ray per native incident edge is kept.
"""
    if len(states) != 8 or len(geometry['initial_self_hidden']) != 8 or len(geometry['initial_roles']) != 8:
        raise ValueError('Eight authoritative states and initial geometry roles required')
    states = [V3._state(s) for s in states]
    if not isinstance(image_binding, Mapping):
        raise ValueError('Caller image binding required')
    binding = deepcopy(dict(image_binding))
    _sha(binding.get('sha256'), 'image binding')
    if tuple(binding.get('image_hw', ())) != tuple(geometry['image_hw']):
        raise ValueError('Image and initial geometry canvas differ')
    _sha(correction.get('image_pixel_sha256'), 'correction pixel binding')
    if ('image_pixel_sha256' in binding
            and binding['image_pixel_sha256'] != correction['image_pixel_sha256']):
        raise ValueError('Caller correction pixel binding differs')
    binding['correction_image_pixel_sha256'] = correction['image_pixel_sha256']
    if not isinstance(localization_model, Mapping):
        raise ValueError('Existing CAL localization model dictionary required; no file IO here')
    model = load_truncation_interval_model(localization_model)
    scale = float(nonself_scale_px)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Frozen positive non-S prediction scale required')
    radius = float(model['radius_ratio']) * scale
    cfg = correction['configuration']
    margin = float(cfg['along_start_px'] + cfg['boundary_context_px'])
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Original finite nonnegative support margin required')
    base = _records_by_id(correction['records'])
    recovery = _records_by_id(correction.get('boundary_recovery_records', []))
    fitted = _records_by_id(fit)
    observations, audits = [], []
    for k, state in enumerate(states):
        audit = dict(native_corner=k, state=state, accepted_partial=False,
                     ray_fields_read=False, coordinate_read=False, RGB_queries=0)
        audits.append(audit)
        if geometry['initial_self_hidden'][k] or state != 'B' or geometry['initial_roles'][k] != 'B':
            audit['reason'] = 'initial_S_or_non_fallback_boundary_excluded_before_fields'
            continue
        record = base.get(k)
        measurement = fitted.get(k)
        if (record is None or measurement is None or record.get('corrected')
                or measurement.get('source') != 'VISIBLE_BACKBONE_FALLBACK'):
            audit['reason'] = 'existing_complete_B_or_missing_fallback_preserved'
            continue
        q = V3._finite_array(measurement['xy'], (2,), 'fallback B coordinate')
        covariance = V3._finite_array(measurement['fit_covariance_px2'], (2, 2), 'fixed old CAL fit covariance')
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
            raise ValueError('Fixed CAL fit covariance must be symmetric')
        np.linalg.cholesky(covariance)
        reliability = float(measurement['fit_reliability_weight'])
        if not np.isfinite(reliability) or reliability <= 0:
            raise ValueError('Fixed positive reliability required')
        audit.update(coordinate_read=True, ray_fields_read=True)
        candidates = [(r, 'legacy_correction') for r in record.get('rays', [])]
        rec = recovery.get(k, {})
        candidates += [(r, 'expanded_boundary_recovery') for r in rec.get('expanded_rays', [])]
        candidates += [(a.get('remeasured_ray'), 'observed_line_recentering')
                       for a in rec.get('observed_line_recentering', [])]
        chosen = {}
        for ray, origin in candidates:
            observed = _supported_ray(ray, k, cfg)
            if observed is None:
                continue
            observed['origin'] = origin
            quality = (observed['inlier_count'], observed['inlier_fraction'],
                       observed['inlier_span_fraction'], -observed['inlier_rms_px'],
                       -abs(observed['angle_change_deg']))
            mate = observed['native_neighbor']
            if mate not in chosen or quality > chosen[mate][0]:
                chosen[mate] = (quality, observed)
        if not chosen:
            audit['reason'] = 'no_existing_supported_actual_boundary_line'
            continue
        # A cuboid silhouette corner has at most two incident hull rays. Avoid
        # accidental overcounting of recovery repetitions of the same edge.
        if len(chosen) > 2:
            raise ValueError('More than two independent incident boundary edges supplied')
        count = len(chosen)
        lines = []
        for mate, (_, observed) in sorted(chosen.items()):
            n = observed['normal']; tangent = np.array([-n[1], n[0]])
            s = observed['points'] @ tangent
            actual_interval = [float(s.min()), float(s.max())]
            raw_tangent = float(tangent @ q)
            raw_interval = [raw_tangent - radius, raw_tangent + radius]
            interval = [min(actual_interval[0], raw_interval[0]), max(actual_interval[1], raw_interval[1])]
            normal_variance = count * float(n @ covariance @ n)
            lines.append(dict(normal_unit_xy=n.tolist(), offset_px=observed['offset'],
                normal_variance_px2=normal_variance, support_tangent_interval_px=interval,
                support_extrapolation_px=margin,
                provenance=dict(native_neighbor=mate, record_origin=observed['origin'],
                    actual_observed_inlier_tangent_interval_px=actual_interval,
                    raw_tangent_CAL_borrowed_interval_px=raw_interval,
                    tangent_domain_rule='hull(actual inliers, raw tangent +/- borrowed CAL nonS maximum)',
                    inlier_count=observed['inlier_count'], inlier_rms_px=observed['inlier_rms_px'],
                    inlier_fraction=observed['inlier_fraction'], inlier_span_fraction=observed['inlier_span_fraction'],
                    angle_change_deg=observed['angle_change_deg'],
                    rgb_supported_sample_count=observed['rgb_supported_sample_count'],
                    legacy_mask_only_line_is_not_used_as_RGB_evidence=True,
                    normal_variance_rule='line_count * normal.T @ fixed_original_CAL_fit_Sigma @ normal',
                    variance_count_inflation=count, new_boundary_calibration=False,
                    edge_samples_are_not_corner_coordinates=True)))
        # Display only: the orthogonal projection loses no tangent information
        # because its tangent remains the raw reference. It is not a complete
        # measured KP and is never a PnP correspondence or a point objective.
        n = np.asarray(lines[0]['normal_unit_xy']); d = lines[0]['offset_px']
        display = q - n * (n @ q + d)
        observations.append(dict(native_corner=k, state='B', lines=lines,
            image_binding=deepcopy(binding), fit_reliability_weight=reliability,
            replaces_raw_B_likelihood=True,
            association_provenance=dict(initial_native_boundary_association_is_conditional=True,
                record_source='already_supported_actual_RGB_boundary_lines',
                original_measurement_source='VISIBLE_BACKBONE_FALLBACK',
                original_raw_xy=q.tolist(), normal_projected_display_xy=display.tolist(),
                display_coordinate_is_full_2D_measurement=False, no_new_image_queries=True,
                CAL_localization_model_sha256=model['model_sha256'],
                borrowed_tangent_radius_px=radius, tangent_range_is_probability=False,
                shared_RGB_raw_and_line_independence_claimed=False)))
        audit.update(accepted_partial=True, reason='supported_B_normal_plus_broad_corner_tangent_range',
                     lines=count, selected_neighbor_ids=sorted(chosen),
                     normal_projected_display_xy=display.tolist(),
                     full_2D_corrected_point_created=False)
    return dict(schema='FALLBACK_B_PARTIAL_LINE_OBSERVATIONS_V4_6',
        boundary_observations=observations, records=audits, states_unchanged=True,
        S_coordinate_reads=0, S_ray_reads=0, RGB_queries=0, new_line_fits=0,
        GT_inputs=False, localization_model_sha256=model['model_sha256'],
        borrowed_tangent_radius_px=radius, unchanged_nonself_prediction_scale_px=scale,
        normal_variance_calibrated=False, B_specific_tangent_coverage_calibrated=False,
        raw_B_joint_likelihood_replaced_not_counted_as_independent_measurement=True)


observe_fallback_B_lines = observe_single_supported_b_lines
