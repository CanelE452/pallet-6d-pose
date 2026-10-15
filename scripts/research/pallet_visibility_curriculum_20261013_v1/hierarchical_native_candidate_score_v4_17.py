"""Opt-in native image-evidence authentication and candidate ranking.

No RGB/raw-keypoint query, GT, file IO, model fitting or pose optimization.
Initial S and non-B/I anchors are gated before observation fields. U mates
can supply registered edge geometry only; U remains a region, never a point.
The fitted support midpoint is line geometry, not a measured 2D keypoint.

Ranking is a diagnostic control: native observation energies first, global
mask only at numerical ties. It does not prove native identity, global pose
uniqueness or calibrated uncertainty. All joint guards remain caller-owned.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import numpy as np

from . import joint_ot_pnp as V3
from . import statewise_partial_BI_u_solver_v4_13 as P13
from . import statewise_archived_BI_tangent_solver_v4_15 as ARCHIVE
from . import face_aware_partial_bi_observations_v4_13d as FACE
from .interior_observation import NEIGHBORS


RGB_SOURCES = frozenset(('OBSERVED_FULL_EDGE_FACE_RGB_WITH_MASK_CONTEXT',
    'OBSERVED_LOCAL_ANCHORED_FACE_RGB_WITH_MASK_CONTEXT',
    'OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION'))
KINDS = frozenset(('BI_COMPLETE', 'PARTIAL_NORMAL', 'ARCHIVED_TANGENT', 'RGB_EDGE'))


def _states(states):
    if len(states) != 8:
        raise ValueError('Eight authoritative native states required')
    result = [V3._state(s) for s in states]
    if any(s not in ('S', 'B', 'I', 'O', 'T', 'U') for s in result):
        raise ValueError('Unknown native state')
    return result


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a nonnegative finite scalar')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be a nonnegative finite scalar')
    return value


def _digest(binding):
    if not isinstance(binding, Mapping):
        raise ValueError('Explicit image binding required')
    value = binding.get('sha256', binding.get('image_sha256'))
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase image SHA256 required')
    return value


def _line_summary(line, cfg):
    provenance = line['provenance']
    actual = provenance.get('line_record')
    if actual is not None:
        if not isinstance(actual, Mapping) or actual.get('supported') is not True:
            raise ValueError('Supported actual RGB line record required')
        if actual.get('measurement_source') not in RGB_SOURCES:
            raise ValueError('Mask-only or model-only line is not RGB evidence')
        points = np.array(actual.get('observed_points'), dtype=float, copy=True)
        mask = np.asarray(actual.get('inlier_mask'), dtype=bool)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or mask.shape != (len(points),):
            raise ValueError('Actual inlier mask must bind the observed point array')
        inliers = points[mask]
        count = len(inliers)
        rgb_count = actual.get('RGB_supported_sample_count', actual.get('rgb_supported_sample_count'))
        kind = actual.get('mask_context_kind')
        if kind is None and actual['measurement_source'] == 'OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION':
            kind = 'TARGET_BACKGROUND_OUTLINE'
        source = actual
        if (not np.array_equal(np.asarray(actual.get('normal'), float), line['normal_unit_xy'])
                or actual.get('offset') != line['offset_px']):
            raise ValueError('Actual RGB TLS coefficients differ from supplied factor')
        observed_points_present = True
    else:
        # Frozen legacy records preserve authenticated RGB guard summaries,
        # not individual pixels. Never reconstruct pixels and call them data.
        if provenance.get('legacy_mask_only_line_is_not_used_as_RGB_evidence') is not True:
            raise ValueError('Legacy line lacks explicit actual-RGB provenance')
        source = provenance
        count = source.get('inlier_count')
        rgb_count = source.get('rgb_supported_sample_count')
        kind = 'TARGET_BACKGROUND_OUTLINE'
        inliers = None
        observed_points_present = False
    for value, name in ((count, 'inlier_count'), (rgb_count, 'RGB_supported_sample_count')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < cfg['minimum_inlier_count']:
            raise ValueError('Existing actual RGB minimum support violated: ' + name)
    fraction = float(source.get('inlier_fraction', np.nan))
    span = float(source.get('inlier_span_fraction', np.nan))
    rms = float(source.get('inlier_rms_px', np.nan))
    if (not np.isfinite([fraction, span, rms]).all() or fraction < cfg['minimum_inlier_fraction']
            or span < cfg['minimum_inlier_span_fraction'] or rms < 0 or rms > cfg['maximum_line_rms_px']):
        raise ValueError('Existing inlier fraction/span/RMS guards violated')
    interval = V3._finite_array(provenance.get('actual_observed_inlier_tangent_interval_px'), (2,),
                                'actual inlier TLS tangent support')
    if interval[1] <= interval[0]:
        raise ValueError('Nondegenerate actual inlier support required')
    if inliers is not None:
        normal = line['normal_unit_xy']; tangent = np.array([-normal[1], normal[0]])
        projected = inliers @ tangent
        if not np.allclose([projected.min(), projected.max()], interval, atol=1e-7, rtol=0):
            raise ValueError('Actual point support interval differs from archived TLS support')
    if kind not in ('TARGET_BACKGROUND_OUTLINE', 'TARGET_TARGET_INTERNAL'):
        raise ValueError('Explicit actual target/background or target/target context required')
    return dict(inlier_count=int(count), RGB_supported_sample_count=int(rgb_count),
        inlier_fraction=fraction, inlier_span_fraction=span, inlier_rms_px=rms,
        actual_support_tangent_interval_px=interval, mask_context_kind=kind,
        actual_inlier_points_present=observed_points_present,
        quality=(fraction, span, int(count), -rms), actual_line_record=actual)


def authenticate_supported_native_edges(xyz, states, visible, boundary_observations, image_hw, *,
        initial_self_hidden, image_binding, angular_variance_rad2,
        angular_variance_provenance, normal_angle_correlation=0., config=None):
    """Authenticate existing partial factors and deduplicate physical edges.

    Return ``edges`` for projected_edge_residual_and_jacobian plus exact old
    normal references to replace. Other BI/normal factors are untouched. A
    supplied CAL-only angular component can be zero for preparation; the
    positive pixel/span variance floor is always added. Zero TLS RMS never
    creates a precise angle. Native association remains conditional.
    """
    states = _states(states)
    hidden = np.asarray(initial_self_hidden, bool)
    if hidden.shape != (8,):
        raise ValueError('Fixed initial self-hidden vector required')
    xyz = V3._finite_array(xyz, (8, 3), 'registered native xyz')
    image_sha = _digest(image_binding)
    image_hw = tuple(int(v) for v in image_hw)
    if len(image_hw) != 2 or min(image_hw) <= 0 or list(image_binding.get('image_hw', ())) != list(image_hw):
        raise ValueError('Bound current canvas required')
    angular_variance = _nonnegative(angular_variance_rad2, 'CAL angular variance component')
    if not isinstance(angular_variance_provenance, Mapping) or angular_variance_provenance.get('fit_partition') != 'CAL':
        raise ValueError('CAL-only angular uncertainty provenance required')
    correlation = float(normal_angle_correlation)
    if not math.isfinite(correlation) or abs(correlation) >= 1. - 1e-8:
        raise ValueError('Normal-angle correlation must give positive definite covariance')
    cfg = FACE._config(None)
    incident_normals = V3.registered_incident_normals(xyz)
    visible_records = []
    for record in V3._records(visible):
        k = V3._id(record)
        if hidden[k] or states[k] not in ('B', 'I'):
            continue
        visible_records.append(record)
    eligible, skipped, source_line_indices = [], [], {}
    for record in boundary_observations or ():
        k = V3._id(record)
        if hidden[k] or states[k] not in ('B', 'I'):
            skipped.append(dict(native_corner=k, reason='initial_S_or_non_BI_before_line_fields'))
            continue
        # The immutable general parser reads line coefficients/support. Gate
        # initial-S mates before calling it, not after it has read that line.
        retained, indices = [], []
        for line_index, line in enumerate(record['lines']):
            provenance = line.get('provenance')
            if not isinstance(provenance, Mapping):
                raise ValueError('Native-neighbor and actual RGB provenance required')
            mate = provenance.get('native_neighbor')
            if isinstance(mate, (bool, np.bool_)) or not isinstance(mate, (int, np.integer)) or not 0 <= mate < 8:
                raise ValueError('Explicit valid native neighbor required')
            if hidden[int(mate)] or states[int(mate)] == 'S':
                skipped.append(dict(native_corner=k, line_index=line_index, native_neighbor=int(mate),
                    reason='initial_S_mate_before_line_support_fields'))
                continue
            retained.append(line); indices.append(line_index)
        if retained:
            eligible.append({**record, 'lines': retained})
            source_line_indices[k] = indices
    normalized = ARCHIVE._boundary_inputs(eligible, states, image_hw, xyz)
    ARCHIVE.validate_archived_predictions(normalized, visible_records, config)
    original = {V3._id(r): r for r in visible_records}
    solver_cfg = P13._configuration(config)
    for record in normalized:
        if record['legacy_fallback_B_record']:
            source = original[record['native_corner']]
            expected_weight = float(source.get('fit_reliability_weight', solver_cfg['state_loss_weights']['B']))
            if record['fit_reliability_weight'] != expected_weight:
                raise ValueError('Legacy RGB factor reliability differs from immutable visible reference')
            raw = record['association_provenance'].get('original_raw_xy')
            if raw is not None and not np.array_equal(np.asarray(raw, float), np.asarray(source['xy'], float)):
                raise ValueError('Legacy raw reference differs from immutable fallback prediction')
    grouped = {}
    for record in normalized:
        k = record['native_corner']
        if _digest(record['image_binding']) != image_sha or list(record['image_binding']['image_hw']) != list(image_hw):
            raise ValueError('Partial factor image/canvas differs from bound current image')
        for retained_index, line in enumerate(record['lines']):
            line_index = source_line_indices[k][retained_index]
            provenance = line.get('provenance')
            if not isinstance(provenance, Mapping):
                raise ValueError('Native-neighbor and actual RGB provenance required')
            mate = provenance.get('native_neighbor')
            if isinstance(mate, (bool, np.bool_)) or not isinstance(mate, (int, np.integer)) or not 0 <= mate < 8:
                raise ValueError('Explicit valid native neighbor required')
            mate = int(mate)
            if hidden[mate] or states[mate] == 'S':
                skipped.append(dict(native_corner=k, line_index=line_index, native_neighbor=mate,
                                    reason='initial_S_mate_before_line_support_fields'))
                continue
            if mate not in NEIGHBORS[k] or np.linalg.norm(xyz[mate]-xyz[k]) <= 0:
                raise ValueError('Registered physical incident edge required')
            shared_faces = sum(any(np.allclose(n, other, atol=1e-8, rtol=0)
                                   for other in incident_normals[mate]) for n in incident_normals[k])
            if shared_faces != 2:
                raise ValueError('Native endpoints must share exactly two registered faces')
            summary = _line_summary(line, cfg)
            actual = summary.pop('actual_line_record')
            pair = tuple(sorted((k, mate)))
            if actual is not None and (actual.get('native_neighbor') != mate or actual.get('native_corner') != k
                    or sorted(actual.get('endpoints', ())) != list(pair)):
                raise ValueError('Actual RGB line native identity differs from factor')
            physical = provenance.get('physical_native_edge')
            if physical is not None and sorted(physical) != list(pair):
                raise ValueError('Shared line physical-edge identity differs from factor')
            grouped.setdefault(pair, []).append(dict(native_corner=k, native_neighbor=mate,
                line_index=line_index, line=line, summary=summary,
                fit_reliability_weight=record['fit_reliability_weight']))
    edges, unresolved = [], []
    for pair, observations in sorted(grouped.items()):
        contexts = {v['summary']['mask_context_kind'] for v in observations}
        if len(contexts) != 1:
            unresolved.append(dict(physical_native_edge=list(pair),
                reason='same_physical_edge_has_conflicting_actual_mask_contexts', old_normals_retained=True))
            continue
        chosen = max(observations, key=lambda v: v['summary']['quality'])
        line, summary = chosen['line'], chosen['summary']
        normal = line['normal_unit_xy']; offset = line['offset_px']
        interval = summary['actual_support_tangent_interval_px']
        tangent = np.array([-normal[1], normal[0]])
        support_anchor = tangent * float(interval.mean()) - normal * offset
        refs = [dict(native_corner=v['native_corner'], line_index=v['line_index']) for v in observations]
        # Cached search-window saturation can create an artificially exact
        # TLS angle. A one-pixel/span floor is retained even when RMS is zero;
        # it is an engineering lower bound, not a fitted coverage guarantee.
        pixel_floor_variance = (1. / float(interval[1]-interval[0]))**2
        effective_angular_variance = angular_variance + pixel_floor_variance
        edges.append(dict(native_corner=chosen['native_corner'], native_neighbor=chosen['native_neighbor'],
            normal_unit_xy=normal.tolist(), offset_px=float(offset), support_anchor_xy=support_anchor.tolist(),
            normal_variance_px2=max(float(v['line']['normal_variance_px2']) for v in observations),
            angular_variance_rad2=effective_angular_variance, normal_angle_correlation=correlation,
            actual_support_tangent_interval_px=interval.tolist(),
            support_extrapolation_px=float(line['support_extrapolation_px']),
            fit_reliability_weight=min(v['fit_reliability_weight'] for v in observations),
            physical_native_edge=list(pair), replaces_normal_factor_refs=refs,
            image_binding=deepcopy(dict(image_binding)), angular_variance_provenance=deepcopy(dict(angular_variance_provenance)),
            provenance=dict(mask_context_kind=summary['mask_context_kind'],
                source='authenticated existing RGB partial-edge evidence',
                support_anchor_is_fitted_support_midpoint_not_actual_pixel_or_KP=True,
                actual_inlier_points_present=summary['actual_inlier_points_present'],
                inlier_count=summary['inlier_count'], RGB_supported_sample_count=summary['RGB_supported_sample_count'],
                inlier_rms_px=summary['inlier_rms_px'], inlier_fraction=summary['inlier_fraction'],
                inlier_span_fraction=summary['inlier_span_fraction'],
                physical_edge_observation_count=len(observations), shared_edge_deduplicated=True,
                covariance_policy='max supplied normal variance, min fixed reliability; shared reuse inflation not applied twice',
                covariance_is_newly_calibrated_total_error=False, native_edge_identity_is_conditional=True,
                supplied_CAL_angular_variance_rad2=angular_variance,
                one_pixel_over_observed_span_angular_variance_floor_rad2=pixel_floor_variance,
                angular_variance_rule='supplied CAL variance + (one pixel / actual TLS support span)^2',
                cached_search_saturation_independently_certified_absent=False,
                neighbor_supplies_model_geometry_only=states[chosen['native_neighbor']] not in ('B', 'I'),
                U_point_or_local_line_measurement_created=False, shared_RGB_independence_claimed=False)))
    return dict(schema='AUTHENTICATED_NATIVE_RGB_EDGES_V4_17', edges=edges,
        replaces_normal_factor_refs=[r for e in edges for r in e['replaces_normal_factor_refs']],
        skipped=skipped, unresolved_edges=unresolved, states_unchanged=True,
        S_coordinate_reads=0, S_local_queries=0, U_coordinate_reads=0, U_local_queries=0,
        RGB_queries=0, new_point_measurements=0, GT_inputs=False,
        native_identity_or_physical_visibility_proved=False, uncertainty_probability_guaranteed=False)


def _observation_key(record, states):
    k = V3._id(record)
    if states[k] not in ('B', 'I'):
        raise ValueError('Native energy must gate S/U/O/T before cost fields')
    kind = record.get('kind')
    if kind not in KINDS:
        raise ValueError('Explicit complete/partial/tangent/RGB-edge native energy kind required')
    if kind == 'RGB_EDGE':
        mate = record.get('native_neighbor')
        if isinstance(mate, bool) or not isinstance(mate, (int, np.integer)) or mate not in NEIGHBORS[k] or states[mate] == 'S':
            raise ValueError('Non-S registered edge mate required; no mate coordinate is read')
        return (kind, *sorted((k, int(mate))))
    line = record.get('line_index', 0)
    if isinstance(line, bool) or not isinstance(line, (int, np.integer)) or line < 0:
        raise ValueError('Valid line index required')
    return (kind, k, int(line))


def _numeric_tolerance(a, b):
    return 32 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(a), np.abs(b)))


def select_hierarchical_native_candidates(states, candidates):
    """Pareto audit; minimum summed native energy, mask only numeric ties.

    Candidate schema: name/family, accepted=True,
    all_joint_constraints_satisfied=True, native_costs=[{kind,native_corner,
    native_neighbor? or line_index?,weighted_robust_cost}], global_mask_cost.
    Energies already include fixed reliability exactly once. Their covariance
    must come from the caller's frozen observation model. No costs are fitted
    here, and rejected candidates are gated before their energy fields.
    """
    states = _states(states)
    rows, rejected, basis = [], [], None
    for index, candidate in enumerate(candidates):
        if candidate.get('accepted') is not True or candidate.get('all_joint_constraints_satisfied') is not True:
            rejected.append(dict(index=index, reason='not_an_accepted_joint_feasible_candidate'))
            continue
        costs = {}
        for record in candidate['native_costs']:
            key = _observation_key(record, states)
            if key in costs:
                raise ValueError('One energy per native observation/physical RGB edge required')
            value = record['weighted_robust_cost']
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError('Finite nonnegative already-weighted native energy required')
            costs[key] = float(value)
        keys = tuple(sorted(costs))
        if basis is None:
            basis = keys
        if keys != basis:
            raise ValueError('Every candidate must use the same complete native evidence basis')
        mask = candidate.get('global_mask_cost')
        if mask is not None and (not math.isfinite(float(mask)) or float(mask) < 0):
            raise ValueError('Nonnegative finite mask cost or explicit unavailable None required')
        rows.append(dict(index=index, name=candidate['name'], family=candidate['family'],
            native_cost_vector=np.array([costs[k] for k in keys]), native_energy=math.fsum(costs.values()),
            global_mask_cost=None if mask is None else float(mask)))
    domination = []
    for b in rows:
        by = []
        for a in rows:
            if a is b:
                continue
            tolerance = _numeric_tolerance(a['native_cost_vector'], b['native_cost_vector'])
            if np.all(a['native_cost_vector'] <= b['native_cost_vector'] + tolerance) and np.any(a['native_cost_vector'] < b['native_cost_vector'] - tolerance):
                by.append(a['index'])
        domination.append(dict(index=b['index'], dominated_by=by))
    dominated = {d['index'] for d in domination if d['dominated_by']}
    frontier = [r for r in rows if r['index'] not in dominated]
    selected, ties = None, []
    if frontier:
        minimum = min(r['native_energy'] for r in frontier)
        ties = [r for r in frontier if abs(r['native_energy']-minimum) <= float(_numeric_tolerance(r['native_energy'], minimum))]
        selected = min(ties, key=lambda r: (float('inf') if r['global_mask_cost'] is None else r['global_mask_cost'], r['index']))
    families = sorted({str(r['family']) for r in frontier})
    output_rows = [{**r, 'native_cost_vector': r['native_cost_vector'].tolist()} for r in rows]
    return dict(schema='HIERARCHICAL_NATIVE_CANDIDATE_SCORE_V4_17',
        selected_index=None if selected is None else selected['index'],
        selected_name=None if selected is None else selected['name'],
        selected_family=None if selected is None else selected['family'],
        candidates=output_rows, rejected_candidates=rejected,
        native_evidence_basis=[list(k) for k in (basis or ())], pareto_domination=domination,
        pareto_frontier_indices=[r['index'] for r in frontier],
        mask_tie_break_indices=[r['index'] for r in ties],
        mask_used_only_at_machine_numeric_native_energy_ties=True,
        native_energy_rule='minimum summed native weighted robust energy among native Pareto frontier',
        machine_tolerance_is_not_an_uncertainty_or_fitted_threshold=True,
        ambiguity=dict(cross_family_native_frontier=len(families)>1, frontier_families=families,
            native_evidence_absent=not bool(basis), global_uniqueness_proved=False),
        representative_is_not_uniqueness_certificate=True, diagnostic_control_not_proved_default=True,
        independent_RGB_or_Gaussian_information_claimed=False,
        old_best_pose_fallback=False, GT_inputs=False, frame_or_partition_parameter_routing=False,
        S_terms=0, U_point_or_line_terms=0, image_queries=0, pose_fits=0)
