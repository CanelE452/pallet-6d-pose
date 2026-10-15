"""Pure, opt-in complete-B/I actual-line authentication and objective prototype.

No image/file/label is read and no corner, pose or covariance is fitted here.
Every physical-edge alternative is retained. Initial S/non-BI gates precede
coordinate/copy operations. A new CAL observation-family model is mandatory;
the old partial-only CAL model is deliberately rejected.

The explicit prototype overlap policy withholds any cross-edge shared support.
That policy is provisional, can reject genuine junction pixels, and must not
be fitted/adopted before independent coverage review. Authenticated archives
retain all such alternatives and distinguish exact-set from partial overlap.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import copy, deepcopy
from itertools import combinations, product
import hashlib

import numpy as np

from . import image_chain_mixture_problem_v4_18 as OLD
from .interior_observation import NEIGHBORS

N, M, OBS = OLD.N, OLD.M, OLD.OBS
NOISE_OBSERVATION_FAMILY = 'ALL_SUPPORTED_COMPLETE_AND_PARTIAL_BI_NMS17C_NATIVE_EDGE_MIXTURES_V20'
PROTOTYPE_OVERLAP_POLICY = 'ABSTAIN_ANY_CROSS_EDGE_INLIER_LATTICE_OVERLAP_PROTOTYPE'
TANGENT_VARIANCE_INFLATION = 4.


def _id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 8:
        raise ValueError('Valid explicit native endpoint ID required')
    return int(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef'for c in value)


def _pixel_digest(points):
    ordered = np.array(sorted(map(tuple, np.round(points, 6))), dtype='<f8')
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _archive_measurements(base, records, hidden):
    parsed = {m['native_corner']: m for m in base.measurements}
    result = {}
    for record in records:
        k = _id(record['native_corner'])
        if hidden[k] or base.states[k] not in ('B', 'I'):
            continue
        if k in result or k not in parsed:
            raise ValueError('Unique current frozen BI archive required')
        m = parsed[k]
        if (not isinstance(record.get('source'), str) or not record['source']
                or not np.array_equal(np.asarray(record['xy'], float), m['xy'])
                or not np.array_equal(np.asarray(record['covariance_px2'], float), m['observed_covariance_px2'])):
            raise ValueError('Exact complete measurement q, observed covariance and actual source required')
        fit = np.asarray(record.get('fit_covariance_px2', base.cfg['covariance_scale']*m['observed_covariance_px2']
            + np.asarray(base.cfg['state_BI_total_error_floor_px2'][base.states[k]])), float)
        weight = float(record.get('fit_reliability_weight', base.cfg['state_loss_weights'][base.states[k]]))
        if not np.array_equal(fit, m['covariance_px2']) or weight != m['fit_reliability_weight']:
            raise ValueError('Frozen effective covariance/reliability mismatch')
        result[k] = dict(native_corner=k, state=base.states[k], original_measurement_source=record['source'],
            original_observed_xy=m['xy'].tolist(),
            original_observed_covariance_px2=m['observed_covariance_px2'].tolist(),
            original_effective_fit_covariance_px2=m['covariance_px2'].tolist(),
            original_fit_reliability_weight=weight, source_record=deepcopy(record),
            archive_is_existing_measurement_not_renamed_original_backbone=True)
    if set(result) != set(parsed):
        raise ValueError('Every authoritative BI measurement must be bound')
    return result


def _old_normal_references(base, hidden):
    result = defaultdict(list)
    for b in base.boundary:
        k = _id(b['native_corner'])
        if hidden[k] or base.states[k] not in ('B', 'I'):
            continue
        for index, line in enumerate(b['lines']):
            mate = line.get('provenance', {}).get('native_neighbor')
            if isinstance(mate, (bool, np.bool_)) or not isinstance(mate, (int, np.integer)) or not 0 <= mate < 8:
                continue  # Unidentified old normals stay exactly as they were.
            mate = int(mate)
            if hidden[mate] or base.states[mate] == 'S':
                continue
            if mate not in NEIGHBORS[k]:
                continue
            result[tuple(sorted((k, mate)))].append(dict(native_corner=k, line_index=index,
                reliability=float(b['fit_reliability_weight'])))
    return result


def _component(chain, k, mate, pair, guide, cfg, image_hw, vn, va, rho, noise):
    if (chain.get('native_corner') != k or chain.get('native_neighbor') != mate
            or sorted(chain.get('native_edge', ())) != list(pair)
            or chain.get('measurement_source') != 'ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT'
            or chain.get('invented_bridge_points') != 0
            or chain.get('corrected_endpoint_normal_guard_pass') is not True):
        raise ValueError('Bound actual NMS native identity and original endpoint guard required')
    xy = np.asarray(chain['observed_points'], float)
    mask = np.asarray(chain['inlier_mask'], bool)
    if xy.ndim != 2 or xy.shape[1] != 2 or mask.shape != (len(xy),) or not np.isfinite(xy).all():
        raise ValueError('Actual finite observed pixels and exact inlier mask required')
    actual = xy[mask]
    quality = np.asarray([chain['inlier_fraction'], chain['inlier_span_fraction'], chain['inlier_rms_px']], float)
    if (not np.isfinite(quality).all() or quality[2] < 0
            or len(actual) < cfg['minimum_inlier_count'] or chain['inlier_count'] != len(actual)
            or quality[0] < cfg['minimum_inlier_fraction'] or quality[1] < cfg['minimum_inlier_span_fraction']
            or quality[2] > cfg['maximum_line_rms_px']
            or not np.isclose(quality[0], mask.mean(), atol=1e-10, rtol=0)):
        raise ValueError('Original count/fraction/span/RMS guards required')
    n, c = np.asarray(chain['normal'], float), float(chain['offset'])
    if n.shape != (2,) or not np.isfinite(n).all() or not np.isfinite(c) or not np.isclose(n@n, 1., atol=1e-7, rtol=0):
        raise ValueError('Finite unit weighted-TLS coefficients required')
    seed_normal = np.array([-guide[1], guide[0]])
    angle = float(np.degrees(np.arccos(np.clip(abs(n@seed_normal), 0., 1.))))
    reported_angle = float(chain['angle_change_deg'])
    if (not np.isfinite(reported_angle) or reported_angle < 0
            or reported_angle > cfg['maximum_direction_change_deg']
            or not np.isclose(angle, reported_angle, atol=1e-5, rtol=0)):
        raise ValueError('Original direction guard must bind the ORIGINAL observer model guide')
    endpoint_residuals = np.array(list(chain['corrected_endpoint_normal_residual_px'].values()), float)
    if (not np.isfinite(endpoint_residuals).all() or np.any(endpoint_residuals < 0)
            or np.any(endpoint_residuals > cfg['along_start_px']+cfg['boundary_context_px'])):
        raise ValueError('Original corrected-endpoint normal guard must remain unchanged')
    h, w = image_hw
    if np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h):
        raise ValueError('Actual pixels must lie on this image canvas')
    # Recheck the archived actual RGB/mask samples, without re-querying RGB.
    samples = {}
    for key in ('RGB_gradient_energy', 'RGB_alignment', 'RGB_contrast', 'mask_plus', 'mask_minus'):
        values = np.asarray(chain[key], float)
        if values.shape != (len(xy),) or not np.isfinite(values).all():
            raise ValueError('Actual RGB/context arrays must bind every observed sample')
        samples[key] = values
    kind = chain['mask_context_kind']
    mp, mm = samples['mask_plus'], samples['mask_minus']
    if np.any(mp < 0) or np.any(mp > 1) or np.any(mm < 0) or np.any(mm > 1):
        raise ValueError('Bound mask occupancy samples required')
    if kind == 'TARGET_TARGET_INTERNAL':
        occupancy = (mp >= cfg['minimum_inside']) & (mm >= cfg['minimum_inside'])
    elif kind == 'TARGET_BACKGROUND_OUTLINE':
        occupancy = ((np.maximum(mp, mm) >= cfg['minimum_inside'])
            & (np.minimum(mp, mm) <= cfg['maximum_outside']) & (np.abs(mp-mm) >= cfg['minimum_transition']))
    else:
        raise ValueError('Explicit actual target/background or target/target context required')
    if not np.all(occupancy & (samples['RGB_gradient_energy'] >= cfg['minimum_gradient'])
            & (samples['RGB_alignment'] >= cfg['minimum_gradient_alignment'])
            & (samples['RGB_contrast'] >= cfg['minimum_color_transition'])):
        raise ValueError('Original actual RGB/context thresholds failed')
    lattice = np.asarray(chain['actual_inlier_image_lattice_xy'], float)
    if (lattice.shape != actual.shape or not np.isfinite(lattice).all()
            or not np.array_equal(lattice, np.round(lattice)) or np.any(lattice < 0)
            or np.any(lattice[:, 0] >= w) or np.any(lattice[:, 1] >= h)):
        raise ValueError('Actual inlier lattice support required for cross-edge overlap audit')
    sites = {tuple(map(int, v)) for v in lattice.astype(int)}
    if len(sites) != len(actual):
        raise ValueError('Repeated inlier lattice sites cannot increase support count')
    rms = float(np.sqrt(np.mean((actual@n+c)**2)))
    if not np.isclose(rms, quality[2], atol=1e-7, rtol=1e-7):
        raise ValueError('Inlier RMS must bind the actual weighted-TLS line')
    # Canonical line sign makes opposite-end duplicate bookkeeping invariant.
    if n[0] < 0 or (n[0] == 0 and n[1] < 0):
        n, c = -n, -c
    center = actual.mean(axis=0)
    anchor = center-n*(n@center+c)
    tangent = np.array([-n[1], n[0]])
    lo, hi = float((actual@tangent).min()), float((actual@tangent).max())
    span = hi-lo
    if span <= 0:
        raise ValueError('Nondegenerate actual inlier support required')
    fn = max(1., rms)
    fs = float(np.sin(np.arctan2(np.sqrt(2.)*fn, span)))
    return dict(native_corner=pair[0], native_neighbor=pair[1], normal_unit_xy=n.tolist(), offset_px=c,
        support_anchor_xy=anchor.tolist(), normal_variance_px2=vn+fn**2,
        angular_variance_rad2=va+fs**2, normal_angle_correlation=rho,
        actual_support_tangent_interval_px=[lo, hi], actual_inlier_span_px=span,
        support_extrapolation_px=float(cfg['along_start_px']+cfg['boundary_context_px']),
        normal_floor_px=fn, angular_sine_floor=fs, mask_context_kind=kind,
        actual_unweighted_inlier_centroid_xy=center.tolist(), actual_inlier_pixel_digest=_pixel_digest(actual),
        actual_inlier_lattice_xy=[list(p) for p in sorted(sites)],
        actual_inlier_lattice_digest=_pixel_digest(np.asarray(sorted(sites))),
        noise_provenance=deepcopy(noise),
        support_anchor_is_projected_statistic_not_actual_pixel_or_precise_KP=True,
        native_identity_and_independent_information_not_certified=True)


def authenticate_complete_bi_image_chain_groups(base, observation, *, visible_measurements,
        initial_self_hidden, image_binding, initial_projection_xy,
        normal_variance_component_px2, angular_variance_component_sine2,
        noise_provenance, normal_angle_correlation=0., calibration_preparation=False):
    """Authenticate all current BI native edges, not only old partial refs.

    ``initial_projection_xy`` MUST be the immutable ORIGINAL observer guide,
    shared across both candidate WD families. It supplies a tangent axis only;
    no model projection becomes an observed corner or normal measurement.
    ``visible_measurements`` binds exact frozen q/noise/source before archiving.
    A preparation bundle can be used for a new CAL fit, never for scoring.
    """
    if base.edges or hasattr(base, 'groups'):
        raise ValueError('Prepare an unmodified old-normal base; no duplicate edge mixtures')
    hidden = np.asarray(initial_self_hidden, bool)
    if hidden.shape != (8,) or list(hidden) != [s == 'S'for s in base.states]:
        raise ValueError('Permanent initial-S gate must match authoritative states')
    binding = observation['image_binding']
    if (not _sha(image_binding.get('sha256')) or binding.get('sha256') != image_binding['sha256']
            or tuple(binding.get('image_hw', ())) != tuple(base.image_hw)
            or tuple(image_binding.get('image_hw', ())) != tuple(base.image_hw)
            or ('image_pixel_sha256'in image_binding and binding.get('image_pixel_sha256') != image_binding['image_pixel_sha256'])):
        raise ValueError('Exact image/canvas/pixel binding required')
    if (noise_provenance.get('fit_partition') != 'CAL'
            or noise_provenance.get('observation_family') != NOISE_OBSERVATION_FAMILY
            or (not calibration_preparation and not _sha(noise_provenance.get('model_digest_sha256')))):
        raise ValueError('Separate new complete+partial population CAL model/provenance required')
    vn, va, rho = float(normal_variance_component_px2), float(angular_variance_component_sine2), float(normal_angle_correlation)
    if not np.isfinite([vn, va, rho]).all() or min(vn, va) < 0 or abs(rho) >= 1.-1e-8:
        raise ValueError('Nonnegative CAL components and positive-definite correlation required')
    archives = _archive_measurements(base, visible_measurements, hidden)
    old_refs = _old_normal_references(base, hidden)
    grouped, skipped = {}, []
    for ray in observation['edge_records']:
        k, mate = _id(ray['native_corner']), _id(ray['native_neighbor'])
        if hidden[k] or hidden[mate] or base.states[k] not in ('B', 'I'):
            skipped.append(dict(native_corner=k, native_neighbor=mate, reason='initial_S_or_non_BI_before_fields'))
            continue
        if mate not in NEIGHBORS[k] or np.count_nonzero(np.abs(base.xyz[k]-base.xyz[mate]) > 1e-9) != 1:
            raise ValueError('A real registered cuboid incident edge is required')
        if not ray.get('supported'):
            continue
        if (ray.get('measurement_source') != 'ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT'
                or ray.get('response_is_not_window_clipped') is not True):
            raise ValueError('Actual lattice NMS, not guide-window-clipped samples, required')
        pair = tuple(sorted((k, mate)))
        first, second = np.asarray(initial_projection_xy[k], float), np.asarray(initial_projection_xy[mate], float)
        if first.shape != (2,) or second.shape != (2,) or not np.isfinite([first, second]).all():
            raise ValueError('Only eligible nonS ORIGINAL observer guide projections required')
        guide = second-first
        if np.linalg.norm(guide) <= 1e-10:
            raise ValueError('Nondegenerate original observer guide required')
        guide /= np.linalg.norm(guide)
        for index, chain in enumerate(ray['chains']):
            component = _component(chain, k, mate, pair, guide, OBS.DEFAULT_CONFIGURATION,
                base.image_hw, vn, va, rho, noise_provenance)
            token = component['actual_inlier_pixel_digest']
            target = grouped.setdefault(pair, dict(components={}, source_rays=[]))
            previous = target['components'].get(token)
            if previous is not None and (not np.allclose(previous['normal_unit_xy'], component['normal_unit_xy'], atol=1e-7, rtol=0)
                    or not np.isclose(previous['offset_px'], component['offset_px'], atol=1e-7, rtol=0)):
                raise ValueError('Same actual pixel set has conflicting weighted-TLS coefficients')
            target['components'][token] = component
            target['source_rays'].append(dict(native_corner=k, native_neighbor=mate, source_chain_index=index,
                actual_inlier_pixel_digest=token))
    groups = []
    for pair, values in sorted(grouped.items()):
        bi = [k for k in pair if not hidden[k] and base.states[k] in ('B', 'I')]
        refs = old_refs.get(pair, [])
        weights = [archives[k]['original_fit_reliability_weight']for k in bi]+[r['reliability']for r in refs]
        groups.append(dict(physical_native_edge=list(pair), native_corner=pair[0]if pair[0]in bi else pair[1],
            native_neighbor=pair[1]if pair[0]in bi else pair[0],
            components=[v for _, v in sorted(values['components'].items())],
            reliability_weight=min(weights), reliability_applied_once=True,
            replaces_normal_factor_refs=[dict(native_corner=r['native_corner'], line_index=r['line_index'])for r in refs],
            source_rays=values['source_rays'], component_priors='uniform over distinct actual-pixel sets'))
    overlaps = []
    for left, right in combinations(groups, 2):
        for i, a in enumerate(left['components']):
            aa = {tuple(p) for p in a['actual_inlier_lattice_xy']}
            for j, b in enumerate(right['components']):
                bb = {tuple(p) for p in b['actual_inlier_lattice_xy']}
                shared = aa & bb
                if shared:
                    overlaps.append(dict(left_edge=left['physical_native_edge'], right_edge=right['physical_native_edge'],
                        left_component=i, right_component=j, shared_lattice_count=len(shared),
                        left_support_count=len(aa), right_support_count=len(bb),
                        identical_lattice_support=aa == bb, identical_subpixel_support=a['actual_inlier_pixel_digest'] == b['actual_inlier_pixel_digest'],
                        incident_physical_edges=bool(set(left['physical_native_edge']) & set(right['physical_native_edge'])),
                        shared_lattice_xy=[list(p) for p in sorted(shared)],
                        may_be_legitimate_junction_or_wrong_native_alias=True, independent_evidence_claimed=False))
    guides = {}
    for group in groups:
        k, mate = group['native_corner'], group['native_neighbor']
        first = np.asarray(initial_projection_xy[k], float)
        second = np.asarray(initial_projection_xy[mate], float)
        if first.shape != (2,) or second.shape != (2,) or not np.isfinite([first, second]).all():
            raise ValueError('Only eligible nonS immutable ORIGINAL observer guide projections required')
        direction = second-first
        if np.linalg.norm(direction) <= 1e-10:
            raise ValueError('Nondegenerate original observer tangent guide required')
        guides[':'.join(map(str, group['physical_native_edge']))] = (direction/np.linalg.norm(direction)).tolist()
    return dict(schema='AUTHENTICATED_ALL_BI_ACTUAL_NMS_GROUPS_PROTOTYPE_V20', groups=groups,
        frozen_measurement_archives={str(k): v for k, v in archives.items()}, original_observer_guide_tangents=guides,
        cross_edge_support_overlaps=overlaps, skipped=skipped,
        calibration_preparation=bool(calibration_preparation),
        noise_observation_family=NOISE_OBSERVATION_FAMILY,
        all_alternatives_retained=True, physical_edge_groups_deduplicated=True,
        S_coordinate_reads=0, S_local_queries=0, U_coordinate_reads=0, image_reads=0, GT_inputs=False,
        independent_edges_or_unique_corner_not_certified=True, new_precise_KPs=0)


def plan_complete_bi_replacement(base, bundle, *, overlap_policy):
    """Return an explicit provisional objective plan; never mutate archives.

    Robust 2-direction support means *every* alternative cross-combination of
    some two distinct edges passes the original sine .25 guard. Several edges
    without that robust distinction abstain rather than selecting a best pair.
    """
    if overlap_policy != PROTOTYPE_OVERLAP_POLICY:
        raise ValueError('Choose the explicit provisional overlap-abstention policy; no silent default')
    blocked = {tuple(o[key]) for o in bundle['cross_edge_support_overlaps']for key in ('left_edge', 'right_edge')}
    groups = {tuple(g['physical_native_edge']): g for g in bundle['groups']}
    complete = {m['native_corner']: m for m in base.complete}
    active = set(groups)-blocked
    ambiguous = set()
    sine = OBS.DEFAULT_CONFIGURATION['minimum_intersection_sine']
    while True:
        remove = set()
        for k in complete:
            incident = sorted(pair for pair in active if k in pair)
            if len(incident) < 2:
                continue
            robust = any(all(abs(float(np.linalg.det(np.array([a['normal_unit_xy'], b['normal_unit_xy']])))) >= sine
                for a, b in product(groups[left]['components'], groups[right]['components']))
                for left, right in combinations(incident, 2))
            if not robust:
                ambiguous.add(k); remove.update(incident)
        if not remove:
            break
        active -= remove
    plans = []
    for k in sorted(complete):
        incident = sorted(pair for pair in active if k in pair)
        archive = bundle['frozen_measurement_archives'][str(k)]
        record = dict(native_corner=k, state=base.states[k], active_physical_edges=[list(p)for p in incident],
            original_measurement_archive=deepcopy(archive), point_cost_revoked=bool(incident),
            reason=('no_active_authenticated_direction'if not incident else
                    'one_actual_edge_and_uncertain_archived_complete_tangent'if len(incident) == 1 else
                    'robust_all_alternative_nonparallel_incident_directions'),
            direction_rank_ambiguous_replacement_abstained=k in ambiguous)
        if len(incident) == 1:
            tangent = np.asarray(bundle['original_observer_guide_tangents'][':'.join(map(str, incident[0]))])
            covariance = np.asarray(archive['original_effective_fit_covariance_px2'])
            record.update(tangent_kind='ARCHIVED_COMPLETE_BI_TANGENT', tangent_unit_xy=tangent.tolist(),
                tangent_variance_px2=float(TANGENT_VARIANCE_INFLATION*(tangent@covariance@tangent)),
                tangent_variance_inflation=TANGENT_VARIANCE_INFLATION,
                tangent_axis_source='IMMUTABLE_ORIGINAL_OBSERVER_MODEL_GUIDE_SAME_ACROSS_WD',
                tangent_is_uncertain_dependent_prediction_not_new_RGB_observation=True,
                tangent_total_error_validation_still_required=True)
        plans.append(record)
    active_groups = [deepcopy(groups[p])for p in sorted(active)]
    return dict(schema='COMPLETE_BI_DIRECTION_REPLACEMENT_PROTOTYPE_V20', active_groups=active_groups,
        corner_plans=plans, revoked_complete_ids=[p['native_corner']for p in plans if p['point_cost_revoked']],
        replaces_normal_factor_refs=[r for g in active_groups for r in g['replaces_normal_factor_refs']],
        overlap_policy=overlap_policy, overlap_policy_provisional_not_adopted=True,
        overlap_withheld_physical_edges=[list(p)for p in sorted(blocked)],
        all_authenticated_groups_and_alternatives_remain_in_bundle=True,
        source_complete_q_covariance_reliability_and_source_unchanged=True,
        minimum_intersection_sine=sine, new_full_2D_points=0,
        old_hard_guards_preserved_not_validated_new_RGB_localization=True)


class CompleteBIImageChainProblem:
    """Analytic objective-only prototype; unchanged original hard guards.

    This is not a solver or admitted full-population experiment. It requires
    a separate frozen CAL model even if the old partial-only model happens to
    have similar numerical parameters. Shared/ambiguous support stays archived.
    """
    def __init__(self, base, bundle, *, overlap_policy):
        if bundle['calibration_preparation']:
            raise ValueError('Fit/freeze the new CAL population before using any pose objective')
        self.plan = plan_complete_bi_replacement(base, bundle, overlap_policy=overlap_policy)
        self.base = copy(base)
        self.groups = self.plan['active_groups']
        self.revoked = set(self.plan['revoked_complete_ids'])
        self.base.complete = [m for m in base.complete if m['native_corner']not in self.revoked]
        refs = {(r['native_corner'], r['line_index'])for r in self.plan['replaces_normal_factor_refs']}
        if len(refs) != len(self.plan['replaces_normal_factor_refs']):
            raise ValueError('Existing normal references must be replaced once')
        self.base.remaining_normals = []
        for b in base.remaining_normals:
            keep = [line for index, line in enumerate(b['lines'])if (b['native_corner'], index)not in refs]
            if keep:
                self.base.remaining_normals.append({**b, 'lines': keep})

    def __getattr__(self, name):
        return getattr(self.base, name)

    def objective(self, p):
        value, gradient, parts = self.base.objective(p)
        audits = []
        for group in self.groups:
            extra, derivative, audit = M.mixture_edge_value_and_gradient(p, self.xyz, self.K, self.scale,
                group['components'], reliability_weight=group['reliability_weight'], huber_scale=self.cfg['huber_scale'])
            value += extra; gradient += derivative; audits.append(audit)
            parts['native_costs'].append(dict(kind='PROJECTED_NATIVE_EDGE', native_corner=group['native_corner'],
                native_neighbor=group['native_neighbor'], weighted_robust_cost=extra))
        q, _, jac, _ = N.J.projection_and_jacobian(p, self.xyz, self.K, self.scale)
        delta = self.cfg['huber_scale']
        for record in self.plan['corner_plans']:
            if record.get('tangent_kind') != 'ARCHIVED_COMPLETE_BI_TANGENT':
                continue
            archive = record['original_measurement_archive']; k = record['native_corner']
            tangent = np.asarray(record['tangent_unit_xy']); sigma = np.sqrt(record['tangent_variance_px2'])
            r = float(tangent@(q[k]-archive['original_observed_xy'])/sigma)
            derivative = tangent@jac[k]/sigma
            weight = archive['original_fit_reliability_weight']
            denominator = np.sqrt(1.+r*r/delta**2)
            extra = float(weight*2.*delta**2*(denominator-1.))
            value += extra; gradient += 2.*weight*r/denominator*derivative
            parts['native_costs'].append(dict(kind='ARCHIVED_COMPLETE_BI_TANGENT', native_corner=k,
                weighted_robust_cost=extra, reliability_applied_once=weight))
        parts.update(new_actual_image_chain_audit=audits, complete_BI_replacement_audit=self.plan,
            complete_BI_likelihood_count=len(self.base.complete), complete_BI_likelihood_revocations=len(self.revoked),
            old_q_not_renamed_raw=True, new_full_2D_KPs=0,
            hard_guard_measurements_not_removed=True, old_guards_not_new_RGB_validation=True,
            overlap_policy_provisional_not_adopted=True, new_CAL_population_required=True)
        return float(value), gradient, parts

    def evaluate(self, R, t):
        checked = self.base.evaluate(R, t)
        if checked.get('objective')is None:
            return checked
        try:
            p = N.V._pose_vector(np.asarray(R), np.asarray(t), self.scale)
            value, _, parts = self.objective(p)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError)as error:
            return dict(accepted=False, objective=None, native_costs=[], reason=str(error))
        return {**checked, **parts, 'objective': value,
            'accepted': bool(checked['accepted']and np.isfinite(value)),
            'accepted_does_not_mean_unique_or_accurate': True}
