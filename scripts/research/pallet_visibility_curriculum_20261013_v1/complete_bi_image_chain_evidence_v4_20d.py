"""Typed finite/topology complete-BI evidence, separate from broader v20.

Support-balanced fit-variant cached-observation preparation/objective prototype. No file/image/label
read, new point, pose fit or source mutation. New CAL population required.
Self-anchor B exterior / I internal evidence is required for complete costs;
all alternatives, finite11, shift12 and existing3px consistency are checked.
B pairs additionally bind archived actual-mask topology at exact chain refs.
These are conditional associations, never precise-KP/ownership certificates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from itertools import combinations, product
import types

import numpy as np

from . import complete_bi_image_chain_evidence_v4_20 as BROAD
from .complete_bi_image_chain_evidence_v4_20c import _same_recipe

NOISE_OBSERVATION_FAMILY = 'SUPPORT_BALANCED_FIT_VARIANT_TYPED_COMPLETE_PARTIAL_NMS17C_MIXTURES_V20D'
PROTOTYPE_OVERLAP_POLICY = BROAD.PROTOTYPE_OVERLAP_POLICY
N, M, OBS = BROAD.N, BROAD.M, BROAD.OBS


def _ref_token(ray, chain):
    return f'{ray}:{chain}'


def _pair_token(k, first, second):
    return str(k)+'|'+ '|'.join(sorted((first, second)))


def _relation(chain, image_hw):
    xy = np.asarray(chain['observed_points'], float)
    mask = np.asarray(chain['inlier_mask'], bool)
    gradient = np.asarray(chain['image_gradient_normal_xy'], float)
    shift = np.asarray(chain['measured_subpixel_shift_px'], float)
    lattice = np.asarray(chain['actual_inlier_image_lattice_xy'], float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or mask.shape != (len(xy),)
            or gradient.shape != xy.shape or shift.shape != (len(xy),)
            or lattice.shape != xy[mask].shape or not np.isfinite(gradient).all()
            or not np.isfinite(shift).all()
            or not np.allclose(np.linalg.norm(gradient, axis=1), 1., atol=1e-6, rtol=0)
            or np.any(np.abs(shift) > .5)):
        raise ValueError('Archived unit actual gradient and bounded NMS subpixel shift required')
    reconstructed = lattice+gradient[mask]*shift[mask, None]*OBS.PIXEL_LOCALIZATION_FLOOR_PX
    error = float(np.max(np.abs(reconstructed-xy[mask])))if len(lattice)else float('inf')
    if error > 1e-7:
        raise ValueError('Actual inlier lattice + gradient * subpixel shift must equal observed points')
    cfg = OBS.DEFAULT_CONFIGURATION
    margin = max(cfg['RGB_probe_px'], cfg['boundary_context_px'])
    h, w = image_hw
    if (np.any(lattice[:, 0] < margin) or np.any(lattice[:, 0] >= w-1-margin)
            or np.any(lattice[:, 1] < margin) or np.any(lattice[:, 1] >= h-1-margin)):
        raise ValueError('Original image-lattice context margin must exclude frame-padding edges')
    return error


def _finite_pair(base, k, first, second, archive):
    cfg = OBS.DEFAULT_CONFIGURATION
    matrix = np.asarray([first['normal_unit_xy'], second['normal_unit_xy']], float)
    sine = abs(float(np.linalg.det(matrix)))
    if sine < cfg['minimum_intersection_sine']:
        return dict(passed=False, reason='near_parallel', intersection_sine=sine)
    point = np.linalg.solve(matrix, -np.asarray([first['offset_px'], second['offset_px']]))
    support = []
    for component in (first, second):
        n = np.asarray(component['normal_unit_xy']); tangent = np.array([-n[1], n[0]])
        lo, hi = component['actual_support_tangent_interval_px']; value = float(tangent@point)
        gap = max(float(lo-value), float(value-hi), 0.)
        support.append(dict(gap_px=gap, passed=gap <= cfg['along_start_px']+cfg['boundary_context_px']))
    shift = float(np.linalg.norm(point-archive['original_observed_xy']))
    h, w = base.image_hw
    passed = bool(all(s['passed']for s in support)and shift <= cfg['maximum_corner_shift_px']
        and 0 <= point[0] < w and 0 <= point[1] < h)
    return dict(passed=passed, reason='finite_shift_canvas_pass'if passed else 'finite_shift_canvas_failed',
        intersection_xy=point.tolist(), intersection_sine=sine, actual_finite_support=support,
        shift_from_exact_archived_complete_q_px=shift, intersection_is_diagnostic_not_new_KP=True)


def _topology_candidate(candidate, k, reference, components, base, archive):
    cfg = OBS.DEFAULT_CONFIGURATION
    refs = candidate['chain_refs']
    if len(refs) != 2 or any(r['native_corner'] != k for r in refs):
        raise ValueError('Both exact self-anchor topology chain refs required')
    tokens = [_ref_token(r['ray_index'], r['chain_index'])for r in refs]
    if any(t not in reference for t in tokens):
        raise ValueError('Topology refs must identify authenticated actual NMS chains')
    mapped = [reference[t]for t in tokens]
    for ref, item in zip(refs, mapped):
        if ref['native_neighbor'] != item['native_neighbor']:
            raise ValueError('Topology and actual chain native IDs differ')
    if mapped[0]['physical_edge'] == mapped[1]['physical_edge']:
        raise ValueError('Two distinct incident physical edges required')
    first, second = [components[(tuple(x['physical_edge']), x['component_id'])]for x in mapped]
    checked = _finite_pair(base, k, first, second, archive)
    if (not checked['passed'] or not np.allclose(checked['intersection_xy'], candidate['intersection_xy'], atol=1e-7, rtol=0)
            or not np.isclose(checked['intersection_sine'], candidate['intersection_sine'], atol=1e-7, rtol=0)
            or not np.isclose(checked['shift_from_exact_archived_complete_q_px'], candidate['shift_from_archived_B_observation_px'], atol=1e-7, rtol=0)):
        raise ValueError('Archived topology candidate must bind recomputed finite11/shift12 intersection')
    topology = candidate['local_topology']
    if topology.get('passed')is not True or len(topology['chain_end_support']) != 2:
        raise ValueError('Actual external topology PASS required')
    margin = cfg['along_start_px']+cfg['boundary_context_px']
    for component, audit in zip((first, second), topology['chain_end_support']):
        points = np.asarray(component['actual_inlier_xy']); n = np.asarray(component['normal_unit_xy'])
        tangent = np.array([-n[1], n[0]])
        signed = (points-checked['intersection_xy'])@tangent
        gap = min(abs(float(signed.min())), abs(float(signed.max())))
        one_sided = max(float(np.mean(signed >= 0)), float(np.mean(signed <= 0)))
        if (gap > margin or one_sided < cfg['minimum_inlier_fraction']or audit.get('passed')is not True
                or not np.isclose(gap, audit['nearest_actual_support_endpoint_tangent_distance_px'], atol=1e-7, rtol=0)
                or not np.isclose(one_sided, audit['actual_support_one_sided_fraction'], atol=1e-7, rtol=0)):
            raise ValueError('Actual one-sided finite chain ends must bind topology audit')
    if not topology['local_external_contours']or not all(
            0 <= float(v['junction_distance_to_actual_external_contour_px']) <= cfg['boundary_context_px']
            and 0 <= float(v['shortest_actual_external_contour_arc_px']) <= 2*margin
            and isinstance(v['component'], int)and v['component'] > 0
            for v in topology['local_external_contours']):
        raise ValueError('Same actual external contour/local arc evidence required')
    point = np.asarray(checked['intersection_xy']); inward = np.asarray(topology['inside_direction_from_actual_chain_context'], float)
    probes = np.asarray(topology['actual_mask_inside_outside_probe_xy'], float)
    values = np.asarray(topology['actual_mask_inside_outside_values'], float)
    h, w = base.image_hw
    if (inward.shape != (2,)or not np.isclose(np.linalg.norm(inward), 1., atol=1e-6, rtol=0)
            or probes.shape != (2, 2)or not np.isfinite(probes).all()
            or not np.allclose(probes, [point+inward*cfg['boundary_context_px'], point-inward*cfg['boundary_context_px']], atol=1e-7, rtol=0)
            or np.any(probes < 0)or np.any(probes[:, 0] >= w-1)or np.any(probes[:, 1] >= h-1)
            or values.shape != (2,)or not np.isfinite(values).all()
            or not 0 <= values[1] <= cfg['maximum_outside']or not cfg['minimum_inside'] <= values[0] <= 1):
        raise ValueError('Archived actual-mask foreground/background wedge must bind the junction')
    return _pair_token(k, *tokens), {**checked, 'archived_mask_topology_bound': True,
        'topology_is_conditional_not_native_identity_certificate': True}


def _fit_identity(component):
    fields = {k: component[k]for k in ('normal_unit_xy', 'offset_px', 'support_anchor_xy',
        'actual_support_tangent_interval_px', 'normal_floor_px', 'angular_sine_floor')}
    fit = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    support = component['actual_inlier_pixel_digest']
    return support, fit, support+':'+fit


def _AUTH(base, observation, *, visible_measurements, initial_self_hidden, image_binding,
        initial_projection_xy, normal_variance_component_px2, angular_variance_component_sine2,
        noise_provenance, normal_angle_correlation=0., calibration_preparation=False):
    """One physical factor; equal support mass, split among exact fit variants."""
    if base.edges or hasattr(base, 'groups'):
        raise ValueError('Unmodified old-normal base required')
    hidden = np.asarray(initial_self_hidden, bool)
    if hidden.shape != (8,) or list(hidden) != [s == 'S'for s in base.states]:
        raise ValueError('Permanent initial S gate required')
    binding = observation['image_binding']
    if (not BROAD._sha(image_binding.get('sha256')) or binding.get('sha256') != image_binding['sha256']
            or tuple(binding.get('image_hw', ())) != tuple(base.image_hw)
            or tuple(image_binding.get('image_hw', ())) != tuple(base.image_hw)
            or ('image_pixel_sha256'in image_binding and binding.get('image_pixel_sha256') != image_binding['image_pixel_sha256'])):
        raise ValueError('Exact image/canvas/pixel binding required')
    if (noise_provenance.get('fit_partition') != 'CAL'
            or noise_provenance.get('observation_family') != NOISE_OBSERVATION_FAMILY
            or (not calibration_preparation and not BROAD._sha(noise_provenance.get('model_digest_sha256')))):
        raise ValueError('Separate support-balanced new-population CAL model required')
    vn, va, rho = float(normal_variance_component_px2), float(angular_variance_component_sine2), float(normal_angle_correlation)
    if not np.isfinite([vn, va, rho]).all() or min(vn, va) < 0 or abs(rho) >= 1.-1e-8:
        raise ValueError('Finite nonnegative CAL components/correlation required')
    archives = BROAD._archive_measurements(base, visible_measurements, hidden)
    refs = BROAD._old_normal_references(base, hidden)
    grouped = {}; skipped = []; references = {}; guides = {}
    for ri, ray in enumerate(observation['edge_records']):
        k, mate = BROAD._id(ray['native_corner']), BROAD._id(ray['native_neighbor'])
        if hidden[k]or hidden[mate]or base.states[k]not in ('B', 'I'):
            skipped.append(dict(native_corner=k, native_neighbor=mate, reason='initial_S_or_non_BI_before_fields'));continue
        if mate not in BROAD.NEIGHBORS[k]or np.count_nonzero(np.abs(base.xyz[k]-base.xyz[mate]) > 1e-9) != 1:
            raise ValueError('Actual registered incident native edge required')
        if not ray.get('supported'):continue
        if ray.get('measurement_source') != 'ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT'or ray.get('response_is_not_window_clipped')is not True:
            raise ValueError('Actual image lattice NMS source required')
        first, second = np.asarray(initial_projection_xy[k], float), np.asarray(initial_projection_xy[mate], float)
        if first.shape != (2,)or second.shape != (2,)or not np.isfinite([first, second]).all():
            raise ValueError('NonS ORIGINAL observer guide required')
        guide = second-first
        if np.linalg.norm(guide) <= 1e-10:raise ValueError('Nondegenerate original guide required')
        guide /= np.linalg.norm(guide);pair = tuple(sorted((k, mate)))
        # An unoriented tangent has identical squared residual under reversal.
        canonical_guide = guide if k == pair[0]else -guide
        guides[':'.join(map(str, pair))] = canonical_guide.tolist()
        for ci, chain in enumerate(ray['chains']):
            c = BROAD._component(chain, k, mate, pair, guide, OBS.DEFAULT_CONFIGURATION,
                base.image_hw, vn, va, rho, noise_provenance)
            support, fit, identity = _fit_identity(c)
            c.update(support_id=support, fit_variant_id=fit, component_id=identity,
                full_observed_pixel_digest=BROAD._pixel_digest(np.asarray(chain['observed_points'])),
                fit_variant_identity='exact canonical fitted TLS/support/floor statistics; numerical variants may remain')
            g = grouped.setdefault(pair, dict(components={}, source_rays=[]))
            g['components'].setdefault(identity, c)
            token = _ref_token(ri, ci)
            item = dict(ray_index=ri, chain_index=ci, native_corner=k, native_neighbor=mate,
                physical_edge=list(pair), support_id=support, fit_variant_id=fit, component_id=identity)
            references[token] = item;g['source_rays'].append(item)
    groups = []
    for pair, group in sorted(grouped.items()):
        values = [v for _, v in sorted(group['components'].items())]
        support_ids = sorted({c['support_id']for c in values})
        counts = {s: sum(c['support_id'] == s for c in values)for s in support_ids}
        for c in values:c['mixture_component_prior'] = 1./len(support_ids)/counts[c['support_id']]
        bi = [k for k in pair if base.states[k]in ('B', 'I')and not hidden[k]]
        old = refs.get(pair, [])
        weights = [archives[k]['original_fit_reliability_weight']for k in bi]+[r['reliability']for r in old]
        groups.append(dict(physical_native_edge=list(pair), native_corner=pair[0]if pair[0]in bi else pair[1],
            native_neighbor=pair[1]if pair[0]in bi else pair[0], components=values,
            component_weights=[c['mixture_component_prior']for c in values],
            support_prior_ledger=[dict(support_id=s, total_prior=1./len(support_ids), fit_variants=counts[s])for s in support_ids],
            reliability_weight=min(weights), reliability_applied_once=True,
            replaces_normal_factor_refs=[dict(native_corner=r['native_corner'], line_index=r['line_index'])for r in old],
            source_rays=group['source_rays'], component_priors='equal distinct inlier-support mass, split over distinct fit variants',
            same_support_variants_not_independent_observations=True))
    overlaps = []
    for left, right in combinations(groups, 2):
        def support_lattices(group):
            result = {}
            for c in group['components']:result.setdefault(c['support_id'], set()).update(map(tuple, c['actual_inlier_lattice_xy']))
            return result
        for sa, aa in support_lattices(left).items():
            for sb, bb in support_lattices(right).items():
                shared = aa&bb
                if shared:overlaps.append(dict(left_edge=left['physical_native_edge'], right_edge=right['physical_native_edge'],
                    left_support_id=sa, right_support_id=sb, shared_lattice_count=len(shared),
                    left_support_count=len(aa), right_support_count=len(bb), identical_lattice_support=aa == bb,
                    identical_subpixel_support=sa == sb,
                    incident_physical_edges=bool(set(left['physical_native_edge'])&set(right['physical_native_edge'])),
                    shared_lattice_xy=[list(p)for p in sorted(shared)],
                    repeated_fit_variants_do_not_multiply_overlap_inventory=True, independent_evidence_claimed=False))
    return dict(schema='SUPPORT_BALANCED_FIT_VARIANT_AUTHENTICATION_V20D',groups=groups,
        source_chain_component_bindings=references,
        frozen_measurement_archives={str(k):v for k,v in archives.items()},original_observer_guide_tangents=guides,
        cross_edge_support_overlaps=overlaps,skipped=skipped,calibration_preparation=bool(calibration_preparation),
        noise_observation_family=NOISE_OBSERVATION_FAMILY,all_alternatives_retained=True,
        physical_edge_groups_deduplicated=True,S_coordinate_reads=0,S_local_queries=0,U_coordinate_reads=0,image_reads=0,
        GT_inputs=False,new_precise_KPs=0,independent_edges_or_unique_corner_not_certified=True)


def _weighted_mixture(*args, **kwargs):
    components = args[4]
    return M.mixture_edge_value_and_gradient(*args,
        component_weights=[c['mixture_component_prior']for c in components], **kwargs)


_WEIGHTED_OBJECTIVE = types.FunctionType(BROAD.CompleteBIImageChainProblem.objective.__code__,
    {**BROAD.CompleteBIImageChainProblem.objective.__globals__,
     'M':types.SimpleNamespace(mixture_edge_value_and_gradient=_weighted_mixture)}, 'support_balanced_objective')


def authenticate_complete_bi_image_chain_groups(base, observation, *, junction_proposals,
        visible_measurements, initial_self_hidden, image_binding, initial_projection_xy,
        normal_variance_component_px2, angular_variance_component_sine2,
        noise_provenance, normal_angle_correlation=0., calibration_preparation=False):
    """Stricter new-generation authentication; S gates before chain fields."""
    observation = {**observation, 'configuration': _same_recipe(observation.get('configuration'))}
    junction_proposals = {**junction_proposals, 'configuration': _same_recipe(junction_proposals.get('configuration'))}
    hidden = np.asarray(initial_self_hidden, bool)
    if hidden.shape != (8,)or list(hidden) != [s == 'S'for s in base.states]:
        raise ValueError('Permanent original-S gate required')
    if observation.get('configuration') != OBS.DEFAULT_CONFIGURATION:
        raise ValueError('Exact original actual NMS recipe required')
    rays = list(observation['edge_records'])
    observation = {**observation, 'edge_records': rays}
    relation = []
    for ri, ray in enumerate(rays):
        k, mate = BROAD._id(ray['native_corner']), BROAD._id(ray['native_neighbor'])
        if hidden[k]or hidden[mate]or base.states[k]not in ('B', 'I')or not ray.get('supported'):
            continue
        for ci, chain in enumerate(ray['chains']):
            relation.append(dict(ray_index=ri, chain_index=ci, maximum_relation_error_px=_relation(chain, base.image_hw)))
    bundle = _AUTH(base, observation, visible_measurements=visible_measurements,
        initial_self_hidden=initial_self_hidden, image_binding=image_binding,
        initial_projection_xy=initial_projection_xy,
        normal_variance_component_px2=normal_variance_component_px2,
        angular_variance_component_sine2=angular_variance_component_sine2,
        noise_provenance=noise_provenance, normal_angle_correlation=normal_angle_correlation,
        calibration_preparation=calibration_preparation)
    if (junction_proposals.get('schema') != 'NATIVE_EDGE_TOPOLOGY_JUNCTION_PROPOSALS_V4_17B'
            or junction_proposals.get('configuration') != OBS.DEFAULT_CONFIGURATION
            or junction_proposals.get('image_binding') != observation['image_binding']
            or junction_proposals['mask_binding'].get('image_sha256') != image_binding['sha256']
            or not BROAD._sha(junction_proposals['mask_binding'].get('mask_pixel_sha256'))):
        raise ValueError('Exact separately sealed chain/image/mask topology archive required')
    components = {(tuple(g['physical_native_edge']), c['component_id']): c
        for g in bundle['groups']for c in g['components']}
    references = {}; typed = {}
    for ri, ray in enumerate(rays):
        k, mate = BROAD._id(ray['native_corner']), BROAD._id(ray['native_neighbor'])
        if hidden[k]or hidden[mate]or base.states[k]not in ('B', 'I')or not ray.get('supported'):
            continue
        pair = tuple(sorted((k, mate)))
        for ci, chain in enumerate(ray['chains']):
            actual = np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'], bool)]
            digest = BROAD._pixel_digest(actual)
            token = _ref_token(ri, ci)
            identity = bundle['source_chain_component_bindings'][token]
            component_id = identity['component_id']
            references[token] = dict(native_corner=k, native_neighbor=mate, physical_edge=list(pair), support_id=digest, fit_variant_id=identity['fit_variant_id'], component_id=component_id)
            components[(pair, component_id)]['actual_inlier_xy'] = actual.tolist()
            required = 'TARGET_BACKGROUND_OUTLINE'if base.states[k] == 'B'else 'TARGET_TARGET_INTERNAL'
            if chain['mask_context_kind'] == required:
                typed.setdefault(str(k), {}).setdefault(':'.join(map(str, pair)), {}).setdefault(component_id, []).append(token)
    topologies = {}
    for record in junction_proposals['records']:
        k = BROAD._id(record['native_corner'])
        if hidden[k]or base.states[k] != 'B':
            continue
        if record.get('state') != 'B':
            raise ValueError('Current B topology state must match authoritative state')
        for candidate in record['candidates']:
            key, audit = _topology_candidate(candidate, k, references, components, base,
                bundle['frozen_measurement_archives'][str(k)])
            topologies[key] = audit
    bundle.update(schema='TYPED_FINITE_TOPOLOGY_ALL_BI_NMS_AUTHENTICATION_PROTOTYPE_V20D',
        actual_lattice_gradient_subpixel_relation=relation, typed_self_anchor_component_refs=typed,
        B_exact_chain_ref_topology=topologies, topology_mask_binding=deepcopy(junction_proposals['mask_binding']),
        noise_observation_family=NOISE_OBSERVATION_FAMILY,
        new_population_and_typed_replacement_not_calibrated=True,
        topology_not_a_new_measured_point=True, configuration_serialization_normalized_only=True)
    return bundle


def plan_complete_bi_replacement(base, bundle, *, overlap_policy):
    """Per-self-anchor planning; one failed anchor cannot erase another.

    A physical line necessarily constrains both model endpoints. An opposite
    complete q remains a distinct, single original factor until that corner's
    own role-correct observation authorizes replacement. Correlation between
    RGB-derived q and line pixels is explicit, never independent information.
    """
    if overlap_policy != PROTOTYPE_OVERLAP_POLICY:
        raise ValueError('Explicit provisional overlap policy required')
    groups = {tuple(g['physical_native_edge']): g for g in bundle['groups']}
    blocked = {tuple(o[key])for o in bundle['cross_edge_support_overlaps']for key in ('left_edge', 'right_edge')}
    complete = {m['native_corner']for m in base.complete}
    rejections = []; plans = []; owners = {p: [] for p in groups if p not in blocked}
    for pair in owners:
        owners[pair].extend(dict(kind='EXISTING_PARTIAL_NORMAL_REF', **r)
            for r in groups[pair]['replaces_normal_factor_refs'])
    cfg = OBS.DEFAULT_CONFIGURATION
    for k in sorted(complete):
        incident = []
        for pair in owners:
            if k not in pair:
                continue
            mapping = bundle['typed_self_anchor_component_refs'].get(str(k), {}).get(':'.join(map(str, pair)), {})
            if any(c['component_id']not in mapping for c in groups[pair]['components']):
                rejections.append(dict(native_corner=k, physical_edge=list(pair), reason='not_all_alternatives_have_role_correct_self_anchor'))
                continue
            incident.append(pair)
        incident.sort()
        passed = bool(incident)
        if len(incident) >= 2:
            cloud = []; pair_audits = []; robust = False; passed = True
            archive = bundle['frozen_measurement_archives'][str(k)]
            for left, right in combinations(incident, 2):
                robust_pair = True
                for a, b in product(groups[left]['components'], groups[right]['components']):
                    audit = _finite_pair(base, k, a, b, archive)
                    if audit['reason'] == 'near_parallel':
                        robust_pair = False; continue
                    good = audit['passed']
                    if good and base.states[k] == 'B':
                        typed = bundle['typed_self_anchor_component_refs'][str(k)]
                        aa = typed[':'.join(map(str, left))][a['component_id']]
                        bb = typed[':'.join(map(str, right))][b['component_id']]
                        good = any(_pair_token(k, x, y)in bundle['B_exact_chain_ref_topology']for x, y in product(aa, bb))
                        audit['B_actual_mask_topology_pass'] = good
                    pair_audits.append(audit)
                    passed &= good
                    if good:
                        cloud.append(audit['intersection_xy'])
                robust |= robust_pair
            disagreement = max((float(np.linalg.norm(np.asarray(a)-b))for a, b in combinations(cloud, 2)), default=0.)
            passed = bool(passed and robust and cloud and disagreement <= cfg['maximum_candidate_disagreement_px'])
            if not passed:
                rejections.append(dict(native_corner=k, reason='all_alternative_finite_shift_typed_topology_or_disagreement_failed',
                    robust_all_alternative_rank2_pair=robust, candidate_disagreement_px=disagreement,
                    diagnostic_pair_audits=pair_audits, diagnostic_intersections_are_not_KPs=True))
        admitted = incident if passed else []
        archive = bundle['frozen_measurement_archives'][str(k)]
        record = dict(native_corner=k, state=base.states[k],
            active_physical_edges=[list(p) for p in admitted],
            original_measurement_archive=deepcopy(archive), point_cost_revoked=bool(admitted),
            reason=('no_eligible_self_anchor_replacement'if not admitted else
                'one_self_typed_direction_and_uncertain_archived_complete_tangent'if len(admitted) == 1 else
                'all_alternative_finite_self_typed_junction_association'),
            opposite_anchor_group_does_not_authorize_this_point_revocation=True)
        if len(admitted) == 1:
            tangent = np.asarray(bundle['original_observer_guide_tangents'][':'.join(map(str, admitted[0]))])
            covariance = np.asarray(archive['original_effective_fit_covariance_px2'])
            record.update(tangent_kind='ARCHIVED_COMPLETE_BI_TANGENT', tangent_unit_xy=tangent.tolist(),
                tangent_variance_px2=float(BROAD.TANGENT_VARIANCE_INFLATION*(tangent@covariance@tangent)),
                tangent_variance_inflation=BROAD.TANGENT_VARIANCE_INFLATION,
                tangent_axis_source='IMMUTABLE_ORIGINAL_OBSERVER_MODEL_GUIDE_SAME_ACROSS_WD',
                tangent_is_uncertain_dependent_prediction_not_new_RGB_observation=True,
                tangent_total_error_validation_still_required=True)
        for pair in admitted:
            owners[pair].append(dict(kind='ROLE_CORRECT_SELF_COMPLETE_ANCHOR', native_corner=k))
        plans.append(record)
    active = []
    for pair, evidence in sorted(owners.items()):
        if evidence:
            active.append({**deepcopy(groups[pair]), 'activation_owners': evidence})
    return dict(schema='TYPED_FINITE_TOPOLOGY_COMPLETE_BI_REPLACEMENT_PROTOTYPE_V20D',
        active_groups=active, corner_plans=plans,
        revoked_complete_ids=[r['native_corner']for r in plans if r['point_cost_revoked']],
        replaces_normal_factor_refs=[r for g in active for r in g['replaces_normal_factor_refs']],
        overlap_policy=overlap_policy, overlap_policy_provisional_not_adopted=True,
        overlap_withheld_physical_edges=[list(p)for p in sorted(blocked)],
        new_full_2D_points=0, source_complete_q_covariance_reliability_and_source_unchanged=True,
        ledger=dict(original_complete_q_factors_once=len(complete)-sum(r['point_cost_revoked']for r in plans),
            archived_complete_tangent_factors_once=sum('tangent_kind'in r for r in plans),
            unique_physical_edge_factors_once=len(active),
            opposite_complete_q_and_edge_may_be_RGB_correlated=True,
            independent_information_increase_claimed=False),
        typed_self_anchor_required=True, B_exterior_and_I_internal_not_mixed=True,
        all_alternative_finite11_shift12_and_old3px_consistency=True,
        B_archived_actual_topology_bound_at_exact_chain_refs=True,
        I_internal_pair_is_conditional_line_association_not_external_topology=True,
        main_replacement_rejections=rejections, all_original_groups_retained_in_authentication_bundle=True,
        no_new_finite_projected_3D_segment_guard=True,
        old_guard_calibration_is_not_new_RGB_validation=True,
        failed_anchor_does_not_delete_another_valid_anchor_or_oldpartial_evidence=True)


class CompleteBIImageChainProblem(BROAD.CompleteBIImageChainProblem):
    __init__ = types.FunctionType(BROAD.CompleteBIImageChainProblem.__init__.__code__,
        {**BROAD.CompleteBIImageChainProblem.__init__.__globals__,
         'plan_complete_bi_replacement': plan_complete_bi_replacement}, '__init__')

    def objective(self, p):
        value, gradient, audit = _WEIGHTED_OBJECTIVE(self, p)
        audit.update(typed_finite_topology_complete_BI_generation='V20D',
            actual_lattice_to_subpixel_points_checked=True, projected_finite_3D_segment_guard_added=False,
            new_population_CAL_model_required_before_real_scoring=True,
            support_balanced_fit_variant_mixture_priors=True)
        return value, gradient, audit
