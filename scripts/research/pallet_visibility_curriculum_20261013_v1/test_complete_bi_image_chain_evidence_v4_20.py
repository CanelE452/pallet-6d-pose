"""Synthetic observation algebra only; no files, images, GT or pose fits."""
from copy import copy, deepcopy
from itertools import combinations
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import complete_bi_image_chain_evidence_v4_20 as NEW
from .common import cuboid


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Forbidden coordinate access')

    def __deepcopy__(self, memo):
        raise AssertionError('Forbidden hidden observation copy')


def fixture(partial=False):
    xyz = cuboid(.8, .4, .6)
    R = Rotation.from_rotvec([.2, -.3, .05]).as_matrix()
    t = np.array([.08, -.04, 3.])
    K = np.array([[430., 7., 320.], [0., 420., 240.], [0., 0., 1.]])
    projection, _ = NEW.N.V.project_native(xyz, R, t, K)
    camera = -R.T@t
    normals = NEW.N.V.registered_incident_normals(xyz)
    cosines = [n@((camera-p)/np.linalg.norm(camera-p))for n, p in zip(normals, xyz)]
    states = ['S'if max(c) < 0 else 'I'if min(c) >= 0 else 'B'for c in cosines]
    records = [dict(native_corner=k, state=s, xy=projection[k].tolist(), covariance_px2=[[4., 0.], [0., 4.]],
        fit_covariance_px2=[[25., 2.], [2., 16.]], fit_reliability_weight=.7+.01*k,
        protection_px=2., source='FROZEN_RGB_COMPLETE_SOURCE')for k, s in enumerate(states)if s != 'S']
    usable = []
    for k, state in enumerate(states):
        mates = [m for m in NEW.NEIGHBORS[k]if states[m] != 'S']
        for a, b in combinations(mates, 2):
            x, y = projection[a]-projection[k], projection[b]-projection[k]
            if state == 'B' and abs(np.linalg.det([x/np.linalg.norm(x), y/np.linalg.norm(y)])) >= .25:
                usable.append((k, a, b))
    k, a, b = usable[0]
    boundary = []
    if partial:
        record = next(r for r in records if r['native_corner'] == k)
        record['source'] = 'VISIBLE_BACKBONE_FALLBACK'
        direction = projection[a]-projection[k]; direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        ss = float(direction@projection[k])
        boundary = [dict(native_corner=k, state='B', fit_reliability_weight=record['fit_reliability_weight'],
            replaces_raw_B_likelihood=True, image_binding=dict(sha256='a'*64, image_hw=[480, 640]),
            association_provenance=dict(original_raw_xy=record['xy']), lines=[dict(normal_unit_xy=normal.tolist(),
                offset_px=-float(normal@projection[k]), normal_variance_px2=25.,
                support_tangent_interval_px=[ss-100., ss+100.], support_extrapolation_px=11.,
                provenance=dict(native_neighbor=a))])]
    base = NEW.N.prepare_native_edge_problem(xyz, K, states, records, [], [480, 640],
        dict(R=R.tolist(), t=t.tolist(), physical_registration_Q=np.eye(3).tolist()),
        config=dict(prior_weight=0., analytic_jacobian=True), boundary_observations=boundary,
        u_regions=[], edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
    observation = dict(image_binding=dict(sha256='a'*64, image_hw=[480, 640], image_pixel_sha256='c'*64), edge_records=[])
    kwargs = dict(visible_measurements=records, initial_self_hidden=[s == 'S'for s in states],
        image_binding=deepcopy(observation['image_binding']), initial_projection_xy=projection.tolist(),
        normal_variance_component_px2=16., angular_variance_component_sine2=.002,
        noise_provenance=dict(fit_partition='CAL', model_digest_sha256='d'*64,
            observation_family=NEW.NOISE_OBSERVATION_FAMILY))
    return base, observation, kwargs, (k, a, b), R, t


def ray(base, guide, k, mate, *, points=None, offset=0.):
    start, end = np.asarray(guide[k]), np.asarray(guide[mate])
    direction = end-start; direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    if points is None:
        points = start+np.linspace(.15, .85, 15)[:, None]*(end-start)+offset*normal
    else:
        points = np.asarray(points)
        _, vectors = np.linalg.eigh(np.cov(points.T)); normal = vectors[:, 0]
    c = -float(normal@points.mean(axis=0))
    n = len(points)
    angle = float(np.degrees(np.arccos(np.clip(abs(normal@np.array([-direction[1], direction[0]])), 0., 1.))))
    internal = base.states[k] == 'I'
    chain = dict(native_corner=k, native_neighbor=mate, native_edge=[k, mate],
        measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT', invented_bridge_points=0,
        observed_points=points.tolist(), inlier_mask=[True]*n, normal=normal.tolist(), offset=c,
        inlier_count=n, inlier_fraction=1., inlier_span_fraction=1., angle_change_deg=angle,
        inlier_rms_px=float(np.sqrt(np.mean((points@normal+c)**2))),
        actual_inlier_image_lattice_xy=np.round(points).tolist(),
        RGB_gradient_energy=[8.]*n, RGB_alignment=[1.]*n, RGB_contrast=[20.]*n,
        mask_plus=[.99]*n, mask_minus=[.99 if internal else .01]*n,
        mask_context_kind='TARGET_TARGET_INTERNAL'if internal else 'TARGET_BACKGROUND_OUTLINE',
        corrected_endpoint_normal_guard_pass=True, corrected_endpoint_normal_residual_px={str(k):0.})
    return dict(native_corner=k, native_neighbor=mate, supported=True,
        measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT', response_is_not_window_clipped=True,
        chains=[chain])


def auth(base, observation, kwargs):
    return NEW.authenticate_complete_bi_image_chain_groups(base, observation, **kwargs)


def problem(base, bundle):
    return NEW.CompleteBIImageChainProblem(base, bundle, overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)


def test_complete_single_direction_replaces_full_point_once_and_keeps_exact_source_archive():
    base, obs, kw, (k, mate, _), R, t = fixture()
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, mate)]
    saved = deepcopy(kw['visible_measurements']); old_complete = list(base.complete)
    bundle = auth(base, obs, kw); p = problem(base, bundle)
    record = next(r for r in p.plan['corner_plans']if r['native_corner'] == k)
    assert record['tangent_kind'] == 'ARCHIVED_COMPLETE_BI_TANGENT'
    assert record['original_measurement_archive']['original_measurement_source'] == 'FROZEN_RGB_COMPLETE_SOURCE'
    tangent = np.array(record['tangent_unit_xy']); covariance = np.asarray(saved[0]['fit_covariance_px2'])
    assert record['tangent_variance_px2'] == pytest.approx(4.*tangent@covariance@tangent)
    assert k in p.revoked and k not in [m['native_corner']for m in p.complete]
    assert [m['native_corner']for m in base.complete] == [m['native_corner']for m in old_complete]
    assert kw['visible_measurements'] == saved and not bundle['groups'][0]['replaces_normal_factor_refs']
    at = NEW.N.V._pose_vector(R, t, base.scale)+[.01, -.02, .005, .008, -.011, .025]
    value, gradient, parts = p.objective(at)
    numeric = NEW.N.V._jacobian(lambda x: np.array([p.objective(x)[0]]), at)[0]
    assert np.allclose(gradient, numeric, rtol=4e-6, atol=4e-5)
    assert value == pytest.approx(sum(c['weighted_robust_cost']for c in parts['native_costs']))
    assert sum(c['kind'] == 'ARCHIVED_COMPLETE_BI_TANGENT'and c['native_corner'] == k for c in parts['native_costs']) == 1
    assert not any(c['kind'] == 'COMPLETE_BI'and c['native_corner'] == k for c in parts['native_costs'])
    oldguard, oldjac = base.constraints(at, base.sides[0]); newguard, newjac = p.constraints(at, p.sides[0])
    assert np.array_equal(oldguard, newguard)and np.array_equal(oldjac, newjac)
    assert p.evaluate(R, t)['objective'] == pytest.approx(p.objective(NEW.N.V._pose_vector(R, t, base.scale))[0])


def test_two_robust_nonparallel_directions_use_lines_jointly_without_fabricated_KP_or_tangent():
    base, obs, kw, (k, a, b), R, t = fixture()
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, m)for m in (a, b)]
    bundle = auth(base, obs, kw); p = problem(base, bundle)
    record = next(r for r in p.plan['corner_plans']if r['native_corner'] == k)
    assert record['point_cost_revoked']and len(record['active_physical_edges']) == 2
    assert 'tangent_kind'not in record and p.plan['new_full_2D_points'] == 0
    _, _, parts = p.objective(NEW.N.V._pose_vector(R, t, base.scale))
    assert not any(c['native_corner'] == k and c['kind']in ('COMPLETE_BI', 'ARCHIVED_COMPLETE_BI_TANGENT')for c in parts['native_costs'])


def test_all_alternatives_required_for_rank_not_one_favorable_pair():
    base, obs, kw, (k, a, b), _, _ = fixture()
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, m)for m in (a, b)]
    bundle = auth(base, obs, kw)
    # A geometric fixture of two ambiguous mixture directions: one good pair
    # is insufficient when another cross-combination is parallel.
    first = bundle['groups'][0]['components'][0]
    bundle['groups'][1]['components'].append({**deepcopy(first), 'native_corner':bundle['groups'][1]['native_corner'],
        'native_neighbor':bundle['groups'][1]['native_neighbor']})
    plan = NEW.plan_complete_bi_replacement(base, bundle, overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)
    record = next(r for r in plan['corner_plans']if r['native_corner'] == k)
    assert not record['point_cost_revoked']and record['direction_rank_ambiguous_replacement_abstained']
    assert not plan['active_groups']


def test_shared_reverse_ray_and_oldpartial_endpoint_replace_one_physical_factor_without_dummy_refs():
    base, obs, kw, (k, mate, _), R, t = fixture(partial=True)
    first = ray(base, kw['initial_projection_xy'], k, mate)
    reverse = ray(base, kw['initial_projection_xy'], mate, k,
        points=first['chains'][0]['observed_points'])
    reverse['chains'][0]['mask_context_kind'] = first['chains'][0]['mask_context_kind']
    reverse['chains'][0]['mask_minus'] = first['chains'][0]['mask_minus']
    obs['edge_records'] = [first, reverse]
    bundle = auth(base, obs, kw); p = problem(base, bundle)
    assert len(bundle['groups']) == len(p.groups) == 1 and len(p.groups[0]['components']) == 1
    assert p.plan['replaces_normal_factor_refs'] == [dict(native_corner=k, line_index=0)]
    assert len(base.remaining_normals) == 1 and not p.remaining_normals
    assert p.groups[0]['reliability_weight'] == min(m['fit_reliability_weight']for m in base.measurements if m['native_corner']in (k, mate))
    _, _, costs = p.objective(NEW.N.V._pose_vector(R, t, base.scale))
    assert sum(c['kind'] == 'PROJECTED_NATIVE_EDGE'for c in costs['native_costs']) == 1
    assert sum(c['kind'] == 'UNCERTAIN_ARCHIVED_TANGENT'and c['native_corner'] == k for c in costs['native_costs']) == 1


def test_cross_native_alias_overlap_is_archived_quantified_and_explicitly_abstained():
    base, obs, kw, _, _, _ = fixture()
    P = np.asarray(kw['initial_projection_xy'])
    edges = [(k, m)for k, state in enumerate(base.states)if state != 'S'
             for m in NEW.NEIGHBORS[k]if state != 'S'and base.states[m] != 'S'and k < m]
    pairs = [(a, b)for a, b in combinations(edges, 2)if not set(a)&set(b)
        and abs(np.dot((P[a[1]]-P[a[0]])/np.linalg.norm(P[a[1]]-P[a[0]]),
                       (P[b[1]]-P[b[0]])/np.linalg.norm(P[b[1]]-P[b[0]]))) > np.cos(np.deg2rad(12.))]
    (a, b), (c, d) = pairs[0]
    first = ray(base, P, a, b)
    second = ray(base, P, c, d, points=np.asarray(first['chains'][0]['observed_points'])[:8])
    obs['edge_records'] = [first, second]
    bundle = auth(base, obs, kw)
    overlap = bundle['cross_edge_support_overlaps'][0]
    assert overlap['shared_lattice_count'] == 8 and not overlap['identical_lattice_support']
    assert not overlap['incident_physical_edges']and len(bundle['groups']) == 2
    with pytest.raises(TypeError):NEW.plan_complete_bi_replacement(base, bundle)
    p = problem(base, bundle)
    assert not p.groups and not p.revoked and p.plan['overlap_policy_provisional_not_adopted']


def test_initial_S_and_nonBI_gates_precede_coordinate_reads_and_deepcopies():
    base, obs, kw, (k, mate, _), _, _ = fixture()
    hidden = next(i for i, s in enumerate(base.states)if s == 'S')
    obs['edge_records'] = [dict(native_corner=hidden, native_neighbor=k, chains=Poison()),
        dict(native_corner=k, native_neighbor=hidden, chains=Poison())]
    kw['initial_projection_xy'][hidden] = Poison()
    kw['visible_measurements'] = kw['visible_measurements']+[dict(native_corner=hidden, xy=Poison(), covariance_px2=Poison())]
    assert not auth(base, obs, kw)['groups']


@pytest.mark.parametrize('state', ['U', 'O', 'T'])
def test_nonBI_mate_supplies_model_geometry_only_not_poison_coordinate(state):
    base, obs, kw, (k, mate, _), _, _ = fixture()
    first = ray(base, kw['initial_projection_xy'], k, mate)
    # Authentication-only fixture: no pose is evaluated from this state view.
    view = copy(base); states = list(base.states); states[mate] = state; view.states = tuple(states)
    view.measurements = [m for m in base.measurements if m['native_corner'] != mate]
    view.complete = [m for m in base.complete if m['native_corner'] != mate]
    kw['visible_measurements'] = [r for r in kw['visible_measurements']if r['native_corner'] != mate]
    kw['visible_measurements'].append(dict(native_corner=mate, xy=Poison(), covariance_px2=Poison(), source=Poison()))
    obs['edge_records'] = [first, dict(native_corner=mate, native_neighbor=k, chains=Poison())]
    bundle = auth(view, obs, kw)
    assert len(bundle['groups']) == 1 and bundle['U_coordinate_reads'] == 0
    assert str(mate)not in bundle['frozen_measurement_archives']


def test_JSON_archive_roundtrip_and_same_ORIGINAL_guide_axis_are_stable():
    base, obs, kw, (k, mate, _), _, _ = fixture()
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, mate)]
    bundle = auth(base, obs, kw)
    original = NEW.plan_complete_bi_replacement(base, bundle, overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)
    restored = NEW.plan_complete_bi_replacement(base, json.loads(json.dumps(bundle)), overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)
    assert original == restored
    view = copy(base); view.xyz = np.asarray(base.xyz)[:, [2, 1, 0]]
    alternate = auth(view, obs, kw)
    other = NEW.plan_complete_bi_replacement(view, alternate, overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)
    left = next(r for r in original['corner_plans']if r['native_corner'] == k)
    right = next(r for r in other['corner_plans']if r['native_corner'] == k)
    assert left['tangent_unit_xy'] == right['tangent_unit_xy']
    assert left['tangent_variance_px2'] == right['tangent_variance_px2']


@pytest.mark.parametrize('change', ['old_model', 'TEST_model', 'image', 'RMS', 'RGB', 'mask', 'endpoint', 'guide', 'source', 'fitSigma'])
def test_wrong_generation_or_unbound_image_support_archive_rejected(change):
    base, obs, kw, (k, mate, _), _, _ = fixture()
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, mate)]
    c = obs['edge_records'][0]['chains'][0]
    if change == 'old_model':kw['noise_provenance']['observation_family'] = NEW.OLD.NOISE_OBSERVATION_FAMILY
    elif change == 'TEST_model':kw['noise_provenance']['fit_partition'] = 'TEST'
    elif change == 'image':kw['image_binding']['sha256'] = 'b'*64
    elif change == 'RMS':c['inlier_rms_px'] += .2
    elif change == 'RGB':c['RGB_gradient_energy'][0] = 0.
    elif change == 'mask':c['mask_plus'][0] = c['mask_minus'][0] = 0.
    elif change == 'endpoint':c['corrected_endpoint_normal_residual_px'][str(k)] = 12.
    elif change == 'guide':kw['initial_projection_xy'][mate] = (np.asarray(kw['initial_projection_xy'][k])+[0., 100.]).tolist()
    elif change == 'source':kw['visible_measurements'][0]['source'] = None
    elif change == 'fitSigma':kw['visible_measurements'][0]['fit_covariance_px2'][0][0] += 1.
    with pytest.raises(ValueError):auth(base, obs, kw)


def test_empty_exact_parent_and_preparation_cannot_be_scored():
    base, obs, kw, _, R, t = fixture()
    bundle = auth(base, obs, kw); p = problem(base, bundle)
    at = NEW.N.V._pose_vector(R, t, base.scale)
    old, new = base.objective(at), p.objective(at)
    assert old[0] == new[0]and np.array_equal(old[1], new[1])
    kw['calibration_preparation'] = True
    preparation = auth(base, obs, kw)
    with pytest.raises(ValueError, match='Fit/freeze'):problem(base, preparation)
