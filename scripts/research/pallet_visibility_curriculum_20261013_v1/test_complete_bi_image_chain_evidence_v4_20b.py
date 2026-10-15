"""Cached-pixel relation and typed finite/topology contracts; no real data."""
from copy import deepcopy
import numpy as np
import pytest

from . import complete_bi_image_chain_evidence_v4_20b as NEW
from .test_complete_bi_image_chain_evidence_v4_20 import fixture as parent_fixture, Poison


def ray(base, P, k, mate):
    P = np.asarray(P); q = P[k]; vector = P[mate]-q
    length = np.linalg.norm(vector); tangent = vector/length; normal = np.array([-tangent[1], tangent[0]])
    c = -float(normal@q)
    lattice = np.round(q+np.linspace(4., min(36., .8*length), 25)[:, None]*tangent)
    _, indices = np.unique(lattice, axis=0, return_index=True); lattice = lattice[np.sort(indices)]
    shift = -(lattice@normal+c)
    lattice, shift = lattice[np.abs(shift) <= .5], shift[np.abs(shift) <= .5]
    points = lattice+shift[:, None]*normal
    n = len(points); assert n >= 6
    internal = base.states[k] == 'I'
    chain = dict(native_corner=k, native_neighbor=mate, native_edge=[k, mate],
        measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT', invented_bridge_points=0,
        observed_points=points.tolist(), inlier_mask=[True]*n, normal=normal.tolist(), offset=c,
        inlier_count=n, inlier_fraction=1., inlier_span_fraction=1., angle_change_deg=0.,
        inlier_rms_px=float(np.sqrt(np.mean((points@normal+c)**2))),
        actual_inlier_image_lattice_xy=lattice.tolist(), image_gradient_normal_xy=[normal.tolist()]*n,
        measured_subpixel_shift_px=shift.tolist(), RGB_gradient_energy=[8.]*n,
        RGB_alignment=[1.]*n, RGB_contrast=[20.]*n, mask_plus=[.99]*n,
        mask_minus=[.99 if internal else .01]*n,
        mask_context_kind='TARGET_TARGET_INTERNAL'if internal else 'TARGET_BACKGROUND_OUTLINE',
        corrected_endpoint_normal_guard_pass=True, corrected_endpoint_normal_residual_px={str(k):0.})
    return dict(native_corner=k, native_neighbor=mate, supported=True,
        measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT', response_is_not_window_clipped=True, chains=[chain])


def candidate(rays, k, P):
    first, second = [r['chains'][0]for r in rays[:2]]
    matrix = np.asarray([first['normal'], second['normal']]); point = np.linalg.solve(matrix, -np.asarray([first['offset'], second['offset']]))
    ends = []
    for c in (first, second):
        n = np.asarray(c['normal']); tangent = np.array([-n[1], n[0]])
        values = (np.asarray(c['observed_points'])-point)@tangent
        ends.append(dict(nearest_actual_support_endpoint_tangent_distance_px=min(abs(float(values.min())), abs(float(values.max()))),
            actual_support_one_sided_fraction=max(float(np.mean(values >= 0)), float(np.mean(values <= 0))), passed=True))
    inward = np.array([1., 1.])/np.sqrt(2.)
    return dict(chain_refs=[dict(ray_index=i, chain_index=0, native_corner=k, native_neighbor=r['native_neighbor'])for i, r in enumerate(rays[:2])],
        intersection_xy=point.tolist(), intersection_sine=abs(float(np.linalg.det(matrix))),
        shift_from_archived_B_observation_px=float(np.linalg.norm(point-np.asarray(P[k]))),
        local_topology=dict(passed=True, chain_end_support=ends,
            local_external_contours=[dict(component=1, junction_distance_to_actual_external_contour_px=0., shortest_actual_external_contour_arc_px=0.)],
            inside_direction_from_actual_chain_context=inward.tolist(),
            actual_mask_inside_outside_probe_xy=[(point+7*inward).tolist(), (point-7*inward).tolist()],
            actual_mask_inside_outside_values=[.99, .01]))


def fixture(pair=True):
    base, obs, kw, (k, a, b), R, t = parent_fixture()
    kw['noise_provenance']['observation_family'] = NEW.NOISE_OBSERVATION_FAMILY
    obs['configuration'] = deepcopy(NEW.OBS.DEFAULT_CONFIGURATION)
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, a)]
    if pair:obs['edge_records'].append(ray(base, kw['initial_projection_xy'], k, b))
    topology = dict(schema='NATIVE_EDGE_TOPOLOGY_JUNCTION_PROPOSALS_V4_17B',
        configuration=deepcopy(NEW.OBS.DEFAULT_CONFIGURATION), image_binding=deepcopy(obs['image_binding']),
        mask_binding=dict(image_sha256=kw['image_binding']['sha256'], mask_pixel_sha256='e'*64),
        records=[dict(native_corner=i, state=s, candidates=[])for i, s in enumerate(base.states)])
    if pair:topology['records'][k]['candidates'] = [candidate(obs['edge_records'], k, kw['initial_projection_xy'])]
    kw['junction_proposals'] = topology
    return base, obs, kw, (k, a, b), R, t


def auth(base, obs, kw):
    return NEW.authenticate_complete_bi_image_chain_groups(base, obs, **kw)


def plan(base, bundle):
    return NEW.plan_complete_bi_replacement(base, bundle, overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)


def test_exact_lattice_relation_and_all_alternative_B_topology_pair():
    base, obs, kw, (k, _, _), R, t = fixture()
    bundle = auth(base, obs, kw); result = plan(base, bundle)
    assert max(r['maximum_relation_error_px']for r in bundle['actual_lattice_gradient_subpixel_relation']) < 1e-7
    record = next(r for r in result['corner_plans']if r['native_corner'] == k)
    assert record['point_cost_revoked']and len(record['active_physical_edges']) == 2 and 'tangent_kind'not in record
    assert result['B_archived_actual_topology_bound_at_exact_chain_refs']
    p = NEW.CompleteBIImageChainProblem(base, bundle, overlap_policy=NEW.PROTOTYPE_OVERLAP_POLICY)
    at = NEW.N.V._pose_vector(R, t, base.scale)+[.01, -.02, .005, .008, -.011, .025]
    value, gradient, parts = p.objective(at)
    numeric = NEW.N.V._jacobian(lambda x:np.array([p.objective(x)[0]]), at)[0]
    assert np.allclose(gradient, numeric, atol=4e-5, rtol=4e-6)
    assert value == pytest.approx(sum(c['weighted_robust_cost']for c in parts['native_costs']))
    a, j = p.constraints(at, p.sides[0]); b, jj = base.constraints(at, base.sides[0])
    assert np.array_equal(a, b)and np.array_equal(j, jj)


@pytest.mark.parametrize('change', ['lattice', 'gradient', 'shift', 'mask_binding', 'B_topology_point', 'B_topology_wedge'])
def test_unbound_pixel_or_topology_receipt_is_rejected(change):
    base, obs, kw, (k, _, _), _, _ = fixture()
    chain = obs['edge_records'][0]['chains'][0]
    if change == 'lattice':chain['actual_inlier_image_lattice_xy'][0][0] += 1
    elif change == 'gradient':chain['image_gradient_normal_xy'][0] = [2., 0.]
    elif change == 'shift':chain['measured_subpixel_shift_px'][0] = .7
    elif change == 'mask_binding':kw['junction_proposals']['mask_binding']['image_sha256'] = 'b'*64
    elif change == 'B_topology_point':kw['junction_proposals']['records'][k]['candidates'][0]['intersection_xy'][0] += 1
    elif change == 'B_topology_wedge':kw['junction_proposals']['records'][k]['candidates'][0]['local_topology']['actual_mask_inside_outside_values'] = [.5, .5]
    with pytest.raises(ValueError):auth(base, obs, kw)


def test_rank2_without_any_external_topology_does_not_revoke_complete_B():
    base, obs, kw, (k, _, _), _, _ = fixture()
    kw['junction_proposals']['records'][k]['candidates'] = []
    result = plan(base, auth(base, obs, kw))
    record = next(r for r in result['corner_plans']if r['native_corner'] == k)
    assert not record['point_cost_revoked']and not result['active_groups']
    assert result['main_replacement_rejections']


def test_single_direction_self_type_required_opposite_anchor_not_silently_promoted():
    base, obs, kw, (k, a, _), _, _ = fixture(False)
    good = plan(base, auth(base, obs, kw))
    assert next(r for r in good['corner_plans']if r['native_corner'] == k)['tangent_kind'] == 'ARCHIVED_COMPLETE_BI_TANGENT'
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], a, k)]
    result = plan(base, auth(base, obs, kw))
    assert not next(r for r in result['corner_plans']if r['native_corner'] == k)['point_cost_revoked']
    assert any(r['reason'] == 'not_all_alternatives_have_role_correct_self_anchor'for r in result['main_replacement_rejections'])


def test_extra_unsupported_B_alternative_requires_its_own_finite_topology_receipt():
    base, obs, kw, (k, _, _), _, _ = fixture()
    extra = deepcopy(obs['edge_records'][0]['chains'][0])
    # A genuine second parallel image-lattice path, without a B junction
    # topology receipt, may not be silently discarded in favor of the first.
    extra['observed_points'] = (np.asarray(extra['observed_points'])+[0., 1.]).tolist()
    extra['actual_inlier_image_lattice_xy'] = (np.asarray(extra['actual_inlier_image_lattice_xy'])+[0., 1.]).tolist()
    extra['offset'] -= extra['normal'][1]
    extra['inlier_rms_px'] = float(np.sqrt(np.mean((np.asarray(extra['observed_points'])@np.asarray(extra['normal'])+extra['offset'])**2)))
    obs['edge_records'][0]['chains'].append(extra)
    result = plan(base, auth(base, obs, kw))
    assert not next(r for r in result['corner_plans']if r['native_corner'] == k)['point_cost_revoked']


def test_hidden_S_gate_precedes_new_pixel_relation_and_topology_reads():
    base, obs, kw, (k, _, _), _, _ = fixture(False)
    hidden = next(i for i, s in enumerate(base.states)if s == 'S')
    obs['edge_records'].append(dict(native_corner=hidden, native_neighbor=k, chains=Poison()))
    kw['junction_proposals']['records'][hidden]['candidates'] = Poison()
    assert auth(base, obs, kw)['S_coordinate_reads'] == 0
