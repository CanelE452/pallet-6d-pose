"""Prediction-only guard boundaries with the actual frozen runtime schema."""
from copy import deepcopy
import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.research.pallet_guarded_bi_final_v1 import guard as G


def fixture():
    K = np.array([[100., 0, 80.], [0, 100., 60.], [0, 0, 1.]])
    Q = np.array([[0., 0, 1.], [0, 1., 0], [-1., 0, 0]])
    coarse = dict(available=True, R_cf=np.eye(3).tolist(), R_physical=Q.tolist(),
        centroid=[0., 0., 3.], cf_extents=[1.2, .2, .8],
        source='UNCHANGED_INITIAL', arbitrary_metadata={'preserve': [1, 2, 'e']})
    candidate = deepcopy(coarse)
    candidate.update(centroid=[.03, .03, 3.], source='CORRECTED_BI_ONLY_PNP')
    X = G._cuboid(coarse['cf_extents'])
    p0, p1 = G._project(X, coarse, K), G._project(X, candidate, K)
    raw = np.vstack([p0, [80., 60.]])
    states = ['B'] * 4 + ['S', 'O', 'T', 'U']
    rows, measurements = [], []
    for k, state in enumerate(states):
        record = dict(native_corner=k, state=state, pixel_queried=state == 'B',
                      corrected=False, reason='excluded_before_image_query')
        if state == 'B':
            record.update(method='B_TARGET_BACKGROUND_LINES', raw_xy=raw[k].tolist(),
                measured_xy=p1[k].tolist(), corrected=True,
                covariance_px2=np.eye(2).tolist(), reason='actual_image_intersection', rays=[])
            measurements.append(dict(native_corner=k, state=state, xy=p1[k].tolist(),
                covariance_px2=np.eye(2).tolist(), protection_px=2., source='B_TARGET_BACKGROUND_LINES',
                image_measurement=True, measurement_is_pose_projection=False))
        rows.append(record)
    # This ray's image samples lie on the candidate edge, so its error reduces.
    k, mate = 0, 1
    points = p1[k] + np.linspace(.1, .8, 9)[:, None] * (p1[mate] - p1[k])
    direction = p1[mate] - p1[k]
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    rows[0]['rays'] = [dict(edge=k * 8 + mate, endpoints=[k, mate], native_corner=k,
        native_neighbor=mate, supported=True, measurement_source='OBSERVED_TARGET_BACKGROUND_TRANSITION',
        guide_source='initial_registered_model_direction_only', normal=normal.tolist(),
        offset=-float(normal @ points[0]), observed_points=points.tolist(),
        observation_confidence=[.2, .4, .6, .8, 1., .8, .6, .4, .2],
        inlier_mask=[False] + [True] * 7 + [False])]
    correction = dict(schema='visible_BI_measurements_boundary_recovery_v4_3b',
        image_pixel_sha256='a' * 64, records=rows,
        provenance=dict(S_queried=False, projected_coordinates_used_as_measurements=False,
            image_coordinates_frozen_before_joint_OT=True, recovery_RGB_support_required=True))
    observation = dict(schema='STATEWISE_RGB_OBSERVATIONS_V4', states=states,
        original_raw_points=raw.tolist(), frozen_visible_measurements=measurements,
        initial_geometry=dict(native_xyz=X.tolist(), R_cf=deepcopy(coarse['R_cf']),
            centroid=deepcopy(coarse['centroid']), physical_registration_Q=Q.tolist(), K=K.tolist()),
        visible_correction=correction,
        provenance=dict(GT_inputs=False, current_RGB_state_inference=True, S_image_queries=0))
    solved = dict(actual_pose=candidate, accepted=True, input_native_ids=[0, 1, 2, 3],
        observed_states_preserved=states, frozen_visible_measurements=deepcopy(measurements),
        S_image_terms=0, O_T_U_are_not_PnP_points=True,
        solution=dict(accepted=True, rank={'visible': {'rank': 6}},
            ambiguity=dict(detected=False, local_underdetermined=False, pose_family_from_two_BI=False)))
    return observation, solved, coarse, K


def refresh_solved(observation, solved):
    solved['frozen_visible_measurements'] = deepcopy(observation['frozen_visible_measurements'])
    solved['observed_states_preserved'] = deepcopy(observation['states'])
    solved['input_native_ids'] = [m['native_corner'] for m in observation['frozen_visible_measurements']]


def test_valid_A_and_B_select_same_real_candidate_without_mutation():
    o, s, p, K = fixture()
    before = deepcopy((o, s, p, K.tolist()))
    for mode in G.MODES:
        r = G.apply_guard(o, s, p, K, mode)
        assert r['accepted'] and not r['fallback']
        assert r['status'] == 'guarded_BI_replacement'
        assert r['actual_pose'] == s['actual_pose']
        r['actual_pose']['centroid'][0] = 100
        assert s['actual_pose']['centroid'][0] == .03
    assert before == (o, s, p, K.tolist())


@pytest.mark.parametrize('mutation', ['rejected', 'rank', 'missing_rank', 'ambiguous', 'local', 'twoBI', 'missing_flag'])
def test_A_rejects_feasible_but_ambiguous_or_unidentified_solver(mutation):
    o, s, p, K = fixture()
    if mutation == 'rejected': s['solution']['accepted'] = False
    elif mutation == 'rank': s['solution']['rank']['visible']['rank'] = 5
    elif mutation == 'missing_rank': s['solution'].pop('rank')
    elif mutation == 'missing_flag': s['solution']['ambiguity'].pop('local_underdetermined')
    else: s['solution']['ambiguity'][{'ambiguous': 'detected', 'local': 'local_underdetermined', 'twoBI': 'pose_family_from_two_BI'}[mutation]] = True
    r = G.apply_guard(o, s, p, K, 'A')
    assert not r['accepted'] and r['actual_pose'] == p
    assert r['status'] == 'exact_own_baseline_retained'
    r['actual_pose']['arbitrary_metadata']['preserve'][0] = 999
    assert p['arbitrary_metadata']['preserve'][0] == 1


def test_A_requires_four_unique_BI_and_preserves_excluded_states():
    o, s, p, K = fixture()
    o['states'][3] = 'O'
    o['visible_correction']['records'][3]['state'] = 'O'
    o['frozen_visible_measurements'].pop()
    refresh_solved(o, s)
    assert not G.apply_guard(o, s, p, K, 'A')['accepted']
    for state in ('S', 'O', 'T', 'U'):
        o, s, p, K = fixture()
        o['frozen_visible_measurements'][0]['native_corner'] = o['states'].index(state)
        refresh_solved(o, s)
        assert not G.apply_guard(o, s, p, K, 'A')['accepted']


@pytest.mark.parametrize('change', ['duplicate', 'foreign_K', 'foreign_geometry', 'changed_Q', 'candidate_hidden_term', 'foreign_measurements'])
def test_foreign_or_reindexed_observation_solver_proposal_is_rejected(change):
    o, s, p, K = fixture()
    if change == 'duplicate':
        o['frozen_visible_measurements'][1]['native_corner'] = 0
        refresh_solved(o, s)
    elif change == 'foreign_K': K[0, 0] += 1
    elif change == 'foreign_geometry': o['initial_geometry']['centroid'][0] = 1
    elif change == 'changed_Q': s['actual_pose']['R_physical'] = np.eye(3).tolist()
    elif change == 'candidate_hidden_term': s['S_image_terms'] = 1
    elif change == 'foreign_measurements': s['frozen_visible_measurements'][0]['xy'][0] += 1
    assert not G.apply_guard(o, s, p, K, 'A')['accepted']


def test_backbone_fallback_flag_and_pose_projection_do_not_count_as_new_image_correction():
    o, s, p, K = fixture()
    for measurement in o['frozen_visible_measurements']:
        measurement['source'] = 'VISIBLE_BACKBONE_FALLBACK'
    refresh_solved(o, s)
    r = G.apply_guard(o, s, p, K, 'A')
    assert not r['accepted']
    assert all(not m['moved'] for m in r['decision_audit']['measurement_records'])
    o, s, p, K = fixture()
    o['frozen_visible_measurements'][0]['measurement_is_pose_projection'] = True
    refresh_solved(o, s)
    assert not G.apply_guard(o, s, p, K, 'A')['accepted']


def test_zero_actual_movement_is_not_a_real_update_even_when_corrected_flags_are_true():
    o, s, p, K = fixture()
    for m in o['frozen_visible_measurements']:
        k = m['native_corner']
        m['xy'] = deepcopy(o['original_raw_points'][k])
        o['visible_correction']['records'][k]['measured_xy'] = deepcopy(m['xy'])
    refresh_solved(o, s)
    assert not G.apply_guard(o, s, p, K, 'A')['accepted']


def test_B_uses_all_observed_samples_confidence_and_correct_CF_projection():
    o, s, p, K = fixture()
    ray = o['visible_correction']['records'][0]['rays'][0]
    ray['observed_points'][0][1] += .4  # fitted-outlier sample still participates
    r = G.apply_guard(o, s, p, K, 'B')
    a = r['decision_audit']['ray_records'][0]
    X = G._cuboid(p['cf_extents'])
    weights = np.array(ray['observation_confidence'])
    points = np.array(ray['observed_points'])
    for label, pose in [('baseline', p), ('candidate', s['actual_pose'])]:
        projected = G._project(X, pose, K)
        d = projected[1] - projected[0]
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        residual = np.abs((points - projected[0]) @ n)
        loss = np.where(residual <= 1.25, .5 * residual**2, 1.25 * (residual - .625))
        expected = float(weights @ loss / weights.sum())
        assert a[label + '_cost'] == pytest.approx(expected, abs=1e-14)
    assert a['sample_count'] == 9
    # Its physical rotation is intentionally different from R_cf.
    wrong = deepcopy(s['actual_pose']); wrong['R_cf'] = wrong['R_physical']
    assert not np.allclose(G._project(X, wrong, K), G._project(X, s['actual_pose'], K))


@pytest.mark.parametrize('bad', ['unsupported', 'fake', 'neighbor', 'diagonal', 'nan', 'zero_weights', 'curve'])
def test_B_excludes_invalid_or_unsupported_rays_and_records_reason(bad):
    o, s, p, K = fixture()
    ray = o['visible_correction']['records'][0]['rays'][0]
    if bad == 'unsupported': ray['supported'] = False
    elif bad in ('fake', 'curve'): ray['measurement_source'] = 'SYNTHETIC_MODEL_CURVE'
    elif bad == 'neighbor': ray['native_neighbor'] = 8
    elif bad == 'diagonal': ray.update(native_neighbor=2, endpoints=[0, 2], edge=2)
    elif bad == 'nan': ray['observed_points'][0][0] = float('nan')
    elif bad == 'zero_weights': ray['observation_confidence'] = [0.] * 9
    r = G.apply_guard(o, s, p, K, 'B')
    assert not r['accepted'] and r['reason'] == 'rule_B_no_valid_observed_rays'
    assert r['decision_audit']['ray_records'][0]['status'] == 'excluded'
    assert r['decision_audit']['ray_records'][0]['reason']


def test_B_rejects_one_worse_ray_even_when_other_ray_improves_aggregate():
    o, s, p, K = fixture()
    ray = deepcopy(o['visible_correction']['records'][0]['rays'][0])
    # New samples exactly on original edge force its candidate residual to rise.
    X = G._cuboid(p['cf_extents']); proj = G._project(X, p, K)
    ray['observed_points'] = (proj[0] + np.linspace(.1, .8, 9)[:, None] * (proj[1] - proj[0])).tolist()
    ray['observation_confidence'] = [.001] * 9
    o['visible_correction']['records'][0]['rays'].append(ray)
    r = G.apply_guard(o, s, p, K, 'B')
    assert not r['accepted'] and r['reason'] == 'rule_B_observed_ray_worsened'
    assert r['decision_audit']['line_cost']['strict_improvement']


def test_B_requires_strict_improvement_and_no_valid_rays_is_fail_closed():
    o, s, p, K = fixture()
    s['actual_pose'] = deepcopy(p)
    r = G.apply_guard(o, s, p, K, 'B')
    assert not r['accepted'] and r['reason'] == 'rule_B_no_strict_aggregate_improvement'
    o, s, p, K = fixture()
    o['visible_correction']['records'][0]['rays'] = []
    assert G.apply_guard(o, s, p, K, 'B')['reason'] == 'rule_B_no_valid_observed_rays'


@pytest.mark.parametrize('kind', ['interior', 'recovery'])
def test_actual_interior_and_RGB_recovery_ray_schema_and_recovery_provenance(kind):
    o, s, p, K = fixture()
    record = o['visible_correction']['records'][0]
    ray = record['rays'][0]
    if kind == 'interior':
        o['states'][0] = record['state'] = o['frozen_visible_measurements'][0]['state'] = 'I'
        record['method'] = o['frozen_visible_measurements'][0]['source'] = 'I_INTERNAL_RGB_JUNCTION'
        ray['measurement_source'] = 'OBSERVED_INTERNAL_RGB_GRADIENT'
    else:
        record['method'] = o['frozen_visible_measurements'][0]['source'] = 'B_ACTUAL_BOUNDARY_RECOVERY'
        ray.update(measurement_source='OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION',
                   shared_native_edge=[0, 1], rgb_supported_sample_count=9)
    refresh_solved(o, s)
    assert G.apply_guard(o, s, p, K, 'B')['accepted']
    if kind == 'recovery':
        o['visible_correction']['provenance']['recovery_RGB_support_required'] = False
        r = G.apply_guard(o, s, p, K, 'B')
        assert not r['accepted']
        assert r['decision_audit']['ray_records'][0]['reason'] == 'missing_actual_RGB_recovery_support'


def test_missing_pose_and_provenance_fail_closed_preserving_exact_own_baseline():
    o, s, p, K = fixture()
    p = {'available': False, 'source': 'EXISTING_NO_POSE', 'metadata': [1, 'e']}
    r = G.apply_guard(o, s, p, K, 'A')
    assert r['actual_pose'] == p and not r['accepted']
    o, s, p, K = fixture()
    o['provenance']['current_RGB_state_inference'] = False
    assert G.apply_guard(o, s, p, K, 'A')['reason'] == 'invalid_current_RGB_observation_provenance'


def test_guard_has_no_io_gt_or_model_runtime_dependency():
    tree = ast.parse(Path(G.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all((node.module or '').split('.')[0] in {'__future__', 'copy'} if isinstance(node, ast.ImportFrom)
               else all(n.name in {'re', 'numpy'} for n in node.names) for node in imports)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'open', 'eval', 'exec'}
                   for n in ast.walk(tree))
    with pytest.raises(ValueError, match='exactly A or B'):
        G.apply_guard({}, {}, {}, [], 'NEW_TUNED_RULE')
