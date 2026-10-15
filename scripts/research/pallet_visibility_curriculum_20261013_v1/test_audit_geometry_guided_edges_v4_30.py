"""Independent association/state checks and strictly bounded driver changes."""
from copy import deepcopy
import inspect
import ast

import pytest

from . import audit_geometry_guided_edges_v4_30 as A


def fixture(mate='I'):
    states = ['B', mate] + ['S'] * 6
    configuration = dict(normal_search_radius_px=43.)
    obs = dict(selection_profile=A.PROFILE,
        observation_generation='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29',
        association_configuration=configuration,
        chain_observation=dict(configuration=deepcopy(configuration),
            edge_records=[dict(native_corner=0, native_neighbor=1)]),
        selected_chain_refs=[dict(ray_index=0, line_name='actual')],
        joint_graph=dict(line_candidates=[dict(name='actual', native_edge=[0, 1],
            normal=[0., 1.], offset=-19.)]))
    task = dict(v29_observation=obs, states=states, initial_self_hidden=[s == 'S' for s in states],
        spec=dict(visible=[dict(native_corner=0, xy=[-1000., 0.])]
            + ([dict(native_corner=1, xy=[1000., 0.])] if mate == 'I' else [])))
    plan = dict(selection_profile=A.PROFILE, observation_family=A.FAMILY, population=A.POPULATION)
    row = dict(arms={arm: dict(accepted=False, solution=dict(v29_cost_plan=deepcopy(plan)))
        for arm in A.ARM_REPLACEMENTS.values()})
    return task, row


def check(task, row):
    results = []
    A._check_association_frame(task, row,
        lambda passed, reason, ident, arm=None: results.append((passed, reason)), 'synthetic')
    return results


def test_exact_driver_changes_are_only_versions_import_and_one_frame_hook():
    before = dict(A.PARENT.main.__globals__)
    function, transform = A._build_main()
    assert set(transform.replacements.values()) == {1}
    assert transform.imports == transform.frame_hooks == 1
    assert function.__globals__['recompute_native'] is A.PARENT.recompute_native
    assert function.__globals__['observation_checks'] is A.PARENT.observation_checks
    assert function.__globals__['_check_association_frame'] is A._check_association_frame
    assert all(A.PARENT.main.__globals__[key] is value for key, value in before.items())
    constants = set(value for value in function.__code__.co_consts if isinstance(value, str))
    assert not constants & set(A.REPLACEMENTS)


def test_parent_import_ast_shape_is_explicitly_handled():
    tree = ast.parse(inspect.getsource(A.PARENT.main))
    source = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and any(a.name == 'geometry_guided_edge_noise_calibration_v4_29' for a in node.names)]
    assert len(source) == 1 and source[0].module is None and source[0].level == 1


@pytest.mark.parametrize('mate', ['I', 'U'])
def test_independent_normal19_with_arbitrary_alongline_displacement_and_U_no_query(mate):
    task, row = fixture(mate)
    assert all(passed for passed, _ in check(task, row))


def test_bad_association_is_reported_even_if_every_pose_arm_falls_back():
    task, row = fixture()
    task['v29_observation']['joint_graph']['line_candidates'][0]['offset'] = -500.
    failed = [reason for passed, reason in check(task, row) if not passed]
    assert failed.count('all original named B/I endpoints within infinite TLS normal reach') == 2


def test_initial_S_gate_precedes_missing_node_or_line_fields():
    task, row = fixture('S')
    task['v29_observation']['joint_graph']['line_candidates'] = []
    with pytest.raises(ValueError, match='S gate'):
        check(task, row)


def test_new_runtime_population_is_checked_for_fallback_arms_too():
    task, row = fixture()
    arm = next(iter(row['arms']))
    row['arms'][arm]['solution']['v29_cost_plan']['population'] = 'OLD_V29'
    assert any(not passed and 'including fallback' in reason for passed, reason in check(task, row))
