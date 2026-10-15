"""Independent V30 selected-pose arithmetic and original-B/I association audit.

The frozen V29 independent audit already reconstructs the unchanged factor,
Jacobian, O/S/C and support mathematics without the runtime consumer. Its
driver is privately rebound only to new names/model paths and one additional
per-frame association check. No runtime AUTH/planner/objective is called.
"""
import ast
import inspect
from types import FunctionType

import numpy as np

from . import audit_geometry_guided_edges_v4_29 as PARENT
from . import common as C

PARENT_SHA256 = '4e032e591faa1a8a2e34ec0ede74fbee7bacb58cb05807b3f016dbc0d2c3f44a'
PROFILE = 'ALL_ELIGIBLE_NAMED_BI_ENDPOINTS_TO_ACTUAL_INFINITE_TLS_WITHIN_EXISTING_NORMAL_REACH_V30'
FAMILY = 'OBSERVED_BI_ASSOCIATED_MULTIGUIDE_ACTUAL_NATIVE_EDGES_V30'
POPULATION = 'V30_ASSOCIATION_GATED_RUNTIME_ACTIVE_DETERMINISTIC_ONE_LINE_PER_PHYSICAL_EDGE'
ARM_REPLACEMENTS = {
    'GEOMETRY_EDGE_ALL_BI_NATIVE': 'ASSOCIATED_EDGE_ALL_BI_NATIVE',
    'GEOMETRY_EDGE_ALL_BI_CONTAINMENT': 'ASSOCIATED_EDGE_ALL_BI_CONTAINMENT',
    'GEOMETRY_EDGE_SOURCE_NATIVE': 'ASSOCIATED_EDGE_SOURCE_NATIVE',
    'GEOMETRY_EDGE_SOURCE_CONTAINMENT': 'ASSOCIATED_EDGE_SOURCE_CONTAINMENT'}
REPLACEMENTS = {**ARM_REPLACEMENTS,
    'geometry_guided_edge_eval_v4_29': 'geometry_guided_edge_eval_v4_30',
    'CAL_geometry_guided_edge_noise_v4_29': 'CAL_geometry_guided_edge_noise_v4_30',
    'INDEPENDENT_SELECTED_GEOMETRY_EDGE_AUDIT_V29': 'INDEPENDENT_SELECTED_ASSOCIATED_EDGE_AUDIT_V30'}


def _check_association_frame(task, prediction, check, ident):
    observation = task['v29_observation']
    check(observation.get('selection_profile') == PROFILE, 'new V30 graph selection profile', ident)
    check(observation.get('observation_generation') == 'FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29',
        'unchanged actual V29 raw measurements', ident)
    configuration = observation['association_configuration']
    check(configuration == observation['chain_observation']['configuration'],
        'association configuration equals original measurement configuration', ident)
    limit = float(configuration['normal_search_radius_px'])
    check(limit == 43., 'existing unchanged43px normal reach', ident)
    q = {}
    for measurement in task['spec']['visible']:
        k = measurement['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B', 'I'):
            raise ValueError('Initial S/BI gate before independent observed coordinate')
        q[k] = np.asarray(measurement['xy'], float)
    records = observation['chain_observation']['edge_records']
    nodes = {n['name']: n for n in observation['joint_graph']['line_candidates']}
    for ref in observation['selected_chain_refs']:
        ri = ref['ray_index']
        if isinstance(ri, bool) or not isinstance(ri, int) or not 0 <= ri < len(records):
            raise ValueError('Exact nonnegative original ray index required')
        ray = records[ri]
        k, mate = ray['native_corner'], ray['native_neighbor']
        if (task['initial_self_hidden'][k] or task['initial_self_hidden'][mate]
                or task['states'][k] not in ('B', 'I') or task['states'][mate] == 'S'):
            raise ValueError('S gate before independent selected line fields')
        node = nodes[ref['line_name']]
        check(sorted(node['native_edge']) == sorted([k, mate]), 'selected node original native IDs', ident)
        normal = np.asarray(node['normal'], float)
        offset = float(node['offset'])
        check(normal.shape == (2,) and np.isfinite([*normal, offset]).all()
            and np.isclose(normal @ normal, 1., atol=1e-10, rtol=0),
            'finite unit actual TLS in association check', ident)
        eligible = [i for i in (k, mate) if task['states'][i] in ('B', 'I') and i in q]
        check(bool(eligible), 'at least one original eligible B/I endpoint', ident)
        for i in eligible:
            check(abs(float(normal @ q[i] + offset)) <= limit,
                'all original named B/I endpoints within infinite TLS normal reach', ident)
    for arm in ARM_REPLACEMENTS.values():
        plan = prediction['arms'][arm]['solution']['v29_cost_plan']
        check(plan.get('selection_profile') == PROFILE and plan['observation_family'] == FAMILY
            and plan['population'] == POPULATION, 'new V30 runtime plan population including fallback', ident, arm)


class _VersionDriver(ast.NodeTransformer):
    def __init__(self):
        self.replacements = {key: 0 for key in REPLACEMENTS}
        self.imports = self.frame_hooks = 0

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in REPLACEMENTS:
            self.replacements[node.value] += 1
            return ast.copy_location(ast.Constant(REPLACEMENTS[node.value]), node)
        return node

    def visit_ImportFrom(self, node):
        if node.module is None and node.level == 1:
            for alias in node.names:
                if alias.name == 'geometry_guided_edge_noise_calibration_v4_29':
                    self.imports += 1
                    alias.name = 'geometry_guided_edge_noise_calibration_v4_30'
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        for index, statement in enumerate(node.body):
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == 'ident'):
                hook = ast.parse('_check_association_frame(task,row,check,ident)').body[0]
                node.body.insert(index + 1, hook)
                self.frame_hooks += 1
                break
        return node


def _build_main():
    if C.sha(PARENT.__file__) != PARENT_SHA256:
        raise ValueError('Exact frozen independent numerical audit source required')
    tree = ast.parse(inspect.getsource(PARENT.main))
    transform = _VersionDriver()
    tree = transform.visit(tree)
    if (set(transform.replacements.values()) != {1}
            or transform.imports != 1 or transform.frame_hooks != 1):
        raise ValueError('Only declared driver names, model import and one frame hook may change')
    ast.fix_missing_locations(tree)
    namespace = {**PARENT.main.__globals__, '__file__': __file__, '__name__': __name__,
        '_check_association_frame': _check_association_frame}
    code = compile(tree, __file__, 'exec')
    main_code = next(value for value in code.co_consts if isinstance(value, type(PARENT.main.__code__)))
    function = FunctionType(main_code, namespace, 'independent_V30_driver', PARENT.main.__defaults__)
    return function, transform


def main(output=None):
    function, _ = _build_main()
    return function(output)


if __name__ == '__main__':
    main()
