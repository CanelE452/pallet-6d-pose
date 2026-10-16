"""Independent V31 saved-pose audit: frozen V29 numeric kernel plus raw query checks.

Only driver names, the strict M31 exporter import and one per-frame hook are
changed by AST. The hook uses scalar dot/hypot arithmetic on original BI q,
saved raw ray/chain query metadata and ORIGINAL initial model rays. It never
calls runtime F31 admission, AUTH, planner, objective or support constraints.
Decorated nodes have no line_record: original indexed chains are authoritative.
"""
import ast
import inspect
import math
from numbers import Integral
from types import FunctionType

from . import audit_geometry_guided_edges_v4_29 as PARENT
from . import common as C

PARENT_SHA256 = '4e032e591faa1a8a2e34ec0ede74fbee7bacb58cb05807b3f016dbc0d2c3f44a'
PROFILE = 'OBSERVED_BI_Q_QUERY_ANCHOR_AND_FIXED_SIGNED_NATIVE_RAY_PLUS_NORMAL_REACH_V31'
FAMILY = 'OBSERVED_Q_ANCHORED_FIXED_NATIVE_RAY_ACTUAL_EDGES_V31'
POPULATION = 'V31_Q_ANCHORED_SIGNED_RAY_RUNTIME_ACTIVE_ONE_LINE_PER_PHYSICAL_EDGE'
RAW_GENERATION = 'FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
ARM_REPLACEMENTS = {
    'GEOMETRY_EDGE_ALL_BI_NATIVE': 'ANCHOR_RAY_EDGE_ALL_BI_NATIVE',
    'GEOMETRY_EDGE_ALL_BI_CONTAINMENT': 'ANCHOR_RAY_EDGE_ALL_BI_CONTAINMENT',
    'GEOMETRY_EDGE_SOURCE_NATIVE': 'ANCHOR_RAY_EDGE_SOURCE_NATIVE',
    'GEOMETRY_EDGE_SOURCE_CONTAINMENT': 'ANCHOR_RAY_EDGE_SOURCE_CONTAINMENT'}
REPLACEMENTS = {**ARM_REPLACEMENTS,
    'geometry_guided_edge_eval_v4_29': 'geometry_guided_edge_eval_v4_31',
    'CAL_geometry_guided_edge_noise_v4_29': 'CAL_geometry_guided_edge_noise_v4_31',
    'INDEPENDENT_SELECTED_GEOMETRY_EDGE_AUDIT_V29': 'INDEPENDENT_SELECTED_Q_ANCHORED_SIGNED_EDGE_AUDIT_V31'}


def _index(value, upper, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < upper:
        raise ValueError('Strict original ' + name + ' integer index required')
    return int(value)


def _xy(value, name):
    if len(value) != 2:
        raise ValueError('Finite original two-vector required: ' + name)
    answer = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in answer):
        raise ValueError('Finite original two-vector required: ' + name)
    return answer


def _check_association_frame(task, prediction, check, ident):
    """All selected raw queries, even when all pose arms ultimately fall back."""
    observation = task['v29_observation']
    check(observation.get('selection_profile') == PROFILE, 'new V31 graph selection profile', ident)
    check(observation.get('observation_generation') == RAW_GENERATION,
        'unchanged actual V29 raw measurements', ident)
    configuration = observation['association_configuration']
    check(configuration == observation['chain_observation']['configuration'],
        'association configuration equals original actual query configuration', ident)
    limit = float(configuration['normal_search_radius_px'])
    check(math.isfinite(limit) and limit == 43., 'existing unchanged43px normal reach', ident)
    states, hidden = task['states'], task['initial_self_hidden']
    q = {}
    for measurement in task['spec']['visible']:
        k = _index(measurement['native_corner'], 8, 'BI archive')
        # Do not read, copy or coerce excluded coordinate/source payloads.
        if hidden[k] or states[k] not in ('B', 'I'):
            continue
        if k in q:
            raise ValueError('One immutable source BI archive per native corner required')
        q[k] = _xy(measurement['xy'], 'immutable BI q')
    records = observation['chain_observation']['edge_records']
    nodes = {node['name']: node for node in observation['joint_graph']['line_candidates']}
    for ref in observation['selected_chain_refs']:
        ri = _index(ref['ray_index'], len(records), 'raw ray')
        ray = records[ri]
        k = _index(ray['native_corner'], 8, 'source native')
        mate = _index(ray['native_neighbor'], 8, 'mate native')
        if hidden[k] or hidden[mate] or states[k] not in ('B', 'I') or states[mate] == 'S':
            raise ValueError('S/BI gate before selected raw chain, line or query payload')
        if k not in q or (states[mate] in ('B', 'I') and mate not in q):
            raise ValueError('Immutable measured BI source/mate required; model mate cannot rescue missing BI')
        if states[mate] not in ('B', 'I', 'O', 'T', 'U'):
            raise ValueError('Authoritative original mate state required')
        ci = _index(ref['chain_index'], len(ray['chains']), 'raw chain')
        chain = ray['chains'][ci]
        node = nodes[ref['line_name']]
        pair = sorted([k, mate])
        check(k != mate and sorted(node['native_edge']) == pair
            and ref['native_corner'] == k and ref['native_neighbor'] == mate,
            'selected raw node/ref original native IDs', ident)
        normal = _xy(node['normal'], 'actual TLS normal');offset = float(node['offset'])
        check(math.isfinite(offset) and math.isclose(math.fsum(v*v for v in normal),
            1., rel_tol=0., abs_tol=1e-10), 'finite unit actual TLS normal', ident)
        for endpoint in (k, mate):
            if states[endpoint] not in ('B', 'I'):
                continue
            distance = abs(math.fsum((normal[0]*q[endpoint][0], normal[1]*q[endpoint][1], offset)))
            check(distance <= limit, 'all original BI q within actual infinite TLS normal43', ident)
        reference = chain['guide_reference']
        check(reference == ref['guide_reference'], 'exact indexed raw guide dictionary equals selected ref', ident)
        check(chain['guide_reference_id'] == ref['guide_reference_id']
            == reference['guide_reference_id']
            == PARENT._digest({key:value for key,value in reference.items() if key != 'guide_reference_id'}),
            'original raw guide self-ID-excluded digest and nested IDs', ident)
        check(reference['native_corner'] == k and reference['native_neighbor'] == mate,
            'original actual query native IDs', ident)
        check(node['provenance'][0] == str(ri) + ':' + str(ci),
            'first original provenance is the selected exact raw ray/chain', ident)
        if 'line_record' in node:
            check(node['line_record']['guide_reference'] == reference,
                'optional undecorated node retains the exact raw query', ident)
        anchor_kind, guide_kind = reference.get('search_anchor_kind'), reference.get('guide_kind')
        allowed = ((anchor_kind == 'OBSERVED_BI_REFERENCE_BAND' and guide_kind in
            ('PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR', 'MEASURED_BI_ENDPOINT_DIRECTION'))
            or (anchor_kind is None and guide_kind == 'FROZEN17C_ORIGINAL_MODEL_SEED'))
        check(anchor_kind != 'MODEL_PROJECTION_SEARCH_ONLY_BAND',
            'model-anchor search forbidden even when numerically equal q', ident)
        check(allowed, 'original observed-q query provenance kind allowed', ident)
        anchor = _xy(reference['actual_search_anchor_xy'], 'actual recorded query anchor')
        direction = _xy(reference['actual_search_direction_xy'], 'actual recorded query direction')
        check(max(abs(a-b) for a,b in zip(anchor,q[k])) <= 1e-6,
            'actual query anchor equals immutable source q metadata1e-6', ident)
        check(_xy(chain['generating_guide_direction_xy'], 'raw generating query direction') == direction,
            'original actual query direction alias exact', ident)
        if states[mate] in ('B', 'I'):
            delta = tuple(b-a for a,b in zip(q[k],q[mate]))
        else:
            # Only ORIGINAL initial nonS MODEL projections; never observed
            # unknown-mate coordinates, trial WD or selected-pose geometry.
            projection = task['v29_initial_geometry']['initial_projection_xy']
            start = _xy(projection[k], 'ORIGINAL model ray source')
            other = _xy(projection[mate], 'ORIGINAL model ray unknown mate')
            delta = tuple(b-a for a,b in zip(start,other))
        length, query_length = math.hypot(*delta), math.hypot(*direction)
        check(length >= 10., 'fixed native reference at least existing10px', ident)
        check(query_length > 1e-12, 'nondegenerate immutable query direction', ident)
        if length > 0. and query_length > 1e-12:
            cosine = math.fsum(a/length*b/query_length for a,b in zip(delta,direction))
            check(abs(cosine) > 1e-12, 'fixed native ray not orthogonal ambiguous', ident)
            check(cosine > 1e-12, 'actual query positive fixed native hemisphere', ident)
    for arm in ARM_REPLACEMENTS.values():
        plan = prediction['arms'][arm]['solution']['v29_cost_plan']
        check(plan.get('selection_profile') == PROFILE and plan['observation_family'] == FAMILY
            and plan['population'] == POPULATION and plan['raw_observation_generation'] == RAW_GENERATION,
            'new V31 runtime population including fallback', ident, arm)


class _VersionDriver(ast.NodeTransformer):
    def __init__(self):
        self.replacements = {key:0 for key in REPLACEMENTS}
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
                    alias.name = 'geometry_guided_edge_noise_calibration_v4_31'
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        for index, statement in enumerate(node.body):
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'ident'):
                node.body.insert(index+1, ast.parse('_check_association_frame(task,row,check,ident)').body[0])
                self.frame_hooks += 1
                break
        return node


def _build_main():
    if C.sha(PARENT.__file__) != PARENT_SHA256:
        raise ValueError('Exact frozen independent V29 numerical audit source required')
    transform = _VersionDriver()
    tree = transform.visit(ast.parse(inspect.getsource(PARENT.main)))
    if set(transform.replacements.values()) != {1} or transform.imports != 1 or transform.frame_hooks != 1:
        raise ValueError('Only declared driver names, one noise import and one frame hook may change')
    ast.fix_missing_locations(tree)
    namespace = {**PARENT.main.__globals__, '__file__':__file__, '__name__':__name__,
        '_check_association_frame':_check_association_frame}
    code = compile(tree, __file__, 'exec')
    main_code = next(value for value in code.co_consts if isinstance(value, type(PARENT.main.__code__)))
    return FunctionType(main_code, namespace, 'independent_V31_driver', PARENT.main.__defaults__), transform


def main(output=None):
    function, _ = _build_main()
    return function(output)


if __name__ == '__main__':
    main()
