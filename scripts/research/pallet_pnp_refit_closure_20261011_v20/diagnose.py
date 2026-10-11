"""Own-frozen saved-pose closure audit; no new pose fitting or truth inputs.

Compare the sealed V17 HYBRID ordinary pose with the sealed V19 point-only
consensus best on exactly the same q/K/xyz/U/H. Scientific arithmetic is ROOT
execution only. A better saved consensus key is not proof of a solver bug,
physical correspondence, accuracy improvement, or a viable new refit.
"""
import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

PRIVATE = Path('/tmp/pallet-pnp-refit-closure-private-20261011-v20')
ORDINARY = 'HYBRID_STANDARD_PNP'
ORDINARY_METHODS = (ORDINARY, 'NATIVE_STANDARD_PNP')
POINT = 'HYBRID_OBSERVED_FRAME_POINT_ONLY'
POINT_METHODS = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', POINT)
FILES = ('PROTOCOL.json', 'STARTED.json', 'ROWS.jsonl.gz', 'SUMMARY.json', 'CHECKS.json')
SCALAR_SHA = 'd9f8079d42515d6c72eac4c75676493766b95395c63e049414fb67fe48631121'
V19_CHECKER_SHA = '0456bbc94b039e32fa689e09b60b349626427634c2be80067b80f53d5629e4a4'
NAMES = {'require', 'safe', 'Audit', 'finite', 'ids', 'eligible', 'dimensions', 'dot',
         'model', 'project', 'determinant', 'physical_rotation', 'equivalent'}
CONSTANTS = {'ATOL', 'RTOL', 'RADIUS', 'TIE'}
TRUTH_FIELDS = {'pose', 'corner', 'human', 'GT', 'ground_truth', 'evaluation_reference',
                'human_states_native', 'mask_audit', 'metric_targets', 'reference_pose'}
LIMITS = dict(new_PnP_optimizer_Rodrigues_Jacobian_SVD_model_RGB_training_GT_calls=0,
    saved_pose_projection_and_scalar_distance_only=True, same_hybrid_q_not_native_backend_confound=True,
    ordinary_whole_U_rank_is_not_inlier_subset_rank=True, stored_spectra_only_no_new_rank_calculation=True,
    stored_candidate_equivalence_is_recounted_from_saved_projection_physical_R_t=True,
    foreign_ordinary_logical_start_index_not_invented=True, candidate_or_output_decision_changed=False,
    no_new_loss_seed_weights_thresholds_or_performance_retry=True,
    old_full_V16_FAIL_remains_FAIL=True, parent_accuracy_failure_not_reclassified=True,
    audit_PASS_is_not_solver_validity_or_accuracy_goal_PASS=True,
    physical_ownership_truth='UNKNOWN', global_root_or_basin_completeness_proven=False,
    camera_facing_cube_ID_order_is_not_independent_physical_axis_ownership=True,
    missing_or_nonNEW_best_is_not_an_operational_pose=True, automatic_retry=False)


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'missing or symlink input: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def same(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and all(
        left.get(k) == right.get(k) for k in ('bytes', 'sha256'))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            require(bool(line.strip()), 'blank geometry row')
            yield json.loads(line)


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def inputs(args):
    f = dict(checker=Path(__file__).resolve(),
        scalar_source=args.repo/'scripts/research/pallet_same_observation_controls_20261010_v8/validation_checks.py',
        ordinary_checker=args.repo/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/validation_checks.py',
        point_checker=args.point_checker,
        cohort=args.repo/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json',
        old_failed_validation=args.repo/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16/VALIDATION_READY_CHECKS.json')
    for prefix, folder, names in (
        ('ordinary:', args.ordinary, ('PROTOCOL.json', 'CONTROL_STARTED.json', 'CONTROL_RECEIPT.json',
            'GEOMETRY_SEAL.json', 'GEOMETRY_SEALED.jsonl.gz', 'CONTROL_LEDGERS.jsonl.gz', 'PARENT_POPULATION_CHECKS.json')),
        ('point:', args.point, ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json',
            'GEOMETRY.jsonl.gz', 'LEDGERS.jsonl.gz')),
        ('ordinary_validation:', args.ordinary_validation, ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')),
        ('point_validation:', args.point_validation, ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json'))):
        for name in names: f[prefix + name] = folder/name
    return f


def source_tree(path, bound):
    require(bound['sha256'] == SCALAR_SHA, 'exact immutable V8 scalar source')
    nodes, names, constants = [], set(), set()
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES:
            nodes.append(copy.deepcopy(node)); names.add(node.name)
        elif isinstance(node, ast.Assign):
            assigned = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if assigned and assigned <= CONSTANTS:
                nodes.append(copy.deepcopy(node)); constants.update(assigned)
    require(names == NAMES and constants == CONSTANTS, 'whole exact scalar whitelist')
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    literal = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            values = ast.literal_eval(node.value)
            literal.update(zip([n.id for n in target.elts], values))
    require(literal == dict(ATOL=1e-7, RTOL=1e-12, RADIUS=8.0, TIE=1e-8),
            'inherited arithmetic/equivalence/consensus constants unchanged')
    proof = dict(source=bound, function_names=sorted(NAMES), constants=literal,
        whole_AST_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        scalar_formula_AST_changes=0, native_cube_ID_order='V8 model() exact signs/order, registry W/H/D meters',
        score_formula='V8 candidate lines353-363: math.dist, <=8, sum squared and cap64',
        logical_start_policy='score prefix only for foreign ordinary; no invented S15 start')
    return tree, proof


def validation_gate(f, b, prefix, schema, expected, aliases, checker_alias):
    own, started, check = [read(f[prefix + name]) for name in
        ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')]
    require(check['schema'] == schema and check['complete'] is check['passed'] is True
        and check['failure_count'] == 0 and check['exception'] is None
        and check['new_production_calls'] == 0, prefix + 'actual complete independent scalar PASS')
    require(own['inputs'] == started['inputs'] == check['inputs']
        and same(check['own_protocol'], b[prefix + 'VALIDATION_PROTOCOL.json'])
        and same(started['own_protocol'], b[prefix + 'VALIDATION_PROTOCOL.json'])
        and same(own['checker'], b[checker_alias]) and same(check['checker'], b[checker_alias]),
        prefix + 'same source/own protocol/STARTED authority')
    expected_protocol = ('native_hybrid_standard_scalar_protocol_v17' if prefix == 'ordinary_validation:'
                         else 'observed_frame_hybrid_scalar_protocol_v19')
    require(own['schema'] == expected_protocol and own['frames'] == 245 and own['rows'] == 490 and own['methods'] == expected,
            prefix + 'whole paired population')
    for alias, target in aliases.items():
        require(same(check['inputs'][alias], b[target]), prefix + 'checked exact current input ' + alias)
    return own, check


def metadata(f, b):
    cohort = read(f['cohort']); ids = cohort['ids']
    require(len(ids) == len(set(ids)) == 245 and [v['id'] for v in cohort['frames']] == ids
        and cohort['counts'] == dict(clean=153, moderate=92, severe_excluded=74, original=319), 'fixed whole245 cohort')
    ordinary = read(f['ordinary:PROTOCOL.json']); seal = read(f['ordinary:GEOMETRY_SEAL.json'])
    receipt = read(f['ordinary:CONTROL_RECEIPT.json']); started = read(f['ordinary:CONTROL_STARTED.json'])
    pop = read(f['ordinary:PARENT_POPULATION_CHECKS.json'])
    require(ordinary['schema'] == 'fixed_native_hybrid_standard_protocol_v17'
        and ordinary['methods'] == list(ORDINARY_METHODS) and ordinary['frames'] == ordinary['ledger_rows'] == 245
        and ordinary['rows'] == 490, 'V17 fixed ordinary scope')
    require(seal['schema'] == 'native_hybrid_standard_geometry_seal_v17' and seal['complete'] is True
        and seal['frames'] == seal['ledger_rows'] == 245 and seal['rows'] == 490
        and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,
        'V17 GT-free seal')
    require(receipt['schema'] == 'native_hybrid_standard_execution_v17' and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is receipt['cleanup_error'] is None
        and receipt['protection_before'] == receipt['protection_after'], 'V17 clean completion')
    for name in ('primitive_and_pose_guards_restored', 'opencv_threads_restored', 'split_environment_restored', 'asset_canary_open_functions_restored'):
        require(receipt['cleanup'][name] is True, 'V17 restoration ' + name)
    for alias, name in (('protocol', 'PROTOCOL.json'), ('geometry', 'GEOMETRY_SEALED.jsonl.gz'),
        ('ledgers', 'CONTROL_LEDGERS.jsonl.gz'), ('control_receipt', 'CONTROL_RECEIPT.json'),
        ('parent_population_checks', 'PARENT_POPULATION_CHECKS.json')):
        require(same(seal[alias], b['ordinary:' + name]), 'V17 current sealed ' + alias)
    require(same(receipt['protocol'], b['ordinary:PROTOCOL.json']) and same(started['protocol'], b['ordinary:PROTOCOL.json'])
        and same(receipt['raw_rows']['geometry'], b['ordinary:GEOMETRY_SEALED.jsonl.gz'])
        and same(receipt['raw_rows']['ledgers'], b['ordinary:CONTROL_LEDGERS.jsonl.gz'])
        and pop['complete'] is pop['passed'] is True and pop['frames'] == 245 and pop['ordered_ids'] == ids
        and pop['full_V7_population_checked'] is True and same(ordinary['parent_population_checks'], b['ordinary:PARENT_POPULATION_CHECKS.json']),
        'V17 full current source population')
    vo, vc = validation_gate(f, b, 'ordinary_validation:', 'native_hybrid_standard_scalar_checks_v17', list(ORDINARY_METHODS),
        dict(geometry='ordinary:GEOMETRY_SEALED.jsonl.gz', ledgers='ordinary:CONTROL_LEDGERS.jsonl.gz',
            repair_protocol='ordinary:PROTOCOL.json', geometry_seal='ordinary:GEOMETRY_SEAL.json',
            repair_receipt='ordinary:CONTROL_RECEIPT.json', parent_population_checks='ordinary:PARENT_POPULATION_CHECKS.json',
            started='ordinary:CONTROL_STARTED.json', cohort='cohort', ordinary_scalar='scalar_source',
            **{'v16_ready:ready_checks': 'old_failed_validation'}), 'ordinary_checker')
    require(vc['counts']['frames'] == 245 and vc['counts']['rows'] == 490, 'V17 full independent counts')
    point, point_start, pr = [read(f['point:' + name]) for name in
        ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json')]
    require(point['schema'] == 'observed_frame_repair_geometry_protocol_v19' and point['methods'] == list(POINT_METHODS)
        and point['frames'] == point['ledger_rows'] == 245 and point['rows'] == 490
        and point['ordered_ids'] == ids and point['configured_before_geometry'] is True, 'V19 fixed paired source')
    require(pr['schema'] == 'observed_frame_repair_geometry_receipt_v19' and pr['complete'] is True
        and pr['error'] is None and pr['cleanup_errors'] == [] and pr['inputs_unchanged'] is pr['environment_restored'] is True
        and pr['inputs'] == point['inputs'] == point_start['inputs']
        and pr['protection_before'] == pr['protection_after'] == point['protection'] == point_start['protection'],
        'V19 clean sealed geometry')
    require(pr['limits'] == point['limits'] == point_start['limits'] and point['limits']['GT_allowed'] is False
        and point['limits']['old564_scalar_FAIL_retained'] is True, 'V19 GT-free scope and historical FAIL')
    for value, name in ((pr['protocol'], 'GEOMETRY_PROTOCOL.json'), (point_start['protocol'], 'GEOMETRY_PROTOCOL.json'),
        (pr['geometry'], 'GEOMETRY.jsonl.gz'), (pr['ledgers'], 'LEDGERS.jsonl.gz')):
        require(same(value, b['point:' + name]), 'V19 current receipt/STARTED/raw byte join')
    po, pc = validation_gate(f, b, 'point_validation:', 'observed_frame_hybrid_scalar_checks_v19', list(POINT_METHODS),
        dict(geometry='point:GEOMETRY.jsonl.gz', ledgers='point:LEDGERS.jsonl.gz', geometry_protocol='point:GEOMETRY_PROTOCOL.json',
            geometry_started='point:GEOMETRY_STARTED.json', geometry_receipt='point:GEOMETRY_RECEIPT.json',
            cohort='cohort', old_failed_validation='old_failed_validation'), 'point_checker')
    require(b['point_checker']['sha256'] == V19_CHECKER_SHA and pc['actual_counts']['frames'] == 245
        and pc['actual_counts']['geometry_rows'] == 490 and pc['actual_counts']['ledger_rows'] == 245
        and pc['old_full_V16_validation_passed'] is False and pc['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True,
        'V19 complete independent full geometry; old FAIL remains FAIL')
    for r in (receipt, pr):
        for key, value in dict(frames_started=245, frames_complete=245, geometry_rows=490, ledger_rows=245).items():
            require(r['actual_execution'][key] == value, 'actual whole245 execution ' + key)
    old = read(f['old_failed_validation'])
    require(old['complete'] is True and old['passed'] is False and old['checks'] == 198134007
        and old['failure_count'] == 564 and old['exception'] is None, 'original full FAIL preserved')
    return cohort


def compact_common(row):
    require(not TRUTH_FIELDS & set(row), 'only sealed GT-free geometry may enter this audit')
    names = ('id', 'session', 'input_points', 'original_native_N3_points', 'K', 'xyz', 'raw_hw',
        'hidden_initial', 'selected_corner_ids', 'head_checkpoint_sha256', 'calibration_binding',
        'parent_geometry_row_semantic_sha256', 'parent_observation_row_semantic_sha256', 'parent_native_N3_semantic_sha256')
    result = {k: copy.deepcopy(row[k]) for k in names}
    result['initial_pose_semantic_sha256'] = semantic(row['initial_pose'])
    return result


def compact_pose(candidate, ordinary=False):
    if candidate is None: return None
    names = ('dimension_index', 'R_cf', 'R_physical', 'centroid', 'projected')
    result = {k: copy.deepcopy(candidate[k]) for k in names}
    result['cf_extents'] = copy.deepcopy(candidate['dimensions' if ordinary else 'cf_extents'])
    return result


def compact_rank(geometry):
    return {k: copy.deepcopy(v) for k, v in geometry.items()
        if k in ('numerical_rank', 'singular_values', 'weak_condition', 'condition_number', 'normalization', 'local_only')}


def compact_ordinary(row):
    s = row['solver']; result = compact_common(row)
    result.update(method=row['method'], output_status=row['output_status'], new_pose_estimated=row['new_pose_estimated'],
        fallback_used=row['fallback_used'], solver_state=s['state'], available=s['available'], used=s['used'], hidden=s['hidden'],
        fit_input_ids=s['fit_input_ids'], final_inliers=s['final_inliers'], best=compact_pose(s.get('best_candidate'), True),
        stored_rank=compact_rank(s['geometry'].get('jacobian', {})), prior_used=s['prior_used'],
        known_dimension_constraint_used=s['known_dimension_constraint_used'])
    best = s.get('best_candidate')
    if best is not None:
        result['best_metadata'] = {k: copy.deepcopy(best[k]) for k in ('generator', 'generator_ids', 'actual_fit_input_ids', 'solution_index',
            'residuals_used_px', 'inlier_ids', 'inlier_count', 'sse_px2', 'truncated_sse_px2')}
    return result


def score_pose(pose, q, K, xyz, U, h, audit, counts, label):
    if pose is None: return None
    dims = h['dimensions'](xyz); d = pose['dimension_index']
    require(type(d) is int and 0 <= d < len(dims) and pose['cf_extents'] == dims[d], 'same registered model dimensions')
    counts['independent_project_entries'] += 1
    try: projected = h['project'](pose['cf_extents'], pose['R_cf'], pose['centroid'], K)
    except BaseException:
        counts['independent_project_exceptions'] += 1
        raise
    counts['independent_project_returns'] += 1; counts['independently_projected_model_corners'] += 8
    audit.near(pose['projected'], projected, label + '.same_Rt_projection')
    audit.near(pose['R_physical'], h['physical_rotation'](pose['R_cf'], d), label + '.physical_phase')
    audit.ok(abs(h['determinant'](pose['R_cf']) - 1.0) <= 1e-6, label + '.proper_R')
    for i, axis in enumerate(pose['R_cf']):
        for j, other in enumerate(pose['R_cf']): audit.ok(abs(h['dot'](axis, other) - int(i == j)) <= 1e-6, label + '.orthogonality')
    residual = [math.dist(projected[k], q[k]) for k in U]
    counts['independent_point_distance_calls'] += len(U)
    inliers = [k for k, value in zip(U, residual) if value <= h['RADIUS']]
    return dict(independent_point_residual_norm_px=residual, final_inliers=inliers, score_count=len(inliers),
        factor_sse_px2=math.fsum(v*v for v in residual), factor_truncated_sse_px2=math.fsum(min(v*v, 64.0) for v in residual),
        primary_decision_key=[-len(inliers), math.fsum(min(v*v, 64.0) for v in residual)],
        dimension_index=d, independently_enforced_all8_positive_depth=True)


def matches(candidate, target, xyz, h, counts):
    counts['stored_candidate_equivalence_checks'] += 1
    return h['equivalent'](candidate, target, xyz)


def match_metadata(candidate, stage):
    result = dict(stage=stage)
    for key in ('candidate_index', 'proposal_index', 'logical_start_index', 'proposal_kind', 'dimension_index',
        'point_generator', 'point_generator_ids', 'point_solution_index', 'fit_input_ids', 'final_inliers',
        'accepted', 'rejected', 'score_count', 'truncated_sse_px2', 'actual_optimizer_called', 'optimizer_success'):
        if key in candidate: result[key] = copy.deepcopy(candidate[key])
    if 'whole_pool_score' in candidate:
        result['saved_full_U_score'] = {k: copy.deepcopy(candidate['whole_pool_score'][k]) for k in
            ('point_ids', 'final_inliers', 'inlier_count', 'factor_truncated_sse_px2', 'factor_sse_px2')}
    return result


def compare(row, ordinary, h, audit, counts):
    common = compact_common(row); label = row['id']; s = row['solver']
    audit.eq(common, {k: ordinary[k] for k in common}, label + '.same_hybrid_q_K_dimensions_H_source')
    require(common == {k: ordinary[k] for k in common}, 'comparison inputs are not exactly the same hybrid observations')
    q, K, xyz, H = common['input_points'], common['K'], common['xyz'], common['hidden_initial']
    require(h['finite'](K, (3, 3)) and K[0][0] > 0 and K[1][1] > 0 and K[2] == [0., 0., 1.]
        and K[0][1] == K[1][0] == 0. and h['finite'](xyz, (3,)) and min(xyz) > 0,
        'same no-distortion pinhole K and registry W/H/D meters')
    require(h['ids'](H) and len(q) == 9 and all(isinstance(v, list) and len(v) == 2 for v in q)
        and q[8] == common['original_native_N3_points'][8], 'H/unchanged center8 contract')
    require(common['head_checkpoint_sha256'] == '882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d',
            'same immutable corrected IMAGE_ROLE head provenance')
    U = [k for k in h['eligible'](q, common['raw_hw']) if k not in H]
    require(s['used'] == ordinary['used'] == U and s['hidden'] == ordinary['hidden'] == H
        and not set(U) & set(H) and all(k < 8 for k in U), 'exact same actual active U/H; center never scored')
    require(s['line_edges'] == s['fit_line_edges'] == s['final_line_inliers'] == s['consumed_edges'] == []
        and s['actual_source_lines'] == [], 'strict point-only source')
    for flag in ('prior_used', 'known_dimension_constraint_used', 'center_used_for_fit_or_score', 'initial_pose_start_used',
        'reprojected_points_reused_as_observations'): require(s[flag] is False, 'point path no prior/start/H-refit ' + flag)
    require(ordinary['prior_used'] is ordinary['known_dimension_constraint_used'] is False, 'ordinary no pose/dimension prior')
    require(ordinary['new_pose_estimated'] is ordinary['available']
        and ordinary['new_pose_estimated'] is (ordinary['output_status'] == 'NEW_POSE')
        and row['new_pose_estimated'] is s['available'] and row['new_pose_estimated'] is (row['output_status'] == 'NEW_POSE'),
        'stored best and actual approved output status stay distinct')
    require(not set(ordinary['fit_input_ids']) & set(H) and (not ordinary['available'] or ordinary['fit_input_ids'] == U),
            'ordinary approved whole-U fit has no H/center')
    best = s.get('best_candidate'); opose = ordinary['best']; ppose = compact_pose(best)
    oscore = score_pose(opose, q, K, xyz, U, h, audit, counts, label + '.ordinary')
    pscore = score_pose(ppose, q, K, xyz, U, h, audit, counts, label + '.point')
    if oscore is not None:
        original = ordinary['best_metadata']; audit.near(original['residuals_used_px'], oscore['independent_point_residual_norm_px'], label + '.ordinary_residual')
        audit.eq(original['inlier_ids'], oscore['final_inliers'], label + '.ordinary_independent8px')
        audit.near(original['sse_px2'], oscore['factor_sse_px2'], label + '.ordinary_full_U_SSE')
        audit.near(original['truncated_sse_px2'], oscore['factor_truncated_sse_px2'], label + '.ordinary_full_U_cappedSSE')
    if pscore is not None:
        w = best['whole_pool_score']; require(best['whole_pool_point_ids'] == w['point_ids'] == U
            and best['whole_pool_line_edges'] == w['line_edges'] == [], 'point best same full-pool score')
        audit.near(w['point_residual_norm_px'], pscore['independent_point_residual_norm_px'], label + '.point_residual')
        audit.eq(best['final_inliers'], pscore['final_inliers'], label + '.point_independent8px')
        audit.eq(best['score_count'], pscore['score_count'], label + '.point_count')
        audit.near(best['truncated_sse_px2'], pscore['factor_truncated_sse_px2'], label + '.point_cappedSSE')
        audit.near(w['factor_sse_px2'], pscore['factor_sse_px2'], label + '.point_full_U_SSE')
    equivalent = None; relation = 'NO_ORDINARY_STORED_BEST' if opose is None else 'NO_POINT_ACCEPTED_BEST' if ppose is None else None
    gap = None
    if relation is None:
        gap = oscore['factor_truncated_sse_px2'] - pscore['factor_truncated_sse_px2']
        equivalent = h['equivalent'](opose, ppose, xyz); counts['stored_best_equivalence_checks'] += 1
        relation = ('ORDINARY_HIGHER_INLIER_COUNT' if oscore['score_count'] > pscore['score_count'] else
            'POINT_HIGHER_INLIER_COUNT' if oscore['score_count'] < pscore['score_count'] else
            'INHERITED_NUMERICAL_SCORE_TIE' if abs(gap) <= h['TIE'] else
            'ORDINARY_LOWER_SAME_COUNT_TRUNCATED_SSE' if gap < 0 else 'POINT_LOWER_SAME_COUNT_TRUNCATED_SSE')
    found = []
    if opose is not None:
        for stage, candidates in (('RAW_FOUR_ID_PROPOSAL', s['proposal_solutions']), ('ACTUAL_INLIER_REFIT', s['all_candidate_solutions'])):
            for c in candidates:
                counts['stored_candidate_records_scanned'] += 1
                if matches(c, opose, xyz, h, counts): found.append(match_metadata(c, stage))
    rank = ordinary['stored_rank']; compatibility = ('NO_STORED_ORDINARY_BEST' if oscore is None else
        'UNKNOWN_NEW_INLIER_SUBSET_RANK_AND_REFIT' if oscore['final_inliers'] != U else
        'UNKNOWN_NO_STORED_WHOLE_U_RANK' if not rank else
        'STORED_WHOLE_U_RANK_FAIL' if rank['numerical_rank'] != 6 else
        'STORED_WHOLE_U_WEAK' if rank['weak_condition'] else 'STORED_WHOLE_U_GEOMETRY_ONLY_COMPATIBLE')
    point_rank = {} if best is None else {k: compact_rank(best['point_line_geometry'][k]) for k in
        ('observed_normal_joint', 'modeled_normal_joint', 'point_only')}
    if best is not None:
        require(best['accepted'] is True and best['rejected'] is None and len(best['fit_input_ids']) >= 4
            and set(best['fit_input_ids']) <= set(U) and not set(best['fit_input_ids']) & set(H), 'stored point accepted fit identity')
    result = dict(id=row['id'], session=row['session'], methods=[ORDINARY, POINT],
        same_input_semantic_sha256=semantic(common), input_points=q, active_U=U, initial_H=H,
        selected_boundary_corner_ids=common['selected_corner_ids'],
        ordinary_output=dict(status=ordinary['output_status'], solver_state=ordinary['solver_state'], approved_NEW=ordinary['new_pose_estimated'],
            fallback=ordinary['fallback_used'], stored_best_present=opose is not None, stored_best_is_operational_pose=ordinary['new_pose_estimated'],
            best_pose=opose, best_metadata=ordinary.get('best_metadata'), stored_whole_U_rank=rank),
        point_output=dict(status=row['output_status'], solver_state=s['state'], approved_NEW=row['new_pose_estimated'], fallback=row['fallback_used'],
            stored_best_present=best is not None, stored_best_is_operational_pose=row['new_pose_estimated'], best_pose=ppose,
            fit_input_ids=[] if best is None else best['fit_input_ids'], final_inliers=[] if best is None else best['final_inliers'],
            stored_actual_fit_rank=point_rank, stable_guard_state=None if best is None else best['stable_incidence_guard']['guard_state'],
            stable_guard_accepted=None if best is None else best['stable_incidence_guard']['accepted']),
        independent_ordinary_score=oscore, independent_point_score=pscore, score_relation=relation,
        ordinary_minus_point_truncated_SSE_px2=gap, best_numerical_equivalence=equivalent,
        foreign_ordinary_logical_start_index=None, ordinary_matching_point_candidates=found,
        accepted_equivalent_refit_present=any(v['stage'] == 'ACTUAL_INLIER_REFIT' and v.get('accepted') is True for v in found),
        ordinary_geometry_compatibility_scope=compatibility,
        ordinary_S15_optimizer_success_at_exact_pose='UNKNOWN_NOT_RUN', ordinary_pose_declared_viable_new_S15_candidate=False,
        score_comparison_scope='STORED_BEST_ONLY; approved_NEW/fallback kept separately',
        physical_accuracy_ownership_or_definite_solver_bug_certified=False)
    counts['frames_compared'] += 1; counts['relation:' + relation] += 1
    counts['output:' + ORDINARY + ':' + ordinary['output_status']] += 1; counts['output:' + POINT + ':' + row['output_status']] += 1
    counts['ordinary_geometry_scope:' + compatibility] += 1
    counts['ordinary_equivalent_refit_accepted_frames'] += int(result['accepted_equivalent_refit_present'])
    return result


def guard(args):
    output = args.output
    require(output.is_relative_to(PRIVATE) and output != PRIVATE, 'output must be a new V20 private subdirectory')
    require(not any(p.is_symlink() for p in (output, *output.parents)), 'symlink output forbidden')
    for path in (args.repo, args.ordinary, args.ordinary_validation, args.point, args.point_validation):
        require(path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)), 'existing nonsymlink input folder')
        require(not output.is_relative_to(path) and not path.is_relative_to(output), 'input/output overlap')
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES if args.stage == 'freeze' else FILES[1:]: require(not (output/name).exists(), 'preserve ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--ordinary', type=Path, required=True); parser.add_argument('--ordinary-validation', type=Path)
    parser.add_argument('--point', type=Path, required=True); parser.add_argument('--point-validation', type=Path, required=True)
    parser.add_argument('--point-checker', type=Path, default=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19/validation.py'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.absolute())
    args.ordinary_validation = args.ordinary if args.ordinary_validation is None else args.ordinary_validation
    guard(args); f = inputs(args); b = {k: binding(p) for k, p in f.items()}
    cohort = metadata(f, b); tree, proof = source_tree(f['scalar_source'], b['scalar_source'])
    out = args.output
    if args.stage == 'freeze':
        write(out/FILES[0], dict(schema='fixed_saved_ordinary_point_closure_protocol_v20', inputs=b, checker=b['checker'],
            frozen_before_projection_score_or_geometry_decode=True, scalar_source_proof=proof,
            frames=245, input_geometry_rows=980, selected_comparison_rows=490,
            methods=[ORDINARY, POINT], ordered_ids=cohort['ids'], limits=LIMITS)); return
    own = read(out/FILES[0]); ob = binding(out/FILES[0])
    require(own['schema'] == 'fixed_saved_ordinary_point_closure_protocol_v20' and own['inputs'] == b
        and own['scalar_source_proof'] == proof and own['limits'] == LIMITS and own['ordered_ids'] == cohort['ids'], 'exact own-frozen source/geometry closure')
    write(out/FILES[1], dict(schema='fixed_saved_ordinary_point_closure_started_v20', inputs=b, own_protocol=ob,
        checker=b['checker'], new_production_calls=0, limits=LIMITS))
    h = dict(math=math, json=json); exec(compile(tree, str(f['scalar_source']) + ':unchanged_saved_scalar_only', 'exec'), h)
    audit, counts, error, emitted = h['Audit'](), Counter(), None, 0
    pending = out/'ROWS.pending.jsonl.gz'; stream = None
    try:
        ordinary = {}; sessions = {v['id']: v['session'] for v in cohort['frames']}
        for i, row in enumerate(rows(f['ordinary:GEOMETRY_SEALED.jsonl.gz'])):
            fid = cohort['ids'][i//2] if i < 490 else None
            require(row['id'] == fid and row['method'] == ORDINARY_METHODS[i%2] and row['session'] == sessions[fid], 'ordered full V17 paired490')
            require(not TRUTH_FIELDS & set(row), 'ordinary geometry is GT-free')
            counts['ordinary_input_geometry_rows'] += 1
            if row['method'] == ORDINARY:
                require(fid not in ordinary, 'duplicate ordinary frame'); ordinary[fid] = compact_ordinary(row)
        require(counts['ordinary_input_geometry_rows'] == 490 and list(ordinary) == cohort['ids'], 'whole245 ordinary compact index')
        stream = gzip.open(pending, 'xt')
        for i, row in enumerate(rows(f['point:GEOMETRY.jsonl.gz'])):
            fid = cohort['ids'][i//2] if i < 490 else None
            require(row['id'] == fid and row['method'] == POINT_METHODS[i%2] and row['session'] == sessions[fid], 'ordered full V19 paired490')
            require(not TRUTH_FIELDS & set(row), 'point geometry is GT-free')
            counts['point_input_geometry_rows'] += 1
            if row['method'] != POINT: continue
            value = compare(row, ordinary.pop(fid), h, audit, counts)
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'); emitted += 1
        require(not ordinary and counts['point_input_geometry_rows'] == 490 and emitted == counts['frames_compared'] == 245,
                'all245 compact sameq joins; no giant point/line index retained')
        audit.eq(counts['independent_project_entries'], counts['independent_project_returns'] + counts['independent_project_exceptions'],
                 'independent_saved_projection_dispatch_closed')
        stream.close(); stream = None
        with pending.open('rb') as raw: os.fsync(raw.fileno())
        os.link(pending, out/FILES[2]); pending.unlink()
        audit.eq({k: binding(p) for k, p in f.items()}, b, 'all_protected_inputs_unchanged')
        audit.eq(binding(out/FILES[0]), ob, 'own_protocol_unchanged')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if stream is not None:
            try: stream.close()
            except BaseException as exc:
                if error is None: error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    complete = error is None and emitted == 245
    summary = dict(schema='fixed_saved_ordinary_point_closure_summary_v20', complete=complete,
        counts=dict(counts), rows=binding(out/FILES[2]) if (out/FILES[2]).exists() else None,
        interrupted_prefix=binding(pending) if pending.exists() else None,
        same_q_comparison_only=True, solver_or_output_changes=0, no_GTexperiment_or_accuracy_goal_verdict=True,
        limits=LIMITS)
    write(out/FILES[3], summary)
    result = dict(schema='fixed_saved_ordinary_point_closure_checks_v20', complete=complete,
        passed=complete and audit.failure_count == 0, exception=error, checks=audit.checks,
        failure_count=audit.failure_count, failures=audit.failures, max_scalar_absolute_difference=audit.max_difference,
        inputs=b, own_protocol=ob, checker=b['checker'], scalar_source_proof=proof,
        rows=summary['rows'], summary=binding(out/FILES[3]), counts=dict(counts), limits=LIMITS,
        new_production_calls=0, original_V16_full_scalar_passed=False,
        audit_PASS_is_only_saved_geometry_and_comparable_key_closure=True)
    write(out/FILES[4], result); print(json.dumps(dict(complete=complete, passed=result['passed'], frames=emitted,
        failure_count=audit.failure_count, exception=error, counts=dict(counts)), ensure_ascii=False), flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
