"""Saved V19 correspondence/layout, mask and pose-outcome crossgroups.

Stdlib only. ROOT freezes and runs this after the actual geometry/statistics
checks pass. This copies saved proxy quality/error witnesses; it never derives
new pixel distances, poses, Jacobians, spectra, bootstrap draws or GT labels.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

PRIVATE = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
METHODS = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
POOLS = ('eligible', 'solver_used', 'accepted_fit', 'accepted_final_inliers')
GOOD, BAD, UNKNOWN = 'CORRECT_WITHIN8PX', 'INCORRECT_OVER8PX', 'UNKNOWN_REFERENCE_OR_INPUT'
VISIBILITY_UNKNOWN = ('UNKNOWN', 'UNANNOTATED')
CUBOID_VERTEX_SIGNS = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                      (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
OUTCOMES = ('BOTH_BETTER', 'BOTH_WORSE', 'MIXED_OR_UNCHANGED', 'POSE_UNAVAILABLE')
FILES = ('CORRESPONDENCE_PROTOCOL.json', 'CORRESPONDENCE_STARTED.json',
         'CORRESPONDENCE_CHECKS.json', 'CORRESPONDENCE_ROWS.jsonl.gz', 'CORRESPONDENCE_SUMMARY.json')
LIMITS = dict(new_GT_model_training_RGB_PnP_optimizer_projection_J_SVD_metric_CI_calls=0,
    recorded_GEOMETRIC_PROXY_quality_is_not_physical_correctness=True,
    unknown_visibility_excluded_from_semantically_known_classification=True,
    known_visibility_does_not_certify_direct_visibility_or_observation_validity=True,
    unknown_reference_not_converted_to_NONE=True, new_pose_is_not_accuracy_success=True,
    physical_ownership_and_global_uniqueness_certified=False,
    old_full_V16_564_FAIL_remains_FAIL=True, independent_pixel_quality_recomputation=False,
    stored_rank_spectrum_copy_only=True, candidate_solver_ids_not_fallback_fit_observations=True)


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink input: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def same(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and all(left.get(k) == right.get(k) for k in ('bytes', 'sha256'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)


def input_paths(args):
    g, v, e, s = map(Path, (args.geometry_input, args.validation_input, args.evaluation_input, args.statistics_input))
    paths = {'geometry:' + n: g / n for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json', 'GEOMETRY.jsonl.gz', 'LEDGERS.jsonl.gz')}
    paths.update({'validation:' + n: v / n for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')})
    paths.update({'evaluation:' + n: e / n for n in ('EVALUATION_PROTOCOL.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json',
        'SCORING_RECEIPT.json', 'STATISTICS_STARTED.json', 'METRICS.json', 'POSTHOC_ROWS.jsonl.gz', 'PREDICTIONS.jsonl.gz')})
    paths.update({'statistics:' + n: s / n for n in ('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')})
    m = read(paths['evaluation:METRICS.json'])
    paths.update(cohort=Path(m['inputs']['cohort']['path']), evaluator=Path(m['inputs']['adapter']['path']),
        geometry_checker=Path(m['inputs']['checker']['path']), statistics_checker=Path(read(paths['statistics:STATISTICS_CHECKS.json'])['checker']['path']),
        posthoc_source=Path(m['inputs']['unchanged_statistics']['path']), checker=Path(__file__))
    repo = Path(args.repo)
    paths['edge_source'] = repo / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py'
    paths['vertex_source'] = repo / 'scripts/research/pallet_observation_refiner_20261009_v1/solver.py'
    status = repo / '_docs/experiments/pallet_observation_refiner_status_20261011'
    for n in ('REMAINING_THIRTEEN_QUERIES.json', 'REMAINING_THIRTEEN_SOURCE_TAGS.json', 'REMAINING_PROBLEM_KO.md'):
        paths['prior_provenance:' + n] = status / n
    return paths


def edge_ids(path):
    nodes = [n.value for n in ast.parse(Path(path).read_bytes()).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'EDGES' for t in n.targets)]
    require(len(nodes) == 1, 'one literal immutable EDGES declaration')
    edges = ast.literal_eval(nodes[0])
    require(edges == [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)], 'original native incident-edge ordering')
    return [list(v) for v in edges]


def vertex_signs(path):
    functions = [n for n in ast.parse(Path(path).read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'cuboid']
    require(len(functions) == 1, 'one immutable cuboid definition')
    returns = [n for n in functions[0].body if isinstance(n, ast.Return)]
    require(len(returns) == 1 and isinstance(returns[0].value, ast.Call), 'literal cuboid vertex array')
    def sign(value, axis):
        if isinstance(value, ast.Name) and value.id == axis: return 1
        require(isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Name) and value.operand.id == axis, 'original cuboid a/b/c ordering')
        return -1
    values = tuple(tuple(sign(v, axis) for v, axis in zip(row.elts, ('a', 'b', 'c')))
                   for row in returns[0].value.args[0].elts)
    require(values == CUBOID_VERTEX_SIGNS, 'original vertex signs/order')
    return values


def gate(f, b):
    core, started, receipt = [read(f['geometry:' + n]) for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json')]
    own, start, check = [read(f['validation:' + n]) for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')]
    require(core['schema'] == 'observed_frame_repair_geometry_protocol_v19' and core['methods'] == list(METHODS)
        and core['frames'] == core['ledger_rows'] == 245 and core['rows'] == 490 and core['limits']['GT_allowed'] is False, 'full GT-free paired245 protocol')
    require(receipt['schema'] == 'observed_frame_repair_geometry_receipt_v19' and receipt['complete'] is True
        and receipt['error'] is None and receipt['cleanup_errors'] == [] and receipt['inputs_unchanged'] is True
        and receipt['environment_restored'] is True and core['inputs'] == started['inputs'] == receipt['inputs'], 'completed cleaned geometry')
    require(receipt['actual_execution']['frames_complete'] == receipt['actual_execution']['ledger_rows'] == 245
        and receipt['actual_execution']['geometry_rows'] == 490, 'full geometry coverage')
    for item in (receipt['protocol'], started['protocol']): require(same(item, b['geometry:GEOMETRY_PROTOCOL.json']), 'geometry protocol byte join')
    for key, name in (('geometry', 'GEOMETRY.jsonl.gz'), ('ledgers', 'LEDGERS.jsonl.gz')):
        require(same(receipt[key], b['geometry:' + name]), 'completed raw ' + key)
    require(own['schema'] == 'observed_frame_hybrid_scalar_protocol_v19' and check['schema'] == 'observed_frame_hybrid_scalar_checks_v19'
        and check['complete'] is True and check['passed'] is True and check['failure_count'] == 0 and check['exception'] is None,
        'actual independent full V19 scalar PASS')
    require(check['inputs'] == own['inputs'] == start['inputs'] and check['new_production_calls'] == start['new_production_calls'] == 0
        and check['actual_counts']['geometry_rows'] == 490 and check['actual_counts']['ledger_rows'] == check['actual_counts']['frames'] == 245,
        'independent full245 envelope')
    require(check['old_full_V16_validation_passed'] is False and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True, 'old full564 FAIL retained')
    for item in (check['own_protocol'], start['own_protocol']): require(same(item, b['validation:VALIDATION_PROTOCOL.json']), 'actual geometry checker protocol')
    for item in (check['checker'], own['checker']): require(same(item, b['geometry_checker']), 'current full geometry checker')
    for alias, name in (('geometry', 'GEOMETRY.jsonl.gz'), ('ledgers', 'LEDGERS.jsonl.gz'), ('geometry_protocol', 'GEOMETRY_PROTOCOL.json'),
                        ('geometry_started', 'GEOMETRY_STARTED.json'), ('geometry_receipt', 'GEOMETRY_RECEIPT.json')):
        require(same(own['inputs'][alias], b['geometry:' + name]), 'geometry checker input ' + alias)
    m, e, stats_start, score, authority, score_start = [read(f['evaluation:' + n]) for n in (
        'METRICS.json', 'EVALUATION_PROTOCOL.json', 'STATISTICS_STARTED.json', 'SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json')]
    require(m['schema'] == 'observed_frame_raw_row_metrics_v19' and m['complete'] is True and m['V16_full_scalar_validation_passed'] is False
        and m['inputs'] == stats_start['inputs'] and m['all245_preserved_no_performance_filter'] is True, 'complete unfiltered saved statistics')
    require(e['schema'] == 'observed_frame_postseal_evaluation_protocol_v19' and e['new_methods'] == score['methods'] == list(METHODS)
        and e['inputs'] == authority['inputs'] and authority['schema'] == 'isolated_observed_frame_proxy_score_authority_v19', 'same own scoring authority')
    require(score['complete'] is True and score['frames'] == 245 and score['scored_rows'] == 490
        and score_start['configured_scored_rows'] == 490 and score_start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,
        'actual post-PASS scored490')
    for actual, saved in (('evaluation:EVALUATION_PROTOCOL.json', authority['own_evaluation_protocol']),
                         ('evaluation:PREDICTIONS.jsonl.gz', score['predictions']), ('evaluation:POSTHOC_ROWS.jsonl.gz', m['posthoc_rows']),
                         ('geometry:GEOMETRY_PROTOCOL.json', score['protocol']), ('geometry:GEOMETRY_RECEIPT.json', score['geometry_seal']),
                         ('validation:VALIDATION_PROTOCOL.json', score_start['independent_validation'])):
        require(same(b[actual], saved), 'saved score/posthoc binding ' + actual)
    for key in ('cohort', 'adapter', 'checker', 'unchanged_statistics'):
        alias = dict(adapter='evaluator', checker='geometry_checker', unchanged_statistics='posthoc_source').get(key, key)
        require(same(b[alias], m['inputs'][key]), 'actual saved producer source ' + key)
    so, ss, sc = [read(f['statistics:' + n]) for n in ('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')]
    require(so['schema'] == 'independent_observed_frame_statistics_protocol_v19' and sc['schema'] == 'independent_observed_frame_statistics_checks_v19'
        and sc['complete'] is True and sc['passed'] is True and sc['error'] is None and sc['new_production_calls'] == 0
        and sc['old_full_V16_validation_passed'] is False and so['inputs'] == ss['inputs'] == sc['inputs'], 'actual independent saved statistics PASS')
    expected = dict(saved_error_rows_streamed=1960, saved_error_rows_selected=1715, frames=245, methods=7,
        moment_distributions_checked=675, moment_scalars_checked=5400, paired_groups_checked=54, metric_CI_slots_checked=162)
    require(all(sc['actual_counts'][k] == v for k, v in expected.items()), 'whole independent moment/CI coverage')
    for item in (sc['own_protocol'], ss['own_protocol']): require(same(item, b['statistics:STATISTICS_CHECK_PROTOCOL.json']), 'statistics checker protocol')
    for item in (sc['checker'], so['checker']): require(same(item, b['statistics_checker']), 'statistics checker current source')
    for alias, name in (('METRICS.json', 'METRICS.json'), ('new_scores', 'PREDICTIONS.jsonl.gz'),
                        ('STATISTICS_STARTED.json', 'STATISTICS_STARTED.json'), ('EVALUATION_PROTOCOL.json', 'EVALUATION_PROTOCOL.json'),
                        ('SCORING_RECEIPT.json', 'SCORING_RECEIPT.json'), ('SCORING_AUTHORITY.json', 'SCORING_AUTHORITY.json'),
                        ('SCORING_STARTED.json', 'SCORING_STARTED.json')):
        require(same(so['inputs'][alias], b['evaluation:' + name]), 'independently checked score/statistic input ' + alias)
    for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json'):
        require(same(so['inputs'][n], b['validation:' + n]), 'statistics geometry authority ' + n)
    c = read(f['cohort']); require(c['ids'] == core['ordered_ids'] and len(c['ids']) == len(set(c['ids'])) == 245, 'unchanged complete ordered245')
    require(Counter(r['label'] for r in c['frames']) == Counter(clean=153, moderate=92), 'Clean153/Moderate92 only')
    for n, schema in (('REMAINING_THIRTEEN_QUERIES.json', 'remaining_thirteen_saved_query_case_index_v18'),
                      ('REMAINING_THIRTEEN_SOURCE_TAGS.json', 'existing_source_tag_provenance_for_thirteen_queries_v18')):
        p = read(f['prior_provenance:' + n]); require(p['schema'] == schema and p['complete'] is True and p['query_count'] == 13, 'existing thirteen-case provenance')
    vertex_signs(f['vertex_source'])
    return c, edge_ids(f['edge_source'])


def compact_geometry(row):
    # Each large raw row is decoded once, then all candidate/optimizer/J banks
    # are released before the next row. No numerical value is recomputed.
    names = ('id', 'session', 'method', 'output_status', 'new_pose_estimated', 'fallback_used', 'pose_available',
        'hidden_initial', 'reprojections_reused_as_observations', 'reprojected_ids', 'hidden_reprojected',
        'native_points', 'original_native_N3_points', 'output_coordinate_sources', 'input_points',
        'selected_corner_ids', 'xyz', 'head_checkpoint_sha256', 'calibration_binding')
    result = {k: row[k] for k in names}; source = row['solver']
    names = ('eligible', 'used', 'hidden', 'fit_input_ids', 'final_inliers', 'fit_line_edges', 'final_line_inliers',
        'line_edges', 'consumed_edges', 'state', 'projected', 'reprojected_points_reused_as_observations')
    solver = {k: source.get(k) if k == 'projected' else source[k] for k in names}
    solver['operation_counts'] = dict(hidden_reprojection_assignments=source['operation_counts'].get('hidden_reprojection_assignments', 0))
    solver['point_line_geometry'] = geometry_copy(source.get('point_line_geometry'))
    stable = source.get('stable_incidence_guard')
    solver['stable_incidence_guard'] = None if stable is None else dict(guard_state=stable.get('guard_state'),
        accepted=stable.get('accepted'), stable_incidence_geometry=geometry_copy(stable.get('stable_incidence_geometry')))
    solver['best_candidate'] = dict(point_line_geometry=geometry_copy((source.get('best_candidate') or {}).get('point_line_geometry')))
    result['solver'] = solver; return result


def compact_score(row):
    return {k: row[k] for k in ('id', 'session', 'method', 'output_status', 'new_pose_estimated', 'fallback_used',
        'pose', 'mask_audit', 'hidden_initial')}


def indexed(path, methods, ids, compact=None):
    result = {}; count = 0
    for row in rows(path):
        key = (row['method'], row['id']); require(row['method'] in methods and row['id'] in ids and key not in result, 'unique declared method/ID')
        result[key] = compact(row) if compact else row; count += 1
    require(count == 490 and set(result) == {(m, i) for m in methods for i in ids}, 'all two-method490 rows retained')
    return result


def ids(value, label):
    require(isinstance(value, list) and len(value) == len(set(value)) and all(type(k) is int and 0 <= k < 8 for k in value), label + ' native corner IDs')
    return list(value)


def geometry_copy(value):
    if value is None: return None
    return {k: copy.deepcopy(v) for k, v in value.items() if k not in ('observed_normal_J', 'modeled_normal_J', 'actual_scalar_residuals_px', 'edge_blocks')}


def pool_copy(pool, corners, expected, label):
    selected = ids(pool['ids'], label); require(selected == expected, label + ' actual geometry join')
    saved = Counter(corners[k]['input_quality'] for k in selected)
    require(dict(saved) == pool['quality_counts'], label + ' saved quality counts')
    require(pool['human_direct_visible_ids'] == [k for k in selected if corners[k]['human_state'] == 'DIRECT_VISIBLE'], label + ' visible IDs')
    known = [k for k in selected if corners[k]['human_state'] not in VISIBILITY_UNKNOWN]
    good = [k for k in selected if corners[k]['input_quality'] == GOOD]
    bad = [k for k in selected if corners[k]['input_quality'] == BAD]
    return dict(ids=selected, quality_counts=copy.deepcopy(pool['quality_counts']),
        proxy_correct_ids=good, proxy_incorrect_ids=bad, proxy_unknown_ids=[k for k in selected if corners[k]['input_quality'] == UNKNOWN],
        known_visibility_ids=known, unknown_visibility_ids=[k for k in selected if k not in known],
        known_visibility_proxy_correct_ids=[k for k in good if k in known], known_visibility_proxy_incorrect_ids=[k for k in bad if k in known],
        directly_visible_proxy_correct_ids=[k for k in good if corners[k]['human_state'] == 'DIRECT_VISIBLE'],
        unknown_visibility_proxy_correct_ids=[k for k in good if k not in known],
        human_direct_visible_ids=copy.deepcopy(pool['human_direct_visible_ids']),
        human_self_occluded_ids=[k for k in selected if corners[k]['human_state'] == 'SELF_OCCLUDED'])


def record(g, p, s, ledger, label, edges, provenance):
    require(all(g[k] == p[k] == s[k] for k in ('id', 'session', 'method', 'output_status', 'new_pose_estimated', 'fallback_used')), 'geometry/posthoc/score identity/status join')
    require(g['id'] == ledger['id'] and g['session'] == ledger['session'], 'same-frame paired ledger')
    new, fallback = g['new_pose_estimated'], g['fallback_used']
    require(type(new) is type(fallback) is bool and not (new and fallback), 'typed NEW/fallback')
    status = 'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'
    require(g['output_status'] == status and g['pose_available'] == s['pose']['available'], 'actual operational status')
    require(p['pose_outcome_vs_N3'] in OUTCOMES, 'saved joint T/R outcome category')
    solver = g['solver']; H = ids(g['hidden_initial'], 'H'); corners = p['corners']
    require([c['id'] for c in corners] == list(range(8)) and all(c['input_quality'] in (GOOD, BAD, UNKNOWN) for c in corners), 'eight saved corner quality states')
    require(p['hidden_initial'] == s['hidden_initial'] == H == ledger['initial_H'] and solver['hidden'] == H, 'same initial H')
    require(p['mask_definition_provenance']['score_known_excludes_UNKNOWN_UNANNOTATED'] is True, 'semantic known mask producer required')
    states = [c['human_state'] for c in corners]; known = {k for k in range(8) if states[k] not in VISIBILITY_UNKNOWN}
    humanH = {k for k in range(8) if states[k] == 'SELF_OCCLUDED'}
    audit = p['mask_audit']; require(audit == s['mask_audit'] and audit['human_states_native'] == states and audit['known_ids'] == sorted(known), 'UNKNOWN/UNANNOTATED excluded from known visibility')
    wrong = bool((set(H) ^ humanH) & known)
    require(audit['mask_applied'] is True and audit['mask_wrong_on_known'] == wrong, 'same attempted H exclusion mask')
    mask = 'NO_KNOWN_VISIBILITY' if not known else 'WRONG_ON_KNOWN' if wrong else 'MATCHES_ON_KNOWN'
    require(p['mask_state'] == mask, 'saved semantic mask category')
    expected = dict(eligible=solver['eligible'], solver_used=solver['used'], accepted_fit=solver['fit_input_ids'] if new else [],
        accepted_final_inliers=solver['final_inliers'] if new else [])
    pools = {name: pool_copy(p['pools'][name], corners, expected[name], name) for name in POOLS}
    require(set(H).isdisjoint(expected['solver_used']) and set(H).isdisjoint(expected['accepted_fit']) and 8 not in expected['accepted_fit'], 'H XY and center excluded from attempted/approved fit')
    require(g['reprojections_reused_as_observations'] is False and solver['reprojected_points_reused_as_observations'] is False, 'projection never re-fit')
    require(g['reprojected_ids'] == p['actually_reprojected_ids'] == (H if new else []) and g['hidden_reprojected'] == bool(new and H), 'only approved NEW actually projects H')
    projected = solver.get('projected'); assignments = solver['operation_counts'].get('hidden_reprojection_assignments', 0)
    require(assignments == (len(H) if new else 0), 'one assignment per hidden ID, no refit')
    for k in range(8):
        require(corners[k]['actually_reprojected'] == (new and k in H), 'actual corner projection flag')
        if new and k in H:
            require(g['native_points'][k] == projected[k] and g['output_coordinate_sources'][k] == 'FINAL_POSE_REPROJECTION', 'saved H output equals final stored projection')
        elif not new: require(g['native_points'][k] == g['original_native_N3_points'][k], 'full original N3 fallback')
    require(g['native_points'][8] == g['original_native_N3_points'][8], 'unchanged detector center')
    fit_edges = solver['fit_line_edges'] if new else []; final_edges = solver['final_line_inliers'] if new else []
    require(set(fit_edges).isdisjoint(solver['consumed_edges']) and set(fit_edges) <= set(solver['line_edges']), 'no consumed line/point double observation')
    if g['method'] == METHODS[1]: require(solver['line_edges'] == fit_edges == final_edges == [], 'same-q no-line ablation')
    selected = ids(p['selected_corner_ids'], 'selected boundary q'); require(selected == g['selected_corner_ids'] == ledger['actual_adopted_boundary_corner_ids'], 'same selected q IDs')
    boundary = copy.deepcopy(p['boundary_candidates'])
    for item in boundary:
        k = item['id']; require(item['supply_selected'] == (k in selected) and item['self_hidden_excluded'] == (k in H), 'boundary q supply/H flags')
        if item['supply_selected']: require(item['candidate_xy'] == g['input_points'][k], 'selected head q unchanged')
        require(item['physical_boundary_ownership_independently_verified'] is False, 'no independent ownership certificate')
    wrong_excluded = sorted((set(H) - humanH) & known); wrongly_retained_H = sorted((humanH - set(H)) & known)
    bad_used = pools['solver_used']['proxy_incorrect_ids']; bad_final = pools['accepted_final_inliers']['proxy_incorrect_ids']
    fit_geometry = solver.get('point_line_geometry') if new else None
    stable = solver.get('stable_incidence_guard') if new else None
    diagnostic = p['finite_geometry_diagnostics']
    require(diagnostic['actual_fit_ids'] == expected['accepted_fit'] and diagnostic['actual_fit_edges'] == fit_edges
        and diagnostic['final8px_inliers'] == expected['accepted_final_inliers'], 'posthoc actual fit/line/inlier IDs')
    layout = dict(input_points_px=copy.deepcopy(g['input_points']), physical_extents_xyz_m=copy.deepcopy(g['xyz']),
        incident_edge_ids_by_native_corner={str(k): [e for e, pair in enumerate(edges) if k in pair] for k in range(8)},
        actual_fit_corner_ids=expected['accepted_fit'], actual_fit_edge_ids=fit_edges, final_inlier_corner_ids=expected['accepted_final_inliers'],
        final_inlier_edge_ids=final_edges, fit_edge_endpoints={str(e): edges[e] for e in fit_edges},
        used_actual_line_ids=copy.deepcopy(solver['line_edges']), consumed_line_ids=copy.deepcopy(solver['consumed_edges']),
        stored_actual_fit_geometry=geometry_copy(fit_geometry),
        stored_stable_incidence_geometry=geometry_copy(stable.get('stable_incidence_geometry')) if stable else None,
        stored_stable_incidence_state=stable.get('guard_state') if stable else None,
        stored_stable_incidence_accepted=stable.get('accepted') if stable else None,
        stored_candidate_geometry=geometry_copy((solver.get('best_candidate') or {}).get('point_line_geometry')),
        stored_candidate_is_approved_NEW=new,
        camera_facing_cuboid_vertex_signs={str(k): list(CUBOID_VERTEX_SIGNS[k]) for k in range(8)},
        known_visibility_proxy_correct_vertex_signs_by_pool={name: {str(k): list(CUBOID_VERTEX_SIGNS[k])
            for k in pool['known_visibility_proxy_correct_ids']} for name, pool in pools.items()},
        vertex_signs_are_existing_model_ID_layout_not_certified_physical_axes=True,
        layout_or_rank_is_not_physical_ownership_or_global_uniqueness=True)
    return dict(id=g['id'], session=g['session'], stratum=label, method=g['method'], output_status=status,
        new_pose_estimated=new, fallback_used=fallback, pose_available=s['pose']['available'], solver_state=solver['state'],
        saved_pose=copy.deepcopy(s['pose']), saved_pose_outcome_vs_N3=p['pose_outcome_vs_N3'], saved_delta_vs_N3=copy.deepcopy(p['delta_vs_N3']),
        mask_state=mask, mask_audit=copy.deepcopy(audit), legacy_mask_audit=p['mask_definition_provenance']['legacy_mask_audit'],
        mask_errors=dict(known_ids=sorted(known), unknown_visibility_ids=sorted(set(range(8))-known),
            wrongly_excluded_known_nonself_ids=wrong_excluded, wrongly_retained_known_self_ids=wrongly_retained_H,
            proxy_correct_wrongly_excluded_ids=[k for k in wrong_excluded if corners[k]['input_quality'] == GOOD],
            incorrect_coordinate_retained_used_ids=bad_used, incorrect_coordinate_retained_final_inlier_ids=bad_final,
            human_self_remaining_used_ids=[k for k in expected['solver_used'] if k in humanH],
            human_self_remaining_final_inlier_ids=[k for k in expected['accepted_final_inliers'] if k in humanH]),
        pools=pools, corners=copy.deepcopy(corners), selected_head_boundary_ids=selected, head_boundary_candidates=boundary,
        selected_wrong_head_q_ids=[c['id'] for c in boundary if c['supply_selected'] and c['candidate_proxy_quality'] == BAD],
        selected_wrong_head_q_final_inlier_ids=[c['id'] for c in boundary if c['supply_selected'] and c['candidate_proxy_quality'] == BAD and c['accepted_new_pose_final_inlier']],
        raw_candidate_final_inlier_ids=copy.deepcopy(solver['final_inliers']), initial_H=H, reprojected_ids=copy.deepcopy(g['reprojected_ids']),
        hidden_assignment_count=assignments, no_H_initial_XY_in_accepted_fit=True, reprojected_points_refit=False,
        hidden_set_changed=p['hidden_set_changed'], layout=layout,
        prior_thirteen_provenance_links=copy.deepcopy(provenance.get(g['id'], [])),
        prior_thirteen_selection_does_not_select_new_V19_cases=True, physical_ownership_truth='UNKNOWN')


def summarize(records, cohort):
    summaries = {}; labels = {r['id']: r['label'] for r in cohort['frames']}
    for stratum, chosen in (('combined', cohort['ids']), ('easy', [i for i in cohort['ids'] if labels[i] == 'clean']),
                            ('medium', [i for i in cohort['ids'] if labels[i] == 'moderate'])):
        block = {}
        for method in METHODS:
            selected = [r for r in records if r['method'] == method and r['id'] in set(chosen)]
            require([r['id'] for r in selected] == chosen, 'complete ordered method/stratum')
            mask_pose, pool_cross, layout_cross, error_cross = (defaultdict(list) for _ in range(4))
            for row in selected:
                head = (row['mask_state'], row['saved_pose_outcome_vs_N3'], row['output_status'])
                mask_pose[head].append(row['id'])
                err = row['mask_errors']
                key = (*head, len(err['wrongly_excluded_known_nonself_ids']), len(err['wrongly_retained_known_self_ids']),
                    len(err['proxy_correct_wrongly_excluded_ids']), len(err['incorrect_coordinate_retained_used_ids']),
                    len(err['incorrect_coordinate_retained_final_inlier_ids']), len(row['selected_wrong_head_q_final_inlier_ids']))
                error_cross[key].append(row['id'])
                for pool in POOLS:
                    value = row['pools'][pool]
                    key = (*head, pool, len(value['proxy_correct_ids']), len(value['proxy_incorrect_ids']), len(value['proxy_unknown_ids']),
                        len(value['known_visibility_proxy_correct_ids']), len(value['unknown_visibility_ids']), len(value['directly_visible_proxy_correct_ids']))
                    pool_cross[key].append(row['id'])
                g = row['layout']; fg = g['stored_actual_fit_geometry'] or {}; stable = g['stored_stable_incidence_geometry'] or {}
                key = (*head, tuple(g['actual_fit_corner_ids']), tuple(g['actual_fit_edge_ids']),
                    (fg.get('observed_normal_joint') or {}).get('numerical_rank'), (fg.get('modeled_normal_joint') or {}).get('numerical_rank'),
                    (stable.get('observed_normal_joint') or {}).get('numerical_rank'), (stable.get('modeled_normal_joint') or {}).get('numerical_rank'))
                layout_cross[key].append(row['id'])
            def table(groups, names):
                return [dict(zip(names, key), n=len(value), ids=value) for key, value in groups.items()]
            base = ('mask_state', 'saved_pose_outcome_vs_N3', 'output_status')
            named = dict(mask_wrong_but_pose_both_better=[r['id'] for r in selected if r['mask_state'] == 'WRONG_ON_KNOWN' and r['saved_pose_outcome_vs_N3'] == 'BOTH_BETTER'],
                mask_matches_but_pose_both_worse=[r['id'] for r in selected if r['mask_state'] == 'MATCHES_ON_KNOWN' and r['saved_pose_outcome_vs_N3'] == 'BOTH_WORSE'],
                accepted_fit_at_least4_proxy_correct_but_pose_both_worse=[r['id'] for r in selected if len(r['pools']['accepted_fit']['proxy_correct_ids']) >= 4 and r['saved_pose_outcome_vs_N3'] == 'BOTH_WORSE'],
                accepted_fit_at_least4_known_visibility_proxy_correct_but_pose_both_worse=[r['id'] for r in selected if len(r['pools']['accepted_fit']['known_visibility_proxy_correct_ids']) >= 4 and r['saved_pose_outcome_vs_N3'] == 'BOTH_WORSE'],
                accepted_fit_at_least4_directly_visible_proxy_correct_but_pose_both_worse=[r['id'] for r in selected if len(r['pools']['accepted_fit']['directly_visible_proxy_correct_ids']) >= 4 and r['saved_pose_outcome_vs_N3'] == 'BOTH_WORSE'],
                wrong_coordinate_final_inlier_remaining=[r['id'] for r in selected if r['pools']['accepted_final_inliers']['proxy_incorrect_ids']],
                wrong_selected_head_q_final_inlier_remaining=[r['id'] for r in selected if r['selected_wrong_head_q_final_inlier_ids']],
                correct_proxy_point_removed_and_incorrect_coordinate_remaining_used=[r['id'] for r in selected
                    if r['mask_errors']['proxy_correct_wrongly_excluded_ids'] and r['mask_errors']['incorrect_coordinate_retained_used_ids']],
                direct_visible_damaged_5px_to_over10px=[r['id'] for r in selected if any(c['directly_visible_damage_5_to_10'] for c in r['corners'])],
                changed_H_set_with_approved_NEW=[r['id'] for r in selected if r['new_pose_estimated'] and r['hidden_set_changed']])
            block[method] = dict(frames=len(selected), ids=chosen, statuses=dict(Counter(r['output_status'] for r in selected)),
                solver_states=dict(Counter(r['solver_state'] for r in selected)),
                mask_pose_status_groups=table(mask_pose, base),
                mask_error_count_pose_status_groups=table(error_cross, (*base, 'wrong_excluded_known_nonself_count',
                    'wrong_retained_known_self_count', 'proxy_correct_wrongly_excluded_count', 'proxy_incorrect_retained_used_count',
                    'proxy_incorrect_retained_final_inlier_count', 'wrong_selected_head_q_final_inlier_count')),
                pool_quality_mask_pose_status_groups=table(pool_cross, (*base, 'pool', 'proxy_correct_count', 'proxy_incorrect_count', 'proxy_unknown_count',
                    'known_visibility_proxy_correct_count', 'unknown_visibility_count', 'directly_visible_proxy_correct_count')),
                layout_mask_pose_status_groups=table(layout_cross, (*base, 'actual_fit_corner_ids', 'actual_fit_edge_ids',
                    'stored_finite_observed_rank', 'stored_finite_modeled_rank', 'stored_incidence_observed_rank', 'stored_incidence_modeled_rank')),
                named_groups={k: dict(n=len(v), ids=v) for k, v in named.items()},
                hidden_corner_outputs=dict(approved_NEW=sum(r['new_pose_estimated'] for r in selected),
                    replaced_H_coordinates=sum(len(r['reprojected_ids']) for r in selected),
                    hidden_assignment_count=sum(r['hidden_assignment_count'] for r in selected)))
        summaries[stratum] = dict(frames=len(chosen), methods=block)
    return summaries


def calculate(f, cohort, edges, target):
    order = cohort['ids']; labels = {r['id']: r['label'] for r in cohort['frames']}; sessions = {r['id']: r['session'] for r in cohort['frames']}
    geometry = indexed(f['geometry:GEOMETRY.jsonl.gz'], METHODS, order, compact_geometry)
    posthoc = indexed(f['evaluation:POSTHOC_ROWS.jsonl.gz'], METHODS, order)
    scored = indexed(f['evaluation:PREDICTIONS.jsonl.gz'], METHODS, order, compact_score)
    ledgers = {}
    for ledger in rows(f['geometry:LEDGERS.jsonl.gz']):
        require(ledger['id'] in order and ledger['id'] not in ledgers, 'unique ledger frame')
        ledgers[ledger['id']] = {k: ledger[k] for k in ('id', 'session', 'initial_H', 'actual_adopted_boundary_corner_ids')}
    require(set(ledgers) == set(order), 'all245 actual paired ledgers')
    index = read(f['prior_provenance:REMAINING_THIRTEEN_QUERIES.json']); tags = read(f['prior_provenance:REMAINING_THIRTEEN_SOURCE_TAGS.json'])
    provenance = defaultdict(list); tag_map = {(v['frame_id'], v['native_corner_id']): v for v in tags['cases']}
    require(len(tag_map) == len(index['cases']) == 13, 'thirteen unique preserved provenance queries')
    for case in index['cases']:
        key = (case['frame_id'], case['corner_id']); require(key in tag_map, 'existing case/tag ID join')
        tag = tag_map[key]; provenance[key[0]].append(dict(native_corner_id=key[1],
            case_index_binding=binding(f['prior_provenance:REMAINING_THIRTEEN_QUERIES.json']),
            source_tags_binding=binding(f['prior_provenance:REMAINING_THIRTEEN_SOURCE_TAGS.json']),
            stored_annotation_source=tag['stored_annotation_point'].get('source', 'FIELD_MISSING'),
            saved_human_state=tag['saved_human_state'], physical_ID_mapping_certified=tag['physical_ID_mapping_certified'],
            provenance_is_prior_V18_literal_not_new_V19_physical_approval=True))
    output = []
    with gzip.open(target, 'xt', encoding='utf-8', compresslevel=6) as stream:
        for method in METHODS:
            for identity in order:
                key = (method, identity); g, p, s = geometry[key], posthoc[key], scored[key]
                require(g['session'] == sessions[identity], 'unchanged cohort session')
                r = record(g, p, s, ledgers[identity], labels[identity], edges, provenance)
                output.append(r); stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
    for identity in order:
        a, b = [geometry[(m, identity)] for m in METHODS]
        require(all(a[k] == b[k] for k in ('input_points', 'original_native_N3_points', 'hidden_initial', 'selected_corner_ids',
            'head_checkpoint_sha256', 'calibration_binding')), 'same-q paired ablation keeps head/native/H unchanged')
    return output, summarize(output, cohort)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('freeze', 'run'))
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--geometry-input', type=Path, required=True)
    p.add_argument('--validation-input', type=Path, required=True); p.add_argument('--evaluation-input', type=Path, required=True)
    p.add_argument('--statistics-input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.absolute()
    require(out.is_relative_to(PRIVATE) and out != PRIVATE and not any(v.is_symlink() for v in (out, *out.parents)), 'new nonsymlink private output subdirectory')
    for value in (args.repo, args.geometry_input, args.validation_input, args.evaluation_input, args.statistics_input):
        root = value.absolute(); require(not out.is_relative_to(root) and not root.is_relative_to(out), 'output overlaps protected input directory')
    out.mkdir(parents=True, exist_ok=True)
    for n in FILES if args.stage == 'freeze' else FILES[1:]: require(not (out / n).exists(), 'preserve existing ' + n)
    f = input_paths(args); require(not any(v.absolute().is_relative_to(out) for v in f.values()), 'output overlaps immutable inputs')
    b = {k: binding(v) for k, v in f.items()}; cohort, edges = gate(f, b)
    policy = dict(methods=list(METHODS), pools=list(POOLS), ordered_ids=cohort['ids'], native_edges=edges,
        camera_facing_cuboid_vertex_signs=[list(v) for v in CUBOID_VERTEX_SIGNS],
        proxy_quality_threshold_px=8, threshold_is_unchanged_saved_label_not_new_rethresholding=True,
        semantic_known_visibility_excludes=list(VISIBILITY_UNKNOWN), stratum_counts=dict(combined=245, easy=153, medium=92), limits=LIMITS)
    if args.stage == 'freeze':
        write(out / FILES[0], dict(schema='saved_observed_frame_correspondence_protocol_v19', inputs=b, checker=b['checker'],
            policy=policy, frozen_before_raw_decode_and_own_counting=True)); return
    own = read(out / FILES[0]); require(own['schema'] == 'saved_observed_frame_correspondence_protocol_v19'
        and own['inputs'] == b and own['policy'] == policy and own['frozen_before_raw_decode_and_own_counting'] is True, 'own frozen source/inputs/policy')
    protocol = binding(out / FILES[0]); write(out / FILES[1], dict(inputs=b, own_protocol=protocol, checker=b['checker'], new_production_calls=0))
    error = None; result = []; summary = {}
    try:
        result, summary = calculate(f, cohort, edges, out / FILES[3])
        require(len(result) == 490 and b == {k: binding(v) for k, v in f.items()}, 'complete490 and unchanged inputs')
        write(out / FILES[4], dict(schema='saved_observed_frame_correspondence_summary_v19', complete=True, inputs=b,
            own_protocol=protocol, methods=list(METHODS), strata=summary, limits=LIMITS,
            remaining_good_counts_are_proxy_evidence_not_pose_or_ownership_guarantee=True,
            direct_visible_and_self_hidden_errors_preserved_separately_in_rows=True))
    except BaseException as exc: error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    write(out / FILES[2], dict(schema='saved_observed_frame_correspondence_checks_v19', complete=error is None,
        passed=error is None, inputs=b, own_protocol=protocol, checker=b['checker'], error=error,
        actual_counts=dict(rows=len(result), frames=245 if error is None else None, methods=2,
            strata=3 if error is None else None, pool_membership_checks=1960 if error is None else None),
        rows=binding(out / FILES[3]) if (out / FILES[3]).exists() else None,
        summary=binding(out / FILES[4]) if (out / FILES[4]).exists() else None, limits=LIMITS,
        new_production_calls=0, old_full_V16_validation_passed=False,
        diagnostic_PASS_is_membership_count_join_not_independent_pose_quality_or_goal_PASS=True))
    if error: raise SystemExit(1)


if __name__ == '__main__': main()
