"""Public-only arithmetic audit of the fixed v5 supplemental line factors.

No research module, model, GT cache, PnP, Jacobian/SVD, optimizer or ray engine
is imported. Saved Jacobian rank labels are counted, not independently replayed.
Run once after the complete geometry and scoring outputs have been published.
"""
from __future__ import annotations

from collections import Counter
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_partial_line_independent_20261010_v5'
PRIMARY = 'N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES'
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
SOURCE_REGISTRY_EDGES = frozenset((0, 2, 4, 6, 8, 9, 10, 11))
NUMERICAL_ATOL = 1e-7
NUMERICAL_RTOL = 1e-12


def read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def binding(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=path.name, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def bound(path, saved):
    actual = binding(path)
    if any(actual[k] != saved[k] for k in ('sha256', 'bytes')):
        raise ValueError('published input binding differs: ' + path.name)


def output_guard(path):
    if path.is_symlink() or path.exists():
        raise ValueError('preserve existing output: ' + str(path))
    resolved = path.resolve()
    if resolved.is_relative_to(REPO) and resolved != DOC / 'LINE_FACTOR_CHECKS.json':
        raise ValueError('only the new LINE_FACTOR_CHECKS receipt may be written inside the repository')
    if not resolved.parent.is_dir():
        raise ValueError('output parent must already exist')
    if not resolved.is_relative_to(REPO):
        roots = (Path(tempfile.gettempdir()).resolve(), Path('/dev/shm'))
        if not any(resolved.is_relative_to(root) for root in roots) or any(resolved.parent.iterdir()):
            raise ValueError('external output requires a new empty temporary directory')
    for parent in path.absolute().parents:
        if parent.is_symlink():
            raise ValueError('symlink output parent')
    return resolved


class Audit:
    def __init__(self):
        self.checks = 0
        self.failure_count = 0
        self.failures = []
        self.max_numeric_difference = 0.0

    def ok(self, value, label, actual=None, expected=None):
        self.checks += 1
        if not value:
            self.failure_count += 1
            if len(self.failures) < 100:
                self.failures.append(dict(check=label, actual=safe(actual), expected=safe(expected)))

    def equal(self, actual, expected, label):
        self.ok(actual == expected, label, actual, expected)

    def close(self, actual, expected, label):
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.ok(False, label + '.shape', actual, expected)
                return
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.close(a, e, label + '.' + str(i))
            return
        valid = isinstance(actual, (int, float)) and math.isfinite(actual) and math.isfinite(expected)
        difference = abs(actual - expected) if valid else None
        if valid:
            self.max_numeric_difference = max(self.max_numeric_difference, difference)
        self.ok(valid and difference <= NUMERICAL_ATOL + NUMERICAL_RTOL * abs(expected), label, actual, expected)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def project(dims, R, t, K):
    a, b, c = (v / 2 for v in dims)
    model = ((-a, -b, -c), (a, -b, -c), (a, b, -c), (-a, b, -c),
             (-a, -b, c), (a, -b, c), (a, b, c), (-a, b, c))
    projected = []
    for point in model:
        camera = [dot(axis, point) + shift for axis, shift in zip(R, t)]
        uvw = [dot(axis, camera) for axis in K]
        if not all(math.isfinite(v) for v in uvw) or uvw[2] <= 0:
            raise ValueError('stored numeric candidate has invalid projected depth')
        projected.append([uvw[0] / uvw[2], uvw[1] / uvw[2]])
    return projected


def line_pool(row, observation, audit):
    solver = row['solver']
    U = solver['used']
    adopted = [k for k in row['observation_contract']['hybrid_boundary_corner_ids'] if k in U]
    corners = {c['id']: c for c in observation['corners']}
    consumed = set()
    for k in adopted:
        c = corners[k]
        audit.equal(row['input_points'][k], c['xy'], row['id'] + '.actual_boundary.' + str(k))
        audit.ok(len(c['edges']) == 2 and all(k in EDGES[e] for e in c['edges']),
                 row['id'] + '.incident_edges.' + str(k))
        consumed.update(c['edges'])
    audit.equal(solver['consumed_edges'], sorted(consumed), row['id'] + '.consumed')
    seen, allowed = {}, {}
    for raw in observation['lines']:
        e = raw['edge']
        if e in seen:
            audit.equal(raw, seen[e], row['id'] + '.duplicate_edge.' + str(e))
            continue
        seen[e] = raw
        audit.equal(raw['endpoints'], list(EDGES[e]), row['id'] + '.edge_endpoint_identity.' + str(e))
        norm = math.hypot(*raw['normal'])
        audit.ok(norm > 0 and math.isfinite(norm), row['id'] + '.normal.' + str(e))
        n = [v / norm for v in raw['normal']]
        offset = raw['offset'] / norm
        support = raw['support_points']
        queries, radii = raw['queries'], raw['query_radii_px']
        audit.ok(len(support) == len(queries) == len(radii), row['id'] + '.support_shape.' + str(e))
        audit.ok(len(set(queries)) >= 3 and len(set(queries)) == len(queries) and
                 all(0 <= q < 84 and q // 7 == e for q in queries),
                 row['id'] + '.support_query_identity.' + str(e))
        errors = [abs(dot(p, n) - offset) for p in support]
        good = all(math.isfinite(r) and r >= 0 and error <= r + 1e-9
                   for error, r in zip(errors, radii))
        if e in consumed:
            audit.ok(good, row['id'] + '.adopted_support_consistency.' + str(e))
        if good and e in SOURCE_REGISTRY_EDGES and e not in consumed:
            allowed[e] = dict(normal=n, offset=offset, queries=queries,
                              support_errors=errors, radii=radii)
    audit.ok(consumed <= set(seen), row['id'] + '.consumed_support_exists')
    audit.equal(solver['line_edges'], sorted(allowed), row['id'] + '.fixed_line_pool')
    audit.ok(not (consumed & set(allowed)), row['id'] + '.no_point_line_duplicate')
    pool = solver.get('factor_pool')
    if pool is not None:
        expected = dict(point_ids=U, line_edges=sorted(allowed), actual_point_factors=len(U),
                        unique_line_factors=len(allowed),
                        line_support_queries=sum(len(v['queries']) for v in allowed.values()),
                        scalar_residuals=2 * (len(U) + len(allowed)))
        audit.equal(pool, expected, row['id'] + '.factor_units')
    return allowed


def factor_values(q, inputs, U, lines):
    point = [math.hypot(q[k][0] - inputs[k][0], q[k][1] - inputs[k][1]) for k in U]
    edge = []
    for e in sorted(lines):
        line = lines[e]
        distances = [dot(q[k], line['normal']) - line['offset'] for k in EDGES[e]]
        edge.append(math.sqrt(math.fsum(d * d for d in distances) / 2))
    return point, edge


def candidate_check(row, candidate, lines, audit, counts, label):
    U = row['solver']['used']
    q = project(candidate['dimensions'], candidate['R_cf'], candidate['centroid'], row['K'])
    audit.close(candidate['projected'], q, label + '.projection')
    audit.equal(candidate['same_scoring_point_ids'], U, label + '.same_point_pool')
    audit.equal(candidate['same_scoring_line_edges'], sorted(lines), label + '.same_line_pool')
    rp, rl = factor_values(q, row['input_points'], U, lines)
    audit.close(candidate['point_residual_norms_px'], rp, label + '.point_errors')
    audit.close(candidate['line_residual_RMS_px'], rl, label + '.line_errors')
    endpoint_errors = [[dot(q[k], lines[e]['normal']) - lines[e]['offset'] for k in EDGES[e]]
                       for e in sorted(lines)]
    audit.close(candidate['line_endpoint_signed_residuals_px'], endpoint_errors, label + '.line_endpoint_errors')
    ip = [k for k, r in zip(U, rp) if r <= 8]
    il = [e for e, r in zip(sorted(lines), rl) if r <= 8]
    for key, expected in dict(inlier_ids=ip, line_inlier_edges=il, point_inlier_count=len(ip),
                              line_inlier_count=len(il), total_inlier_factor_count=len(ip) + len(il)).items():
        audit.equal(candidate[key], expected, label + '.' + key)
    audit.close(candidate['truncated_sse_px2'], math.fsum(min(r * r, 64) for r in rp + rl), label + '.truncated_sse')
    audit.close(candidate['sse_px2'], math.fsum(r * r for r in rp + rl), label + '.sse')
    blocked = set(row['hidden_initial']) | set(row['solver'].get('temporary_excluded', []))
    for key in ('generator_ids', 'actual_fit_input_ids'):
        audit.ok(set(candidate[key]) <= set(U) and not (blocked & set(candidate[key])), label + '.' + key)
    audit.ok(set(candidate['actual_fit_line_edges']) <= set(lines), label + '.actual_fit_edges')
    counts['raw_numeric_candidates'] += 1
    counts['raw_candidate_point_inliers'] += len(ip)
    counts['raw_candidate_line_inliers'] += len(il)
    return q, rp, rl, ip, il


def run(folder, output, protocol_path):
    output = output_guard(output)
    names = ('OBSERVATIONS.jsonl.gz', 'GEOMETRY_SEALED.jsonl.gz', 'PREDICTIONS.jsonl.gz',
             'GEOMETRY_SEAL.json', 'SCORING_RECEIPT.json', 'PROTOCOL.json')
    paths = {name: folder / name for name in names}
    paths['PROTOCOL.json'] = protocol_path
    before = {name: binding(path) for name, path in paths.items()}
    before['checker'] = binding(Path(__file__).resolve())
    result = dict(schema='public_only_v5_line_factor_arithmetic_v1', passed=False, complete=False,
                  inputs=before, numerical_arithmetic_absolute_tolerance=NUMERICAL_ATOL,
                  numerical_arithmetic_relative_tolerance=NUMERICAL_RTOL,
                  observation_inlier_threshold_px=8, source_consensus_tolerance_px=1e-9,
                  new_model_PnP_GT_optimizer_ray_training_RGB_calls=0,
                  independently_certified_real_physical_edge_ownership=False,
                  Jacobian_SVD_recomputed=False, rank_scope='counts of stored local rank labels only',
                  limits=['Stored observations and registry cuboid endpoint arithmetic are verified.',
                          'No real physical edge ownership, GT authenticity or global pose uniqueness is certified.',
                          'Raw fallback candidate factors are not labeled accepted operational line fits.',
                          'Saved no-refit flags and immutable geometry/scored joins are checked; runtime causality is not replayed.'])
    audit, counts = Audit(), Counter()
    try:
        seal, scoring, protocol = (read(paths[n]) for n in ('GEOMETRY_SEAL.json', 'SCORING_RECEIPT.json', 'PROTOCOL.json'))
        audit.ok(seal['complete'] and not seal['GT_read_allowed'] and seal['frames'] == 245 and
                 seal['rows'] == 980 and seal['fixed_rows'] == 490, 'complete_geometry_seal')
        audit.ok(scoring['complete'] and scoring['scored_method_rows'] == 980 and scoring['new_pose_fits'] == 0,
                 'scoring_no_new_fit')
        bound(paths['GEOMETRY_SEAL.json'], scoring['geometry_seal'])
        bound(paths['GEOMETRY_SEALED.jsonl.gz'], seal['geometry'])
        bound(paths['OBSERVATIONS.jsonl.gz'], seal['observations'])
        bound(paths['PREDICTIONS.jsonl.gz'], scoring['predictions'])
        bound(paths['PROTOCOL.json'], seal['protocol'])
        observations, geometry, scored = (rows(paths[n]) for n in names[:3])
        obs = {r['id']: r for r in observations}
        by = {(r['method'], r['id']): r for r in scored}
        audit.equal(len(observations), 245, 'observations_count')
        audit.equal(len(obs), 245, 'observation_uniqueness')
        audit.equal(len(geometry), 980, 'geometry_count')
        audit.equal(len(scored), 980, 'scored_count')
        audit.equal(len({(r['method'], r['id']) for r in geometry}), 980, 'geometry_uniqueness')
        audit.equal(len(by), len(scored), 'scored_uniqueness')
        audit.equal(Counter(r['method'] for r in geometry), {m:245 for m in protocol['methods']}, 'method_population')
        for method in protocol['methods']:
            audit.equal({r['id'] for r in geometry if r['method'] == method}, set(obs), 'method_IDs.' + method)
        audit.equal({(r['method'], r['id']) for r in geometry}, set(by), 'sealed_scored_population')
        primary = [r for r in geometry if r['method'] == PRIMARY]
        audit.equal(len(primary), 245, 'primary_population')
        audit.equal({r['id'] for r in primary}, set(obs), 'primary_observation_IDs')
        for row in geometry:
            saved = by[(row['method'], row['id'])]
            for key, value in row.items():
                audit.ok(key in saved, row['id'] + '.' + row['method'] + '.sealed_key_present.' + key)
                audit.equal(saved.get(key), value, row['id'] + '.' + row['method'] + '.sealed_join.' + key)
        for row in primary:
            fid, solver = row['id'], row['solver']
            H, U = row['hidden_initial'], solver['used']
            new = bool(solver['available'])
            audit.equal(row['new_pose_estimated'], new, fid + '.new_status')
            audit.ok(not new or row['pose_available'], fid + '.new_operational_available')
            audit.ok(not new or len(solver['final_inliers']) >= 4, fid + '.new_min_four_actual_point_inliers')
            audit.ok(set(solver['final_inliers']) <= set(U), fid + '.final_inliers_are_allowed_points')
            audit.equal(solver['global_uniqueness_proven'], False, fid + '.no_global_uniqueness_certificate')
            audit.equal(set(solver['excluded']), set(H), fid + '.H_exclusion')
            for key in ('used', 'fit_input_ids', 'final_inliers', 'generator_ids'):
                audit.ok(not (set(H) & set(solver.get(key, []))), fid + '.H_no_fit.' + key)
            audit.equal(row['native_points'][8], obs[fid]['native_N3_points'][8], fid + '.center')
            audit.equal(solver['reprojected_points_reused_as_observations'], False, fid + '.no_projection_refit')
            audit.equal(row['reprojections_reused_as_observations'], False, fid + '.no_output_refit')
            lines = line_pool(row, obs[fid], audit)
            counts['primary_rows'] += 1
            counts['selected_unique_line_factors'] += len(lines)
            counts['selected_support_queries'] += sum(len(l['queries']) for l in lines.values())
            counts['line_pool_empty_rows' if not lines else 'line_pool_nonempty_rows'] += 1
            if not lines:
                counts['point_only_delegated_rows'] += int(bool(solver.get('empty_line_pool_exact_point_delegate')))
                audit.equal(solver.get('empty_line_pool_exact_point_delegate'), True, fid + '.empty_pool_exact_point_only')
                audit.equal(solver.get('selected_pose_refined_with_lines', False), False, fid + '.empty_no_line_fit')
                audit.equal(solver.get('final_line_inliers', []), [], fid + '.empty_no_line_inliers')
                audit.equal(solver.get('fit_line_edges', []), [], fid + '.empty_no_line_fit_edges')
                control = by[('N3_INDEPENDENT_CORNERWISE_ROLE', fid)]
                audit.equal(row['input_points'], control['input_points'], fid + '.empty_same_control_coordinates')
                for key in ('available', 'state', 'reason', 'fit_input_ids', 'final_inliers', 'R_cf',
                            'R_physical', 'centroid', 'cf_extents', 'projected', 'truncated_sse_px2', 'sse_px2'):
                    audit.equal(solver.get(key), control['solver'].get(key), fid + '.empty_exact_point_delegate.' + key)
                counts['delegated_point_only_numeric_candidates'] += len(solver.get('all_candidate_solutions', []))
            else:
                for i, c in enumerate(solver.get('all_candidate_solutions', [])):
                    candidate_check(row, c, lines, audit, counts, fid + '.candidate.' + str(i))
                if solver.get('best_candidate'):
                    q, rp, rl, ip, il = candidate_check(row, solver['best_candidate'], lines, audit, Counter(), fid + '.best')
                    audit.equal(solver['final_inliers'], ip, fid + '.final_point_inliers')
                    audit.equal(solver['final_line_inliers'], il, fid + '.final_edge_inliers')
                    audit.close(solver['residuals_used_px'], rp, fid + '.point_residuals')
                    audit.close(solver['line_residual_RMS_px'], rl, fid + '.line_residuals')
                    audit.equal(solver['point_inlier_count'], len(ip), fid + '.point_inlier_count')
                    audit.equal(solver['unique_line_inlier_count'], len(il), fid + '.edge_inlier_count')
                    audit.equal(solver['total_inlier_factor_count'], len(ip) + len(il), fid + '.unit_inlier_count')
                    if new:
                        audit.ok(len(ip) >= 4, fid + '.accepted_min_four_actual_point_inliers')
                        audit.close(solver['projected'], q, fid + '.accepted_projection')
                        audit.close(solver['truncated_sse_px2'], math.fsum(min(r*r, 64) for r in rp + rl), fid + '.accepted_truncated_sse')
            for attempt in solver.get('optimizer_attempts', []):
                audit.ok(set(attempt['fit_point_ids']) <= set(U) and len(attempt['fit_point_ids']) >= 4,
                         fid + '.optimizer_actual_point_pool')
                audit.ok(set(attempt['fit_line_edges']) <= set(lines), fid + '.optimizer_line_pool')
                audit.equal(attempt['initial_pose_used'], False, fid + '.optimizer_no_initial_pose')
                counts['stored_optimizer_attempts'] += 1
            geo = solver.get('point_line_geometry')
            if geo:
                counts['stored_candidate_geometry_rows'] += 1
                for key in ('observed_normal_joint', 'modeled_normal_joint', 'point_only',
                            'observed_normal_lines_only', 'modeled_normal_lines_only'):
                    counts['stored_' + key + '_rank' + str(geo[key]['numerical_rank'])] += 1
                audit.equal(geo['modeled_normals_held_fixed_for_derivative'], True, fid + '.rank_normal_scope')
                audit.equal(geo['global_unique_pose_proven'], False, fid + '.rank_not_global_uniqueness')
                if new:
                    audit.equal(geo['observed_normal_joint']['numerical_rank'], 6, fid + '.accepted_observed_local_rank')
                    audit.equal(geo['modeled_normal_joint']['numerical_rank'], 6, fid + '.accepted_modeled_local_rank')
                    counts['accepted_local_point_line_geometry_rows'] += 1
            counts['accepted_new_pose_rows' if new else 'fallback_or_failed_rows'] += 1
            if new:
                q = project(solver['cf_extents'], solver['R_cf'], solver['centroid'], row['K'])
                audit.close(solver['projected'], q, fid + '.final_Rt_projection')
                for k in H:
                    audit.close(row['native_points'][k], q[k], fid + '.actual_H_replacement.' + str(k))
                    counts['accepted_H_reprojected_corners'] += 1
                audit.equal(row['reprojected_ids'], H, fid + '.H_reprojected_IDs')
            else:
                audit.equal(row['native_points'], obs[fid]['native_N3_points'], fid + '.full_native_fallback')
                audit.equal(row['reprojected_ids'], [], fid + '.fallback_no_new_reprojection')
        for name, path in paths.items():
            audit.equal(binding(path), before[name], 'input_unchanged.' + name)
        result.update(complete=True)
    except Exception as error:
        audit.ok(False, 'exception', dict(type=type(error).__name__, message=str(error)))
    result.update(passed=audit.failure_count == 0 and result['complete'], checks=audit.checks,
                  failure_count=audit.failure_count, failures=audit.failures, actual_counts=dict(counts),
                  maximum_numeric_absolute_difference=audit.max_numeric_difference)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], complete=result['complete'], checks=audit.checks,
                          failures=audit.failure_count, output=str(output)), ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DOC, help='published v5 result folder')
    parser.add_argument('--output', type=Path, default=DOC / 'LINE_FACTOR_CHECKS.json')
    parser.add_argument('--protocol', type=Path, default=DOC / 'PROTOCOL.json', help='frozen v5 protocol')
    args = parser.parse_args()
    run(args.input.resolve(), args.output, args.protocol.resolve())


if __name__ == '__main__':
    main()
