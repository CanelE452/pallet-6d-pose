"""One frozen saved-row audit of unused line factors and point evidence.

The reference is the existing GEOMETRIC_PROXY endpoint pair in the fixed N3
phase. Endpoint-to-infinite-line agreement does not certify physical ownership,
visible-edge support, or a fraction of an actual 3D boundary. No solver, model,
reference loader, image reader, or statistics module is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_partial_line_independent_20261010_v5'
PRIVATE = Path('/tmp/pallet-partial-line-independent-private-20261010-v5')
PRIMARY = 'N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES'
POINT_ONLY = 'N3_INDEPENDENT_CORNERWISE_ROLE'
METHODS = (PRIMARY, 'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK', POINT_ONLY)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
LIMIT = 8.0
GOOD = 'PROXY_AGREEMENT_WITHIN8PX'
BAD = 'PROXY_MISMATCH_OVER8PX'
UNKNOWN = 'UNKNOWN_REFERENCE_OR_OBSERVATION'
FILES = ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json',
         'SCORING_RECEIPT.json', 'V4_CONTROL_PARITY.json', 'OBSERVATIONS.jsonl.gz',
         'PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz')
OUTPUTS = ('LINE_QUALITY_STARTED.json', 'LINE_QUALITY_ROWS.jsonl.gz',
           'LINE_QUALITY_CHECKS.json', 'LINE_QUALITY_REPORT_KO.md')
POLICY = dict(
    population='all fixed easy153+medium92=245 frames; four scored arms plus fresh Base/N3 controls',
    reference='saved evaluation_reference.native_points_px and valid_native_ids; fixed N3_SUBPIX phase',
    line_error='RMS of two signed normal distances of known proxy native endpoints to the normalized observed infinite line',
    agreement_threshold_px=LIMIT,
    line_unknown='saved reference matched=False, or either reference endpoint invalid/missing/nonfinite',
    invalid_observed_line_contract='preserve failed audit; do not turn invalid normals into a valid UNKNOWN factor',
    consensus_unit='one distinct semantic edge; support query count is a separate quantity',
    actual_unused_pool='solver.line_edges, independently compared with factor_pool and source checks',
    consumed='incident edges of actually adopted boundary corners present in allowed point pool',
    approved_inliers='NEW_POSE and new_pose_estimated and solver.available and solver.state=NEW_POSE only',
    rejected_candidate_inliers='raw best-candidate diagnostics on baseline fallback; never approved final inliers',
    point_error='Euclidean distance of saved input_points[corner] to same matched known proxy native reference, threshold8px; nonfinite or [-1,-1] is unknown',
    dimension_relation='actual returned cf_extents compared with fixed N3 cf_extents at atol1e-9 rtol0; also test [2,1,0] permutation',
    outcomes='saved T/R/ADDsym deltas only; strict both-better/both-worse; no new pose scoring',
    physical_boundary_ownership_certified=False,
    visible_physical_line_truth_available=False,
    along_edge_3D_support_fraction_estimated=False,
    rank_scope='stored candidate local geometry only; approved NEW kept separate from rejected candidate',
    new_model_PnP_optimizer_ray_training_RGB_GT_pose_scoring_calls=0,
    settings_selected_from_real_scores=False,
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    try:
        name, origin = str(path.relative_to(REPO)), 'public_repository'
    except ValueError:
        name, origin = path.name, 'external_readonly_input'
    return dict(path=name, origin=origin, bytes=path.stat().st_size, sha256=sha(path))


def inputs(folder):
    result = {name: binding(folder / name) for name in FILES}
    result['audit_code'] = binding(__file__)
    return result


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def guard_directories(folder, output):
    for path in (folder, output):
        require(not any(p.is_symlink() for p in (path, *path.parents)),
                'symlink audit input/output ancestor')
    resolved = output.resolve()
    require(resolved == DOC.resolve() or resolved.is_relative_to(PRIVATE.resolve()),
            'audit output must be exact new v5 DOC or new v5 private subtree')


def bind_matches(path, expected):
    actual = binding(path)
    require(actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256'],
            'immutable raw binding mismatch: ' + Path(path).name)


def freeze(folder, output):
    guard_directories(folder, output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('LINE_QUALITY_PROTOCOL.json', *OUTPUTS):
        require(not (output / name).exists(), 'preserve completed output ' + name)
    write(output / 'LINE_QUALITY_PROTOCOL.json', dict(
        schema='fixed_same_observation_proxy_line_quality_protocol_v5',
        frozen_before_actual_arithmetic=True, actual_arithmetic_runs=0,
        policy=POLICY, inputs=inputs(folder)))
    print('LINE_QUALITY_FROZEN', sha(output / 'LINE_QUALITY_PROTOCOL.json'), flush=True)


def ids(value, name):
    result = [int(k) for k in value]
    require(len(set(result)) == len(result), 'duplicate ' + name)
    require(all(0 <= k < 8 for k in result), 'invalid corner ' + name)
    return result


def approved(row):
    solver = row.get('solver') or {}
    result = bool(row['new_pose_estimated'])
    require(result == (row['output_status'] == 'NEW_POSE'), 'row NEW status disagreement')
    if result:
        require(solver.get('available') is True and solver.get('state') == 'NEW_POSE',
                'approved output lacks accepted solver pose')
    return result


def reference(row):
    ref = row['evaluation_reference']
    require(ref['phase_control'] == 'N3_SUBPIX', 'proxy phase is not fixed N3')
    q = np.asarray(ref['native_points_px'], dtype=float)
    require(q.shape == (8, 2), 'proxy native reference shape differs')
    require(isinstance(ref['matched'], bool), 'saved proxy match state missing')
    declared = set(ids(ref['valid_native_ids'], 'valid reference IDs'))
    valid = declared if ref['matched'] else set()
    require(all(np.isfinite(q[k]).all() for k in valid), 'nonfinite declared valid reference')
    return q, valid


def point_evidence(row):
    q, valid = reference(row)
    observed = np.asarray(row['input_points'], dtype=float)
    require(observed.shape == (9, 2), 'native input coordinate shape differs')
    solver = row['solver']
    pool = set(ids(solver.get('used', []), 'point pool'))
    fit = set(ids(solver.get('fit_input_ids', []), 'actual fit IDs'))
    raw_inliers = set(ids(solver.get('final_inliers', []), 'raw candidate inliers'))
    new = approved(row)
    require(fit <= pool and raw_inliers <= pool, 'point fit/inlier outside allowed pool')
    blocked = set(row['hidden_initial']) | set(solver.get('temporary_excluded', []))
    require(not pool & blocked, 'blocked point enters actual pool')
    if new:
        require(len(raw_inliers) >= 4, 'NEW final actual point inlier count below four')
    values = []
    for k in range(8):
        error = (float(np.linalg.norm(observed[k] - q[k]))
                 if k in valid and np.isfinite(observed[k]).all() and not np.all(observed[k] == -1) else None)
        quality = UNKNOWN if error is None else GOOD if error <= LIMIT else BAD
        values.append(dict(id=k, input_xy=observed[k], proxy_reference_xy=q[k],
            proxy_reference_valid=k in valid, proxy_error_px=error, quality=quality,
            proxy_reference_matched=row['evaluation_reference']['matched'],
            in_allowed_point_pool=k in pool, in_actual_fit_ids=k in fit,
            approved_final_point_inlier=new and k in raw_inliers,
            raw_candidate_point_inlier=k in raw_inliers,
            raw_candidate_scope='APPROVED_NEW' if new else 'REJECTED_OR_UNAVAILABLE_CANDIDATE',
            predicted_H_excluded=k in set(row['hidden_initial'])))
    return values


def extent_relation(row, fixed):
    a = (row.get('actual_pose') or {}).get('cf_extents')
    b = (fixed.get('actual_pose') or {}).get('cf_extents')
    if a is None or b is None:
        return dict(relation='UNKNOWN', actual_cf_extents=a, fixed_N3_cf_extents=b)
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape == (3,) and np.isfinite(a).all() and np.isfinite(b).all(),
            'returned extent contract differs')
    relation = ('SAME' if np.allclose(a, b, atol=1e-9, rtol=0) else
                'WD_SWITCHED' if np.allclose(a, b[[2, 1, 0]], atol=1e-9, rtol=0) else
                'OTHER_RETURNED_EXTENTS')
    return dict(relation=relation, actual_cf_extents=a, fixed_N3_cf_extents=b,
                physical_branch_correctness_certified=False)


def saved_scores(row):
    p = row['pose']
    require(p['available'], 'operational proxy pose score is absent')
    result = dict(translation_cm=float(p['translation_cm']), rotation_deg=float(p['rotation_deg']),
                  ADDsym_cm=float(p['ADDsym_m']) * 100.)
    require(all(math.isfinite(v) for v in result.values()), 'nonfinite recorded score')
    return result


def paired(a, b):
    delta = {k: a[k] - b[k] for k in a}
    T, R = delta['translation_cm'], delta['rotation_deg']
    outcome = 'BOTH_BETTER' if T < 0 and R < 0 else 'BOTH_WORSE' if T > 0 and R > 0 else 'MIXED_OR_UNCHANGED'
    return dict(delta=delta, pose_T_R_outcome=outcome)


def distribution(values):
    a = np.asarray(values, float)
    if not len(a):
        return dict(n=0, mean=None, sample_variance=None, sample_std=None, median=None, P90=None, max=None)
    mean = math.fsum(float(v) for v in a) / len(a)
    var = math.fsum((float(v) - mean) ** 2 for v in a) / (len(a) - 1) if len(a) > 1 else None
    return dict(n=len(a), mean=mean, sample_variance=var,
                sample_std=math.sqrt(var) if var is not None else None,
                median=float(np.quantile(a, .5)), P90=float(np.quantile(a, .9)), max=float(a.max()))


def quality_counter(values, flag=None):
    return dict(Counter(v['quality'] for v in values if flag is None or v[flag]))


def line_evidence(row, observation, counters):
    solver = row['solver']
    pool = [int(e) for e in solver.get('line_edges', [])]
    consumed = {int(e) for e in solver.get('consumed_edges', [])}
    checks = solver.get('line_contract_checks', [])
    new = approved(row)
    raw_inliers = {int(e) for e in solver.get('final_line_inliers', [])}
    fit_edges = {int(e) for e in solver.get('fit_line_edges', [])}
    require(len(pool) == len(set(pool)) and not set(pool) & consumed,
            'duplicate or consumed edge in actual unused line pool')
    require(raw_inliers <= set(pool) and fit_edges <= set(pool), 'line fit/inlier outside pool')
    source = {int(l['edge']): l for l in observation['lines']}
    require(len(source) == len(observation['lines']) and all(0 <= e < 12 for e in source),
            'source line identity is duplicate or invalid')
    require(set(pool) <= set(source) and consumed <= set(source), 'solver line not in shared observations')
    accepted_checks = {int(c['edge']) for c in checks if c['accepted']}
    require(accepted_checks == set(pool), 'declared actual line checks and factor pool differ')
    corners = {int(c['id']): c for c in observation['corners']}
    adopted = set((row.get('observation_contract') or {}).get('hybrid_boundary_corner_ids', []))
    U = set(solver.get('used', []))
    expected = set()
    for k in adopted & U:
        require(k in corners, 'adopted corner not in actual source observation')
        require(np.array_equal(np.asarray(corners[k]['xy'], float), np.asarray(row['input_points'][k], float)),
                'adopted line-consuming coordinate differs from observation')
        incident = [int(e) for e in corners[k]['edges']]
        require(len(incident) == len(set(incident)) == 2 and all(k in EDGES[e] for e in incident),
                'adopted corner incident edge identity differs')
        expected.update(incident)
    require(consumed == expected, 'actual consumed edges differ from adopted allowed corner graph')
    factor = solver.get('factor_pool')
    if factor is not None:
        require(set(factor['line_edges']) == set(pool) and set(factor['point_ids']) == U,
                'reported factor pool differs from actual fixed pool')
    delegate = bool(solver.get('empty_line_pool_exact_point_delegate'))
    if delegate:
        require(not pool and not raw_inliers and not fit_edges,
                'point-only delegate has a line factor/inlier')
    for c in solver.get('all_candidate_solutions', []):
        if delegate:
            # Exact v4 delegation deliberately retains its point-only schema.
            require(set(c['inlier_ids']) <= U and set(c['actual_fit_input_ids']) <= U,
                    'delegated point candidate fit/inlier outside retained U')
            counters['delegated_point_candidate_checks'] += 1
        else:
            require(c['same_scoring_line_edges'] == pool and set(c['same_scoring_point_ids']) == U,
                    'candidate used a different scoring factor pool')
            counters['candidate_factor_pool_checks'] += 1
    q, valid = reference(row)
    values = []
    for edge, raw in sorted(source.items()):
        require(tuple(int(k) for k in raw['endpoints']) == EDGES[edge], 'observed semantic endpoints differ')
        normal = np.asarray(raw['normal'], float)
        offset = float(raw['offset'])
        norm = float(np.linalg.norm(normal))
        require(normal.shape == (2,) and np.isfinite(normal).all() and math.isfinite(offset) and norm > 0,
                'invalid observed normalized line')
        normal, offset = normal / norm, offset / norm
        endpoints = EDGES[edge]
        known = all(k in valid for k in endpoints)
        distances = q[list(endpoints)] @ normal - offset if known else None
        error = math.sqrt(float(np.mean(distances ** 2))) if known else None
        quality = UNKNOWN if error is None else GOOD if error <= LIMIT else BAD
        query_ids = [int(k) for k in raw['queries']]
        require(len(set(query_ids)) == len(query_ids) and all(k // 7 == edge for k in query_ids),
                'line support query identities differ')
        check = [c for c in checks if int(c['edge']) == edge]
        values.append(dict(edge=edge, endpoints=list(endpoints), normal=normal, offset=offset,
            original_normal_norm=norm, proxy_endpoint_xy=q[list(endpoints)], proxy_both_endpoints_known=known,
            proxy_reference_matched=row['evaluation_reference']['matched'],
            proxy_signed_endpoint_distances_px=distances, proxy_endpoint_RMS_px=error, quality=quality,
            actual_unused_factor=edge in pool, consumed_by_allowed_adopted_boundary=edge in consumed,
            source_present_but_excluded=edge not in set(pool) | consumed,
            source_exclusion_reasons=[c['reason'] for c in check if not c['accepted']],
            support_query_ids=query_ids, support_query_count=len(query_ids),
            actual_fit_line=edge in fit_edges, approved_final_line_inlier=new and edge in raw_inliers,
            raw_candidate_line_inlier=edge in raw_inliers,
            raw_candidate_scope='APPROVED_NEW' if new else 'REJECTED_OR_UNAVAILABLE_CANDIDATE',
            physical_ownership_certified=False))
        counters['source_lines_compared'] += 1
        counters['known_proxy_endpoint_pairs'] += int(known)
    if factor is not None:
        require(factor['unique_line_factors'] == len(pool), 'unique edge factor count differs')
        require(factor['line_support_queries'] == sum(v['support_query_count'] for v in values if v['actual_unused_factor']),
                'support-query count mislabeled as unique edge factor count')
    return values


def run(folder, output):
    guard_directories(folder, output)
    protocol_path = output / 'LINE_QUALITY_PROTOCOL.json'
    protocol = read(protocol_path)
    require(protocol['policy'] == POLICY, 'frozen arithmetic policy drift')
    before = inputs(folder)
    require(before == protocol['inputs'], 'frozen raw/code binding drift')
    for name in OUTPUTS:
        require(not (output / name).exists(), 'preserve actual attempt ' + name)
    counts = Counter(actual_arithmetic_runs=1)
    write(output / OUTPUTS[0], dict(protocol=binding(protocol_path), actual_arithmetic_runs=1,
        new_model_PnP_optimizer_ray_training_RGB_GT_pose_scoring_calls=0))
    start = time.monotonic()
    result, failure = {}, None
    try:
        inference, seal, score = (read(folder / n) for n in
                                 ('INFERENCE_RECEIPT.json', 'GEOMETRY_SEAL.json', 'SCORING_RECEIPT.json'))
        require(inference['complete'] and inference['cleanup_error'] is None and
                inference['actual_complete_frames'] == 245 and inference['method_rows'] == 980 and inference['fixed_rows'] == 490,
                'complete successful inference missing')
        require(seal['complete'] and seal['GT_read_allowed'] is False and seal['frames'] == 245 and
                seal['rows'] == 980 and seal['fixed_rows'] == 490, 'complete pre-GT seal missing')
        require(score['complete'] and score['frames'] == 245 and score['scored_method_rows'] == 980 and
                score['fixed_control_rows'] == 490 and score['GT_access_only_after_complete_seal'],
                'complete sealed posthoc scoring missing')
        require(read(folder / 'V4_CONTROL_PARITY.json')['passed'], 'unchanged point-only control parity failed')
        bind_matches(folder / 'PROTOCOL.json', seal['protocol'])
        bind_matches(folder / 'OBSERVATIONS.jsonl.gz', seal['observations'])
        bind_matches(folder / 'GEOMETRY_SEAL.json', score['geometry_seal'])
        bind_matches(folder / 'INFERENCE_RECEIPT.json', score['inference_receipt'])
        bind_matches(folder / 'PREDICTIONS.jsonl.gz', score['predictions'])
        bind_matches(folder / 'FIXED_PREDICTIONS.jsonl.gz', score['fixed_predictions'])
        observations = {}
        for row in rows(folder / 'OBSERVATIONS.jsonl.gz'):
            require(row['id'] not in observations and row['GT_input'] is False, 'duplicate/non-GT-free source observation')
            observations[row['id']] = row
            counts['observation_rows_read'] += 1
        require(len(observations) == 245, 'observation population differs')
        data = {m: {} for m in (*METHODS, 'BASE', 'N3_SUBPIX')}
        primary_lines = {}
        for name in ('FIXED_PREDICTIONS.jsonl.gz', 'PREDICTIONS.jsonl.gz'):
            expected = ('BASE', 'N3_SUBPIX') if name.startswith('FIXED') else METHODS
            for row in rows(folder / name):
                method, fid = row['method'], row['id']
                require(method in expected and fid in observations and fid not in data[method], 'scored method/ID differs')
                evidence = point_evidence(row) if method in METHODS else []
                if method == PRIMARY:
                    primary_lines[fid] = line_evidence(row, observations[fid], counts)
                data[method][fid] = dict(id=fid, method=method, session=row['session'],
                    output_status=row['output_status'], approved_NEW=approved(row) if method in METHODS else False,
                    fallback_used=row['fallback_used'], pose=saved_scores(row),
                    actual_pose=dict(cf_extents=(row.get('actual_pose') or {}).get('cf_extents')),
                    solver_state=(row.get('solver') or {}).get('state'),
                    solver_reason=(row.get('solver') or {}).get('reason'),
                    allowed_point_pool=(row.get('solver') or {}).get('used', []),
                    actual_point_fit_ids=(row.get('solver') or {}).get('fit_input_ids', []),
                    raw_final_point_inlier_ids=(row.get('solver') or {}).get('final_inliers', []),
                    adopted_boundary_ids=(row.get('observation_contract') or {}).get('hybrid_boundary_corner_ids', []),
                    point_evidence=evidence, raw_candidate_geometry=(row.get('solver') or {}).get('geometry'),
                    raw_point_line_geometry=(row.get('solver') or {}).get('point_line_geometry'),
                    approved_final_geometry=(row.get('solver') or {}).get('geometry') if method in METHODS and approved(row) else None,
                    proxy_phase_control=row['evaluation_reference']['phase_control'],
                    mask_audit=row.get('mask_audit'),
                    proxy_reference_matched=row['evaluation_reference']['matched'],
                    native_reference_valid_ids=row['evaluation_reference']['valid_native_ids'])
                counts['fixed_scored_rows_read' if method not in METHODS else 'new_scored_rows_read'] += 1
                counts['matched_reference_scored_rows'] += int(row['evaluation_reference']['matched'])
        require(all(set(byid) == set(observations) for byid in data.values()), 'complete common scored population differs')
        all_lines, all_points, joined = [], {m: [] for m in METHODS}, []
        grouped = defaultdict(list)
        for fid in sorted(observations):
            line = primary_lines[fid]
            methods = {}
            for method in METHODS:
                r = data[method][fid]
                r['dimension_vs_fixed_N3'] = extent_relation(r, data['N3_SUBPIX'][fid])
                r['vs_fixed_N3'] = paired(r['pose'], data['N3_SUBPIX'][fid]['pose'])
                r['vs_point_only_same_observations'] = paired(r['pose'], data[POINT_ONLY][fid]['pose'])
                methods[method] = r
                all_points[method].extend(r['point_evidence'])
            main = methods[PRIMARY]
            actual = [l for l in line if l['actual_unused_factor']]
            accepted = [l for l in line if l['approved_final_line_inlier']]
            group = (main['output_status'], main['dimension_vs_fixed_N3']['relation'],
                     len(actual), len(accepted), any(l['quality'] == BAD for l in accepted))
            grouped[group].append(main)
            joined.append(dict(id=fid, session=observations[fid]['session'],
                methods=methods, source_line_evidence=line,
                actual_unused_line_quality_counts=quality_counter(actual),
                approved_final_line_quality_counts=quality_counter(accepted),
                raw_candidate_line_quality_counts=quality_counter([l for l in line if l['raw_candidate_line_inlier']]),
                line_units_are_unique_edges=True, source_support_query_count=sum(l['support_query_count'] for l in line),
                physical_boundary_ownership_certified=False))
            all_lines.extend(line)
            counts['frames_joined'] += 1
        with (output / 'LINE_QUALITY_ROWS.jsonl.gz').open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                for row in joined:
                    stream.write((json.dumps(clean(row), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode())
        summary = dict(
            source_lines=quality_counter(all_lines),
            actual_unused_lines=quality_counter(all_lines, 'actual_unused_factor'),
            consumed_lines=quality_counter(all_lines, 'consumed_by_allowed_adopted_boundary'),
            source_present_but_excluded=quality_counter(all_lines, 'source_present_but_excluded'),
            actual_fit_lines=quality_counter(all_lines, 'actual_fit_line'),
            approved_NEW_final_line_inliers=quality_counter(all_lines, 'approved_final_line_inlier'),
            rejected_candidate_line_inliers=quality_counter([l for l in all_lines
                if l['raw_candidate_scope'] != 'APPROVED_NEW' and l['raw_candidate_line_inlier']]),
            approved_line_proxy_RMS_px=distribution([l['proxy_endpoint_RMS_px'] for l in all_lines
                if l['approved_final_line_inlier'] and l['proxy_endpoint_RMS_px'] is not None]),
            unused_line_proxy_RMS_px=distribution([l['proxy_endpoint_RMS_px'] for l in all_lines
                if l['actual_unused_factor'] and l['proxy_endpoint_RMS_px'] is not None]),
            point_quality_by_method={m: dict(
                allowed_pool=quality_counter(v, 'in_allowed_point_pool'),
                actual_fit_inputs=quality_counter(v, 'in_actual_fit_ids'),
                approved_NEW_final_inliers=quality_counter(v, 'approved_final_point_inlier'),
                rejected_candidate_inliers=quality_counter([p for p in v
                    if p['raw_candidate_scope'] != 'APPROVED_NEW' and p['raw_candidate_point_inlier']]))
                for m, v in all_points.items()},
            actual_returned_dimension_vs_fixed_N3={m: dict(Counter(r['dimension_vs_fixed_N3']['relation'] for r in
                (j['methods'][m] for j in joined))) for m in METHODS},
            source_support_query_total=sum(l['support_query_count'] for l in all_lines),
            actual_unused_support_query_total=sum(l['support_query_count'] for l in all_lines if l['actual_unused_factor']),
            physical_ownership_error_fraction_identified=False)
        group_rows = []
        for key, values in sorted(grouped.items()):
            group_rows.append(dict(output_status=key[0], dimension_vs_fixed_N3=key[1],
                unused_edge_count=key[2], approved_final_line_inlier_count=key[3],
                approved_line_proxy_mismatch_present=key[4], n=len(values), ids=[v['id'] for v in values],
                pose= {k: distribution([v['pose'][k] for v in values]) for k in ('translation_cm', 'rotation_deg', 'ADDsym_cm')},
                delta_vs_fixed_N3={k: distribution([v['vs_fixed_N3']['delta'][k] for v in values]) for k in ('translation_cm', 'rotation_deg', 'ADDsym_cm')},
                delta_vs_point_only={k: distribution([v['vs_point_only_same_observations']['delta'][k] for v in values]) for k in ('translation_cm', 'rotation_deg', 'ADDsym_cm')}))
        after = inputs(folder)
        require(before == after, 'immutable inputs changed during actual arithmetic')
        result = dict(schema='same_observation_proxy_line_quality_audit_v5', passed=True, complete=True,
            policy=POLICY, input_bindings=before, unchanged_inputs_after=True, counts=dict(counts),
            raw_rows=binding(output / 'LINE_QUALITY_ROWS.jsonl.gz'), summary=summary, groups=group_rows,
            actual_arithmetic_runs=1, new_model_PnP_optimizer_ray_training_RGB_GT_pose_scoring_calls=0,
            physical_GT_or_boundary_ownership_certification=False,
            rejected_candidate_inliers_are_not_accepted_output=True)
        report = ['# v5 실제 미사용 선 관측의 사후 proxy 일치 진단', '',
            '기하 봉인과 점수화가 완료된 동일245장의 저장 원행만 한 번 읽었다. 새 모델·PnP·optimizer·ray·학습·RGB·GT 자세 점수 계산은0이다.', '',
            '선 오차는 고정 N3 phase의 기존 geometric-proxy native endpoint 두 개에서 관측 무한직선까지의 법선 거리 RMS다. 저장된 reference matched=True이고 두 endpoint가 모두 알려진 경우에만 기존8px 기준으로 일치/불일치를 나눴다.',
            '이 값은 실제 물리 경계 소유권·가시선의 정확한 GT·선 방향의 support·실제3D경계 중 관측된 비율을 인증하지 않는다.', '',
            '| 집합 | proxy≤8px | proxy>8px | 미상 |', '|---|---:|---:|---:|']
        for name, label in [('source_lines', '동일 IMAGE_ROLE 관측선'), ('actual_unused_lines', '실제 미사용 선 factor pool'),
                            ('consumed_lines', '채택 코너가 소비한 선'), ('source_present_but_excluded', '추가 제외'),
                            ('approved_NEW_final_line_inliers', '승인 NEW의 최종 선 inlier'),
                            ('rejected_candidate_line_inliers', '기본 반환의 raw 후보 선 inlier')]:
            s = summary[name]
            report.append(f'| {label} | {s.get(GOOD, 0)} | {s.get(BAD, 0)} | {s.get(UNKNOWN, 0)} |')
        report += ['', '선 factor 단위는 고유 semantic edge1개다. 같은 선의 여러 query 수를 독립 선 inlier 수로 세지 않았다. 채택한 allowed boundary corner의 incident edge는 선 pool에서 제외한 것을 독립 확인했다.',
            '최종 선 inlier는 승인 NEW에서만 집계했다. rank 실패·다중해·기본 반환의 best-candidate 선/점 inlier는 별도 raw 진단이며 출력 성공으로 처리하지 않았다.', '',
            '모든 방법의 점 pool/실제 fit/승인 inlier의≤8/>8/unknown, 실제 반환 W/D분기, 사후 T/R/ADD 차이, 저장된 raw/승인 rank·layout은 ROWS와 CHECKS에 남겼다.',
            '불일치 선과 큰 자세 오차의 동반은 연관이다. 현 proxy로 실제 물리 경계 소유권 오류의 원인 비율을 단정하거나 설정을 다시 선택하지 않는다.', '',
            '[고정 프로토콜](LINE_QUALITY_PROTOCOL.json) · [245 원행](LINE_QUALITY_ROWS.jsonl.gz) · [검산](LINE_QUALITY_CHECKS.json)']
        with (output / 'LINE_QUALITY_REPORT_KO.md').open('x') as stream:
            stream.write('\n'.join(report) + '\n')
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        result = dict(schema='same_observation_proxy_line_quality_audit_v5', passed=False, complete=False,
            exception=failure, actual_arithmetic_runs=1, input_bindings=before, counts=dict(counts),
            new_model_PnP_optimizer_ray_training_RGB_GT_pose_scoring_calls=0,
            physical_GT_or_boundary_ownership_certification=False)
    result['elapsed_seconds'] = time.monotonic() - start
    write(output / 'LINE_QUALITY_CHECKS.json', result)
    print(json.dumps(dict(passed=result['passed'], counts=result['counts'], exception=failure)), flush=True)
    if failure:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, help='new audit output directory; default input directory')
    args = parser.parse_args()
    guard_directories(args.input.absolute(), (args.output or args.input).absolute())
    folder, output = args.input.resolve(), (args.output or args.input).resolve()
    (freeze if args.stage == 'freeze' else run)(folder, output)


if __name__ == '__main__':
    main()
