"""Audit whether saved point-only LOO packets externally validate unused lines.

This is a saved-input diagnostic. It runs no model, PnP, optimization, image
decode, ray, learning or GT scoring. A packet qualifies only when BOTH line
endpoint observations and H are absent from its scoring and actual fit/generator
IDs. Support-line self-consistency is reported separately. All qualifying
packets are retained; their residuals never select a new deployed policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_partial_line_independent_20261010_v5'
PRIVATE = Path('/tmp/pallet-partial-line-independent-private-20261010-v5')
PRIMARY = 'N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES'
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
INPUT_NAMES = ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json',
               'GEOMETRY_SEALED.jsonl.gz', 'OBSERVATIONS.jsonl.gz')
OUTPUT_NAMES = ('LINE_HELDOUT_STARTED.json', 'LINE_HELDOUT_ROWS.jsonl.gz', 'LINE_HELDOUT_CHECKS.json')
PROTOCOL_NAME = 'LINE_HELDOUT_PROTOCOL.json'
NO_PRIOR_FLAGS = ('prior_used', 'initial_pose_used', 'initial_projection_used',
                  'initial_dimension_prior_used', 'known_dimension_constraint_used',
                  'excluded_image_coordinates_used_for_scoring',
                  'excluded_image_coordinates_used_for_equivalence')
POLICY = dict(
    schema='saved_two_endpoint_heldout_line_validation_policy_v1',
    population='all actual unused v5 line edges in all fixed245 frames',
    packet_population='every stored observation_contract.cornerwise_records entry for every unused edge',
    qualification='both line endpoint raw IDs AND H absent from used/scoring, selected generator/fit, and every stored scored/refit candidate generator/fit',
    used_witness='used explicitly represents the scoring point pool in the bound observation-only v4 PoseBank; missing witness is UNVERIFIED',
    score_witness='excluded_image_coordinates_used_for_scoring=False; scored candidate residual vector length equals used length',
    all_candidate_scope='all stored scored numerical branches and refits; unused global lazy-bank generators are not a witness of selected/scored dependence',
    pose='available NEW_POSE with no numeric initial pose/projection/dimension prior, no known dimension constraint, no unresolved ambiguity',
    proposal='no lines enter these v4 LOO fits; unchanged Base-derived proposals and initial-N3-dependent H remain conditional dependencies',
    residual='RMS of projected registry edge endpoint normal distances to the normalized saved observed infinite line',
    geometry_residual_cap_px=8.0,
    categories=['GEOMETRY_SUPPORTED', 'MISMATCH', 'UNVERIFIED'],
    packet_selection='none: retain all qualified packets irrespective of source corner acceptance or residual',
    line_aggregation='all qualified packets agree -> their category; no qualified or conflicting packet categories -> UNVERIFIED with explicit reason',
    support_self_consistency='saved support normal residual<=saved query radius+1e-9; separate from other-point geometric evidence',
    unverified_is_physical_no_match=False,
    physical_boundary_ownership_certified=False,
    fully_statistically_independent=False,
    projection_arithmetic_atol_px=1e-7,
    projection_arithmetic_rtol=1e-12,
    settings_selected_from_real_scores=False,
    new_model_PnP_optimizer_ray_training_RGB_GT_scoring_calls=0)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def binding(path):
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    name = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name
    return dict(path=name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def inputs(folder):
    value = {name: binding(folder / name) for name in INPUT_NAMES}
    value['audit_code'] = binding(Path(__file__))
    v4 = REPO / 'scripts/research/pallet_cornerwise_independent_20261010_v4'
    value['bound_point_only_LOO_solver_code'] = binding(v4 / 'pose.py')
    value['bound_point_only_selection_code'] = binding(v4 / 'selection.py')
    return value


def bound(path, expected):
    actual = binding(path)
    require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'binding differs: ' + path.name)


def guard(folder, output):
    for path in (folder, output):
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink input/output ancestry')
    destination = output.resolve()
    require(destination == DOC.resolve() or destination.is_relative_to(PRIVATE.resolve()),
            'output must be exact new v5 DOC or its private subtree')
    return folder.resolve(), destination


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def freeze(folder, output):
    folder, output = guard(folder, output)
    output.mkdir(parents=True, exist_ok=True)
    for name in (PROTOCOL_NAME, *OUTPUT_NAMES):
        require(not (output / name).exists() and not (output / name).is_symlink(), 'preserve audit attempt: ' + name)
    write(output / PROTOCOL_NAME, dict(schema='frozen_saved_line_heldout_audit_v1', policy=POLICY,
          inputs=inputs(folder), actual_arithmetic_runs=0, frozen_before_actual_arithmetic=True))
    print('LINE_HELDOUT_FROZEN', binding(output / PROTOCOL_NAME)['sha256'], flush=True)


def ids(value):
    if not isinstance(value, list) or any(type(k) is not int or not 0 <= k < 8 for k in value):
        return None
    if len(set(value)) != len(value):
        return None
    return set(value)


def finite_matrix(value, shape):
    if not isinstance(value, list) or len(value) != shape[0]:
        return False
    if len(shape) == 1:
        return all(type(v) in (int, float) and math.isfinite(v) for v in value)
    return all(finite_matrix(row, shape[1:]) for row in value)


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def project(solver, K):
    require(finite_matrix(K, (3, 3)) and finite_matrix(solver.get('R_cf'), (3, 3)) and
            finite_matrix(solver.get('centroid'), (3,)) and finite_matrix(solver.get('cf_extents'), (3,)),
            'invalid qualified numeric pose/camera witness')
    a, b, c = (v/2 for v in solver['cf_extents'])
    require(min(a, b, c) > 0, 'invalid qualified model dimensions')
    model = ((-a,-b,-c),(a,-b,-c),(a,b,-c),(-a,b,-c),
             (-a,-b,c),(a,-b,c),(a,b,c),(-a,b,c))
    result = []
    for point in model:
        camera = [dot(axis, point)+shift for axis, shift in zip(solver['R_cf'], solver['centroid'])]
        uvw = [dot(axis, camera) for axis in K]
        require(all(math.isfinite(v) for v in uvw) and uvw[2] > 0, 'invalid qualified model depth')
        result.append([uvw[0]/uvw[2], uvw[1]/uvw[2]])
    return result


def observed_line(raw):
    edge = raw['edge']
    require(type(edge) is int and 0 <= edge < 12 and raw['endpoints'] == list(EDGES[edge]),
            'semantic observed line endpoints differ')
    require(finite_matrix(raw['normal'], (2,)) and type(raw['offset']) in (int, float) and
            math.isfinite(raw['offset']), 'invalid observed line coefficients')
    scale = math.hypot(*raw['normal'])
    require(scale > 0, 'zero observed line normal')
    normal, offset = [n/scale for n in raw['normal']], raw['offset']/scale
    support, radii = raw['support_points'], raw['query_radii_px']
    require(finite_matrix(support, (len(support), 2)) and len(radii) == len(support) and
            all(type(r) in (int, float) and math.isfinite(r) and r >= 0 for r in radii),
            'invalid saved support/radius arrays')
    errors = [abs(dot(p, normal)-offset) for p in support]
    query = raw['queries']
    require(len(query) == len(support) and len(query) >= 3 and len(set(query)) == len(query) and
            all(type(q) is int and 0 <= q < 84 and q//7 == edge for q in query), 'invalid saved support identities')
    return normal, offset, dict(support_query_ids=query, support_query_count=len(query),
        support_normal_residuals_px=errors, query_radii_px=radii,
        final_support_self_consistent=all(e <= r+1e-9 for e, r in zip(errors, radii)),
        support_is_independent_other_point_evidence=False)


def packet(record, H, endpoints, normal, offset, K):
    info = dict(source_corner_id=record.get('id'), source_corner_originally_accepted=record.get('accepted'),
                heldout_pose_calls=record.get('heldout_pose_calls'), qualified=False,
                category='UNVERIFIED', reason=None, endpoint_RMS_px=None,
                projected_endpoints=None, exclusion_witnesses={}, witness_failures=[])
    fail = info['witness_failures']
    if not record.get('heldout_pose_calls') or not isinstance(record.get('loo_solver'), dict):
        info['reason'] = 'NO_STORED_LOO_POSE_PACKET'
        return info
    solver = record['loo_solver']
    info.update(solver_state=solver.get('state'), solver_reason=solver.get('reason'),
                solver_available=solver.get('available'))
    blocked = set(endpoints) | set(H)
    for key in ('used', 'generator_ids', 'fit_input_ids', 'final_inliers', 'excluded', 'hidden', 'temporary_excluded'):
        values = ids(solver.get(key))
        info['exclusion_witnesses'][key] = solver.get(key)
        if values is None:
            fail.append('MISSING_OR_INVALID_' + key.upper() + '_WITNESS')
        elif key in ('used', 'generator_ids', 'fit_input_ids', 'final_inliers') and values & blocked:
            fail.append(key.upper() + '_USES_LINE_ENDPOINT_OR_H')
    used, excluded = ids(solver.get('used')), ids(solver.get('excluded'))
    if excluded is not None and not set(H) <= excluded:
        fail.append('H_NOT_EXPLICITLY_EXCLUDED')
    if ids(solver.get('hidden')) != set(H):
        fail.append('ACTUAL_H_METADATA_DIFFERS')
    heldout = record.get('heldout_corner_id')
    if type(heldout) is not int or not 0 <= heldout < 8 or ids(solver.get('temporary_excluded')) != {heldout}:
        fail.append('TEMPORARY_HELDOUT_METADATA_MISSING_OR_DIFFERS')
    elif excluded != set(H) | {heldout}:
        fail.append('EXCLUDED_H_UNION_HELDOUT_METADATA_DIFFERS')
    if heldout != record.get('id'):
        fail.append('HELDOUT_CORNER_RECORD_ID_DIFFERS')
    for key in NO_PRIOR_FLAGS:
        if solver.get(key) is not False:
            fail.append('MISSING_OR_NONFALSE_' + key.upper())
    if solver.get('solver') != 'OBSERVATION_ONLY_FINITE_SUBSET_ROBUST':
        fail.append('NO_BOUND_POINT_ONLY_ROBUST_SOLVER_WITNESS')
    candidate_list = solver.get('all_candidate_solutions')
    if not isinstance(candidate_list, list):
        fail.append('MISSING_SCORED_CANDIDATE_POOL_WITNESS')
    else:
        for i, c in enumerate(candidate_list):
            for key in ('generator_ids', 'actual_fit_input_ids', 'inlier_ids'):
                values = ids(c.get(key))
                if values is None or used is None or not values <= used or values & blocked:
                    fail.append('SCORED_CANDIDATE_%d_%s_NOT_ENDPOINT_AND_H_FREE' % (i, key.upper()))
            residual = c.get('residuals_used_px')
            if not isinstance(residual, list) or used is None or len(residual) != len(used):
                fail.append('SCORED_CANDIDATE_%d_SCORING_POOL_WITNESS_MISSING' % i)
    info['stored_scored_candidate_count'] = len(candidate_list) if isinstance(candidate_list, list) else None
    if solver.get('available') is not True or solver.get('state') != 'NEW_POSE':
        fail.append('LOO_NEW_POSE_UNAVAILABLE')
    if solver.get('unresolved_ambiguity') is not False:
        fail.append('LOO_AMBIGUITY_WITNESS_MISSING_OR_TRUE')
    if not finite_matrix(solver.get('projected'), (8, 2)):
        fail.append('LOO_PROJECTION_UNAVAILABLE')
    if solver.get('available') is True and not candidate_list:
        fail.append('AVAILABLE_POSE_WITHOUT_SCORED_CANDIDATE_WITNESSES')
    if fail:
        info['reason'] = 'EXTERNAL_VALIDATION_WITNESS_NOT_QUALIFIED'
        return info
    computed = project(solver, K)
    actual = solver['projected']
    difference = max(abs(a-b) for qa, qb in zip(actual, computed) for a,b in zip(qa,qb))
    require(all(abs(a-b) <= 1e-7+1e-12*abs(b) for qa,qb in zip(actual,computed) for a,b in zip(qa,qb)),
            'qualified saved projection disagrees with R/t/K/model arithmetic')
    coordinates = [actual[k] for k in endpoints]
    distances = [dot(q, normal)-offset for q in coordinates]
    error = math.sqrt(math.fsum(d*d for d in distances)/2)
    info.update(qualified=True, category='GEOMETRY_SUPPORTED' if error <= 8 else 'MISMATCH',
        reason='BOTH_ENDPOINT_AND_H_FREE_OTHER_POINT_POSE', endpoint_RMS_px=error,
        signed_endpoint_normal_distances_px=distances, projected_endpoints=coordinates,
        projected_model_arithmetic_max_abs_difference_px=difference,
        actual_pose=dict(R_cf=solver['R_cf'], centroid=solver['centroid'], cf_extents=solver['cf_extents']),
        conditional_H_and_Base_proposal_dependence_retained=True,
        fully_statistically_independent=False, physical_boundary_ownership_certified=False)
    return info


def run(folder, output):
    folder, output = guard(folder, output)
    protocol_path = output / PROTOCOL_NAME
    protocol = read(protocol_path)
    require(protocol['policy'] == POLICY and protocol['inputs'] == inputs(folder), 'frozen audit code/input/policy drift')
    for name in OUTPUT_NAMES:
        require(not (output/name).exists() and not (output/name).is_symlink(), 'preserve audit attempt: '+name)
    write(output/OUTPUT_NAMES[0], dict(protocol=binding(protocol_path), actual_arithmetic_runs=1,
          no_automatic_retry=True, new_model_PnP_optimizer_ray_training_RGB_GT_scoring_calls=0))
    counts, packet_categories, line_categories, reasons = Counter(), Counter(), Counter(), Counter()
    start, failure, audit_rows = time.monotonic(), None, []
    try:
        seal, inference, original = (read(folder/n) for n in ('GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json', 'PROTOCOL.json'))
        require(seal['complete'] and seal['GT_read_allowed'] is False and seal['frames'] == 245 and seal['rows'] == 980,
                'complete fixed geometry seal missing')
        require(inference['complete'] and inference['cleanup_error'] is None and inference['actual_complete_frames'] == 245,
                'complete inference receipt missing')
        require(original['primary'] == PRIMARY and original['frames'] == 245, 'fixed primary/cohort differs')
        for name, key in (('PROTOCOL.json','protocol'), ('GEOMETRY_SEALED.jsonl.gz','geometry'), ('OBSERVATIONS.jsonl.gz','observations')):
            bound(folder/name, seal[key])
        obs = {}
        for row in rows(folder/'OBSERVATIONS.jsonl.gz'):
            require(row['id'] not in obs and row['GT_input'] is False, 'duplicate/non-GT-free observation')
            obs[row['id']] = row
        geometry = list(rows(folder/'GEOMETRY_SEALED.jsonl.gz'))
        require(len(obs) == 245 and len(geometry) == 980 and len({(r['method'], r['id']) for r in geometry}) == 980,
                'complete saved population differs')
        require(Counter(r['method'] for r in geometry) == {m:245 for m in original['methods']}, 'method counts differ')
        primary = [r for r in geometry if r['method'] == PRIMARY]
        require(len(primary) == 245 and {r['id'] for r in primary} == set(obs), 'primary observation IDs differ')
        for row in primary:
            fid, H = row['id'], row['hidden_initial']
            require(ids(H) is not None, 'invalid saved H')
            records = row['observation_contract'].get('cornerwise_records')
            require(isinstance(records, list), 'missing saved cornerwise record list')
            require(len({r['id'] for r in records}) == len(records), 'duplicate saved LOO corner records')
            source = {line['edge']: line for line in obs[fid]['lines']}
            require(len(source) == len(obs[fid]['lines']), 'duplicate observed semantic edge')
            unused = row['solver']['line_edges']
            require(len(set(unused)) == len(unused) and set(unused) <= set(source), 'actual unused line pool differs')
            require(not set(unused) & set(row['solver']['consumed_edges']), 'consumed edge reappears in unused line pool')
            for edge in unused:
                normal, offset, support = observed_line(source[edge])
                require(support['final_support_self_consistent'], 'actual unused line lacks final support self-consistency')
                packets = [packet(r, H, EDGES[edge], normal, offset, row['K']) for r in records]
                qualified = [p for p in packets if p['qualified']]
                categories = {p['category'] for p in qualified}
                category = next(iter(categories)) if len(categories) == 1 else 'UNVERIFIED'
                reason = ('ALL_QUALIFIED_PACKETS_AGREE' if len(categories) == 1 else
                          'CONFLICTING_QUALIFIED_PACKET_RESULTS' if qualified else 'NO_QUALIFIED_OTHER_POINT_PACKET')
                audit_rows.append(dict(id=fid, session=row['session'], edge=edge, endpoints=list(EDGES[edge]),
                    initial_H=H, actual_unused_factor=True, support_self_consistency=support,
                    normal=normal, offset=offset, packet_count=len(packets), qualified_packet_count=len(qualified),
                    all_saved_LOO_packets=packets, category=category, reason=reason,
                    physical_no_match_predicted=False, deployed_policy_changed=False,
                    fully_statistically_independent=False, physical_boundary_ownership_certified=False))
                counts['actual_unused_edges'] += 1
                counts['all_saved_packet_edge_pairs'] += len(packets)
                counts['qualified_packet_edge_pairs'] += len(qualified)
                packet_categories.update(p['category'] for p in packets)
                line_categories[category] += 1
                reasons[reason] += 1
            counts['frames'] += 1
            counts['stored_cornerwise_records'] += len(records)
            counts['frames_with_unused_lines'] += int(bool(unused))
        counts['frames_without_unused_lines'] = 245-counts['frames_with_unused_lines']
        require(inputs(folder) == protocol['inputs'], 'immutable saved input/code changed during arithmetic')
        with (output/OUTPUT_NAMES[1]).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                for row in audit_rows:
                    stream.write((json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode())
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    result = dict(schema='saved_two_endpoint_heldout_line_audit_v1', complete=failure is None, passed=failure is None,
        exception=failure, protocol=binding(protocol_path), input_bindings=protocol['inputs'], policy=POLICY,
        actual_arithmetic_runs=1, actual_counts=dict(counts), packet_categories=dict(packet_categories),
        line_categories=dict(line_categories), line_aggregation_reasons=dict(reasons),
        no_qualified_packet_is_not_physical_no_match=True, deployed_policy_changed=False,
        new_model_PnP_optimizer_ray_training_RGB_GT_scoring_calls=0, elapsed_seconds=time.monotonic()-start,
        raw_rows=binding(output/OUTPUT_NAMES[1]) if (output/OUTPUT_NAMES[1]).exists() else None,
        limits=['Both endpoint raw observations are absent from the witnessed scoring and actual fit/generator pools; this is conditional numerical other-point evidence.',
                'H remains an initial-N3-derived condition and proposals remain Base-derived; complete statistical independence is not established.',
                'Unused lazy global-bank hypotheses can include withheld coordinates; every stored scored candidate and refit must be endpoint/H-free.',
                'No new line-heldout pose is computed when saved packets are missing, unavailable or unqualified.',
                'Support self-consistency and endpoint infinite-line agreement certify neither actual physical ownership nor along-edge support.',
                'All qualified packets are recorded. Conflicting qualified packets remain UNVERIFIED; no best-residual or GT-driven selection is made.'])
    write(output/OUTPUT_NAMES[2], result)
    print(json.dumps(dict(passed=result['passed'], actual_counts=dict(counts), packet_categories=dict(packet_categories),
                         line_categories=dict(line_categories), exception=failure)), flush=True)
    if failure:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, default=DOC)
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else run)(args.input.absolute(), args.output.absolute())


if __name__ == '__main__':
    main()
