"""Frozen posthoc proxy-quality audit of the v6 two-endpoint validation gate.

This reads completed saved rows only. It imports no model, solver, numerical
array library, image reader, GT loader, PnP, optimizer or ray engine. Proxy
agreement never selects a deployed observation, pose, threshold or setting.
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

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_partial_line_heldout_20261010_v6'
PRIVATE = Path('/tmp/pallet-partial-line-heldout-private-20261010-v6')
PRIMARY = 'N3_INDEPENDENT_CORNERWISE_ROLE_ENDPOINT_VALIDATED_LINES'
CONTROL = 'N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES'
METHODS = (PRIMARY, 'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK', CONTROL)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
LIMIT = 8.0
GOOD, BAD, UNKNOWN = 'PROXY_WITHIN8PX', 'PROXY_OVER8PX', 'UNKNOWN_PROXY'
DECISIONS = ('SUPPORTED_RETAIN', 'CONTRADICTED_REJECT', 'UNVERIFIED_RETAIN')
FILES = ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json',
         'SCORING_RECEIPT.json', 'V5_CONTROL_PARITY.json', 'METRICS.json', 'VALIDATION_CHECKS.json',
         'OBSERVATIONS.jsonl.gz', 'PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz')
PROTOCOL = 'GATE_QUALITY_PROTOCOL.json'
OUTPUTS = ('GATE_QUALITY_STARTED.json', 'GATE_QUALITY_ROWS.jsonl.gz', 'GATE_QUALITY_CHECKS.json')
POLICY = dict(
    schema='posthoc_endpoint_gate_proxy_quality_policy_v6',
    population='all fixed245 easy153+medium92 frames, every original source line and every endpoint validator record',
    frozen_after_accuracy_scoring_statistics=True,
    existing_accuracy_outcome_known_before_this_diagnostic=True,
    reference='saved matched evaluation_reference.native_points_px/valid_native_ids in fixed N3 phase; existing GEOMETRIC_PROXY',
    line_quality='RMS normal distances of both known proxy native endpoints to the normalized observed infinite line',
    point_quality='Euclidean saved native-N3 correspondence error against the same known proxy native corner',
    validation_projection_quality='RMS Euclidean error of the two model-projected endpoints against the known proxy endpoints; distinct from normal-distance line RMS',
    agreement_threshold_px=LIMIT,
    unknown='unmatched proxy or either endpoint missing/invalid/nonfinite; unavailable validator remains UNVERIFIED, never physical no_match',
    accepted_line_inliers='NEW_POSE operational outputs only; rejected/fallback candidate inliers are labeled separately',
    confusion='gate decision crossed with proxy line agreement, never with physical boundary ownership truth',
    extent_relation='returned cf_extents vs fixed N3 and unchanged v5 control; same or width/depth swapped at fixed1e-9; no correctness claim',
    layout='stored native-validator image/object/Jacobian local rank labels only; no new SVD/Jacobian',
    pose_outcomes='saved operational T/R/ADDsym deltas, strict both-better/both-worse/mixed-or-unchanged',
    source_observations_changed=False, deployed_policy_changed=False,
    new_models_PnP_optimizers_rays_RGB_training_GT_pose_scoring_calls=0,
    real_physical_ownership_certified=False,
    fully_statistically_independent=False,
    settings_selected_from_proxy_scores=False)


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
    absolute = path.resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    name = str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name
    return dict(path=name, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def inputs(folder):
    return dict(**{name: binding(folder / name) for name in FILES}, audit_code=binding(Path(__file__)))


def bound(path, saved):
    actual = binding(path)
    require(all(actual[k] == saved[k] for k in ('sha256', 'bytes')), 'frozen binding differs: ' + path.name)


def guard(folder, output):
    for path in (folder.absolute(), output.absolute()):
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink input/output ancestry')
    destination = output.resolve()
    require(destination == DOC.resolve() or destination.is_relative_to(PRIVATE.resolve()),
            'output must be exact new v6 DOC or the new v6 private subtree')
    for name in (PROTOCOL, *OUTPUTS):
        require(not (destination / name).is_symlink(), 'symlink audit artifact')
    return folder.resolve(), destination


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def freeze(folder, output):
    folder, output = guard(folder, output)
    require(all(not (output / name).exists() for name in (PROTOCOL, *OUTPUTS)), 'preserve prior audit attempt')
    output.mkdir(parents=True, exist_ok=True)
    write(output / PROTOCOL, dict(schema='frozen_completed_v6_gate_quality_audit_v1', policy=POLICY,
          frozen_before_actual_saved_arithmetic=True, inputs=inputs(folder), actual_arithmetic_runs=0))
    print('GATE_QUALITY_FROZEN', binding(output / PROTOCOL)['sha256'], flush=True)


def finite_point(value):
    return isinstance(value, list) and len(value) == 2 and all(
        type(v) in (int, float) and math.isfinite(v) for v in value)


def valid_point(value):
    return finite_point(value) and value != [-1, -1]


def ids(value, label):
    require(isinstance(value, list) and all(type(k) is int and 0 <= k < 8 for k in value) and
            len(set(value)) == len(value), 'invalid/duplicate IDs: ' + label)
    return set(value)


def reference(row):
    saved = row['evaluation_reference']
    require(saved['phase_control'] == 'N3_SUBPIX' and type(saved['matched']) is bool,
            'fixed matched N3-phase reference contract missing')
    q = saved['native_points_px']
    require(isinstance(q, list) and len(q) == 8, 'proxy must have eight native corners')
    valid = ids(saved['valid_native_ids'], 'proxy') if saved['matched'] else set()
    require(all(finite_point(q[k]) for k in valid), 'declared proxy endpoint is nonfinite')
    return q, valid


def normalize(raw):
    edge = raw['edge']
    require(type(edge) is int and 0 <= edge < 12 and raw['endpoints'] == list(EDGES[edge]),
            'source edge endpoint contract differs')
    require(finite_point(raw['normal']) and type(raw['offset']) in (int, float) and
            math.isfinite(raw['offset']), 'nonfinite observed line')
    norm = math.hypot(*raw['normal'])
    require(norm > 0, 'zero observed normal')
    return [v / norm for v in raw['normal']], raw['offset'] / norm


def dot(p, normal):
    return math.fsum(v * n for v, n in zip(p, normal))


def rms(distances):
    return math.sqrt(math.fsum(d * d for d in distances) / len(distances))


def quality(error):
    return UNKNOWN if error is None else GOOD if error <= LIMIT else BAD


def quantile(values, p):
    a = sorted(values)
    if not a:
        return None
    index = (len(a) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return a[lo] + (a[hi] - a[lo]) * (index - lo)


def distribution(values):
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'nonfinite diagnostic arithmetic')
    n = len(values)
    mean = math.fsum(values) / n if n else None
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else None
    return dict(n=n, mean=mean, sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=quantile(values, .5), P90=quantile(values, .9), max=max(values) if n else None)


def saved_scores(row):
    pose = row['pose']
    if not pose['available']:
        return None
    result = dict(translation_cm=pose['translation_cm'], rotation_deg=pose['rotation_deg'],
                  ADDsym_cm=pose['ADDsym_m'] * 100)
    require(all(type(v) in (int, float) and math.isfinite(v) for v in result.values()), 'nonfinite saved metric')
    return result


def paired(a, b):
    if a is None or b is None:
        return dict(delta=None, pose_T_R_outcome='POSE_UNAVAILABLE')
    delta = {key: a[key] - b[key] for key in a}
    T, R = delta['translation_cm'], delta['rotation_deg']
    outcome = 'BOTH_BETTER' if T < 0 and R < 0 else 'BOTH_WORSE' if T > 0 and R > 0 else 'MIXED_OR_UNCHANGED'
    return dict(delta=delta, pose_T_R_outcome=outcome)


def extents_relation(a, b):
    if a is None or b is None:
        result = 'UNKNOWN'
    else:
        require(len(a) == len(b) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in a + b),
                'nonfinite returned model extents')
        close = lambda x, y: all(abs(u - v) <= 1e-9 for u, v in zip(x, y))
        result = 'SAME' if close(a, b) else 'WD_SWITCHED' if close(a, [b[2], b[1], b[0]]) else 'OTHER_RETURNED_EXTENTS'
    return dict(relation=result, candidate_cf_extents=a, comparator_cf_extents=b,
                physical_dimension_or_phase_correctness_certified=False)


def layout(solved):
    geometry = solved.get('geometry') or {}
    values = {}
    for key in ('image', 'object', 'jacobian'):
        part = geometry.get(key)
        if isinstance(part, dict):
            values[key] = {k: part[k] for k in ('numerical_rank', 'condition_number', 'singular_values') if k in part}
        else:
            values[key] = None
    return dict(stored_local_layout_labels=values, recomputed=False, global_unique_pose_certified=False)


def point_quality(points, proxy, valid):
    output = []
    for k, p in enumerate(points[:8]):
        error = math.hypot(p[0] - proxy[k][0], p[1] - proxy[k][1]) if k in valid and valid_point(p) else None
        output.append(dict(id=k, proxy_error_px=error, quality=quality(error)))
    return output


def point_pool(values, pool):
    selected = [values[k] for k in sorted(pool)]
    return dict(ids=sorted(pool), n=len(pool),
                correct_ids=[v['id'] for v in selected if v['quality'] == GOOD],
                incorrect_ids=[v['id'] for v in selected if v['quality'] == BAD],
                unknown_ids=[v['id'] for v in selected if v['quality'] == UNKNOWN],
                quality_counts=dict(Counter(v['quality'] for v in selected)))


def validator(record, row, line, proxy, valid):
    solved = record['solved']
    H, endpoints = set(row['hidden_initial']), set(EDGES[line['edge']])
    require(record['decision'] in DECISIONS and record['attempted_pose_calls'] == 1,
            'explicit endpoint validator decision/call witness differs')
    temporary = set(record['temporary_excluded']) | endpoints
    blocked = H | temporary
    require(set(record['validation_excluded']) == blocked and
            set(record['validator_temporary_excluded']) == temporary and
            set(solved['excluded']) == blocked and set(solved['hidden']) == H,
            'explicit H/two-endpoint validator mask differs')
    native = row['solver']['endpoint_validation']['native_points']
    values = point_quality(native, proxy, valid)
    pools = {}
    used = ids(solved.get('used', []), 'validator used')
    for name, key in (('used', 'used'), ('selected_fit', 'fit_input_ids'), ('raw_final_inliers', 'final_inliers')):
        present = key in solved
        pool = ids(solved[key], 'validator ' + name) if present else set()
        require(not pool & blocked and (name == 'used' or pool <= used), 'blocked/non-scoring validator point')
        pools[name] = point_pool(values, pool) if present else dict(witness_missing=True)
    projection = solved.get('projected')
    available = solved.get('available') is True
    endpoints_list = list(EDGES[line['edge']])
    normal, offset = normalize(line)
    both_known = all(k in valid for k in endpoints_list)
    projected_available = (available and isinstance(projection, list) and len(projection) == 8 and
                           all(finite_point(projection[k]) for k in endpoints_list))
    predicted_signed = [dot(projection[k], normal) - offset for k in endpoints_list] if projected_available else None
    predicted_RMS = rms(predicted_signed) if predicted_signed is not None else None
    projection_errors = [math.hypot(projection[k][0] - proxy[k][0], projection[k][1] - proxy[k][1])
                         for k in endpoints_list] if projected_available and both_known else None
    projection_RMS = rms(projection_errors) if projection_errors is not None else None
    checkable = record['decision'] != 'UNVERIFIED_RETAIN'
    if checkable:
        require(projected_available and predicted_RMS is not None, 'checkable validator has no actual projected endpoints')
        require(abs(predicted_RMS - record['endpoint_RMS_px']) <= 1e-7 + 1e-12 * abs(predicted_RMS),
                'saved validator normal-distance RMS differs')
        expected = 'SUPPORTED_RETAIN' if predicted_RMS <= LIMIT else 'CONTRADICTED_REJECT'
        require(record['decision'] == expected, 'fixed8 endpoint decision differs')
    return dict(decision=record['decision'], reason=record['reason'], solver_available=available,
        solver_state=solved.get('state'), solver_reason=solved.get('reason'), checkable=checkable,
        observed_line_used_to_choose_pose=False, native_input_hash=record['native_input_hash'],
        actual_H=sorted(H), explicit_endpoint_ids=endpoints_list, excluded_ids=sorted(blocked),
        native_point_evidence=values, native_point_pools=pools, local_layout=layout(solved),
        projected_endpoints_px=[projection[k] for k in endpoints_list] if projected_available else None,
        observed_line_signed_endpoint_distances_px=predicted_signed,
        predicted_observed_line_normal_RMS_px=predicted_RMS,
        proxy_endpoint_Euclidean_errors_px=projection_errors,
        proxy_endpoint_projection_Euclidean_RMS_px=projection_RMS,
        proxy_projection_quality=quality(projection_RMS),
        actual_cf_extents=solved.get('cf_extents') if available else None,
        conditional_H_and_Base_proposal_dependence_remain=True,
        physical_edge_ownership_certified=False)


def primary(row, observation):
    solved = row['solver']
    validation = solved['endpoint_validation']
    proxy, valid = reference(row)
    native = observation['native_N3_points']
    require(native == validation['native_points'], 'validator native RGB coordinate identity differs')
    new = row['new_pose_estimated']
    require(new == (row['output_status'] == 'NEW_POSE') == solved['available'], 'primary NEW status differs')
    require(not new or row['pose_available'] and solved['state'] == 'NEW_POSE', 'accepted primary pose missing')
    require(row['reprojections_reused_as_observations'] is False, 'projection fed back into fit')
    U = ids(solved['used'], 'final used')
    H = ids(row['hidden_initial'], 'actual H')
    require(not U & H, 'H raw coordinate entered final scoring')
    adopted = set(row['observation_contract']['hybrid_boundary_corner_ids']) & U
    corners = {c['id']: c for c in observation['corners']}
    require(len(corners) == len(observation['corners']), 'duplicate source corner identity')
    consumed = set()
    for k in adopted:
        require(corners[k]['xy'] == row['input_points'][k] and len(set(corners[k]['edges'])) == 2 and
                all(k in EDGES[e] for e in corners[k]['edges']), 'actual adopted corner/incident graph differs')
        consumed.update(corners[k]['edges'])
    require(consumed == set(validation['consumed_edges']) == set(solved['consumed_edges']), 'consumed graph differs')
    original, retained, rejected = (set(validation[k]) for k in ('original_unused_edges', 'retained_edges', 'rejected_edges'))
    require(retained == original - rejected and not consumed & original and
            set(solved['line_edges']) == retained, 'original/filtered factor pool differs')
    records = {r['edge']: r for r in validation['records']}
    require(len(records) == len(validation['records']) and set(records) == original,
            'not exactly one endpoint validator record per original unused edge')
    unverified = {e for e, r in records.items() if r['decision'] == 'UNVERIFIED_RETAIN'}
    require(not unverified & rejected and rejected == {e for e, r in records.items() if r['decision'] == 'CONTRADICTED_REJECT'},
            'unverified retained/rejected decision mismatch')
    raw_inliers = set(solved.get('final_line_inliers', []))
    require(raw_inliers <= retained, 'raw candidate line inlier outside retained pool')
    source = {}
    for line in observation['lines']:
        edge = line['edge']
        if edge in source:
            require(source[edge] == line, 'inconsistent duplicate source edge')
        source[edge] = line
    require(original | consumed <= set(source), 'pool/consumed line absent from original observations')
    values = []
    for edge, raw in sorted(source.items()):
        normal, offset = normalize(raw)
        endpoints = list(EDGES[edge])
        known = all(k in valid for k in endpoints)
        signed = [dot(proxy[k], normal) - offset for k in endpoints] if known else None
        error = rms(signed) if known else None
        query = raw['queries']
        require(len(set(query)) >= 3 and all(type(q) is int and 0 <= q < 84 and q // 7 == edge for q in query),
                'source support identity differs')
        support, radii = raw['support_points'], raw['query_radii_px']
        require(len(support) == len(query) == len(radii), 'source support shape differs')
        support_errors = [abs(dot(p, normal) - offset) for p in support]
        self_consistent = all(math.isfinite(r) and r >= 0 and e <= r + 1e-9 for e, r in zip(support_errors, radii))
        require(edge not in original | consumed or self_consistent, 'accepted source support inconsistent')
        gate = validator(records[edge], row, raw, proxy, valid) if edge in records else None
        values.append(dict(edge=edge, endpoints=endpoints, normalized_normal=normal, normalized_offset=offset,
            source_line=True, original_unused_eligible=edge in original,
            consumed_by_actual_adopted_boundary=edge in consumed,
            retained_factor=edge in retained, contradicted_rejected=edge in rejected,
            unverified_retained=edge in unverified,
            approved_NEW_final_line_inlier=new and edge in raw_inliers,
            raw_final_candidate_line_inlier=edge in raw_inliers,
            raw_final_candidate_scope='APPROVED_NEW' if new else 'FALLBACK_OR_UNAVAILABLE_CANDIDATE',
            proxy_reference_both_endpoints_known=known, proxy_reference_matched=row['evaluation_reference']['matched'],
            proxy_signed_endpoint_distances_px=signed, proxy_observed_line_normal_RMS_px=error,
            proxy_line_quality=quality(error), support_query_count=len(query), support_query_ids=query,
            support_self_consistent=self_consistent, validation=gate,
            physical_ownership_certified=False))
    native_values = point_quality(native, proxy, valid)
    hw = row['raw_hw']
    native_eligible = {k for k, p in enumerate(native[:8]) if valid_point(p) and 0 <= p[0] < hw[1] and 0 <= p[1] < hw[0]}
    return dict(id=row['id'], session=row['session'], output_status=row['output_status'],
        approved_NEW=new, fallback_used=row['fallback_used'], pose=saved_scores(row),
        actual_cf_extents=row['actual_pose'].get('cf_extents'),
        mask_relation='WRONG_ON_KNOWN' if row['mask_audit']['mask_wrong_on_known'] else 'MATCHES_ON_KNOWN',
        mask_audit=row['mask_audit'], reference_scope='fixed-N3-phase existing GEOMETRIC_PROXY',
        proxy_matched=row['evaluation_reference']['matched'], native_point_evidence=native_values,
        native_all_evidence=point_pool(native_values, set(range(8))),
        native_nonH_eligible_evidence=point_pool(native_values, native_eligible - H),
        final_input_point_evidence=point_quality(row['input_points'], proxy, valid),
        final_point_pools={name: sorted(ids(solved[key], key)) for name, key in
                           (('used', 'used'), ('selected_fit', 'fit_input_ids'), ('raw_final_inliers', 'final_inliers'))},
        source_lines=values, final_local_layout=layout(solved),
        stored_final_point_line_geometry=({key: value.get('numerical_rank') for key, value in
            (solved.get('point_line_geometry') or {}).items() if isinstance(value, dict)}),
        source_observation_changed=False, GT_used_for_deployed_selection=False)


def summarize(joined):
    groups = ('source_line', 'original_unused_eligible', 'consumed_by_actual_adopted_boundary',
              'retained_factor', 'contradicted_rejected', 'unverified_retained', 'approved_NEW_final_line_inlier',
              'raw_final_candidate_line_inlier')
    line_quality = {g: Counter() for g in groups}
    line_refs = {g: defaultdict(list) for g in groups}
    confusion = {d: Counter() for d in DECISIONS}
    confusion_refs = {d: defaultdict(list) for d in DECISIONS}
    point_quality_by_decision = {d: {p: Counter() for p in ('used', 'selected_fit', 'raw_final_inliers')} for d in DECISIONS}
    bins = {d: Counter() for d in DECISIONS}
    states, reasons, counts = Counter(), Counter(), Counter()
    error_samples = defaultdict(list)
    projection_relations = Counter()
    frame_groups = Counter()
    paired_groups = {c: Counter() for c in ('N3', 'V5')}
    delta_samples = {c: defaultdict(list) for c in ('N3', 'V5')}
    classifications = defaultdict(list)
    for row in joined:
        counts['frames'] += 1
        counts['accepted_NEW_frames' if row['approved_NEW'] else 'fallback_or_failed_frames'] += 1
        correct = len(row['native_nonH_eligible_evidence']['correct_ids'])
        for comparator in ('N3', 'V5'):
            paired = row['vs_' + comparator]
            paired_groups[comparator][paired['pose_T_R_outcome']] += 1
            if paired['delta'] is not None:
                for metric, value in paired['delta'].items():
                    delta_samples[comparator][metric].append(value)
        frame_groups[json.dumps(dict(output_status=row['output_status'], mask_relation=row['mask_relation'],
            native_nonH_correct_count=correct, native_nonH_incorrect_count=len(row['native_nonH_eligible_evidence']['incorrect_ids']),
            outcome_vs_N3=row['vs_N3']['pose_T_R_outcome'], outcome_vs_V5=row['vs_V5']['pose_T_R_outcome']),
            sort_keys=True, separators=(',', ':'))] += 1
        for line in row['source_lines']:
            ref = dict(id=row['id'], edge=line['edge'])
            q = line['proxy_line_quality']
            counts['distinct_source_frame_edges'] += 1
            for group in groups:
                if line[group]:
                    line_quality[group][q] += 1
                    line_refs[group][q].append(ref)
            gate = line['validation']
            if gate is None:
                continue
            d = gate['decision']
            counts['actual_endpoint_validator_records'] += 1
            counts['solver_available_validator_records'] += int(gate['solver_available'])
            counts['checkable_validator_records'] += int(gate['checkable'])
            confusion[d][q] += 1
            confusion_refs[d][q].append(ref)
            states[str(gate['solver_state'])] += 1
            reasons[gate['reason']] += 1
            if q == GOOD and d == 'CONTRADICTED_REJECT':
                classifications['rejected_proxy_correct'].append(ref)
            if q == BAD and d == 'CONTRADICTED_REJECT':
                classifications['rejected_proxy_incorrect'].append(ref)
            if q == BAD and d != 'CONTRADICTED_REJECT':
                classifications['retained_proxy_incorrect'].append(ref)
                classifications['supported_proxy_incorrect' if d == 'SUPPORTED_RETAIN' else 'unverified_proxy_incorrect'].append(ref)
            if d == 'UNVERIFIED_RETAIN':
                classifications['all_unverified_retained'].append(ref)
            for pool, data in gate['native_point_pools'].items():
                if not data.get('witness_missing'):
                    point_quality_by_decision[d][pool].update(data['quality_counts'])
            pool = gate['native_point_pools']['used']
            bins[d][str(len(pool.get('correct_ids', []))) if not pool.get('witness_missing') else 'MISSING_WITNESS'] += 1
            for key in ('predicted_observed_line_normal_RMS_px', 'proxy_endpoint_projection_Euclidean_RMS_px'):
                if gate[key] is not None:
                    error_samples[key].append(gate[key])
                    error_samples[d + '/' + key].append(gate[key])
            projection_relations[gate['extent_relation_vs_N3']['relation']] += 1
    return dict(actual_counts=dict(counts), proxy_line_quality_by_pool={k: dict(v) for k, v in line_quality.items()},
        proxy_line_quality_frame_edge_ids={k: dict(v) for k, v in line_refs.items()},
        decision_x_proxy_line_quality={k: dict(v) for k, v in confusion.items()},
        decision_x_proxy_line_quality_frame_edge_ids={k: dict(v) for k, v in confusion_refs.items()},
        classification_counts={k: len(v) for k, v in classifications.items()},
        classification_frame_edge_ids=dict(classifications),
        validator_native_point_quality_by_decision={d: {p: dict(c) for p, c in pools.items()} for d, pools in point_quality_by_decision.items()},
        validator_native_used_correct_count_histogram={d: dict(c) for d, c in bins.items()},
        validator_solver_states=dict(states), validator_decision_reasons=dict(reasons),
        validator_returned_extents_vs_N3=dict(projection_relations),
        scalar_error_distributions_px={k: distribution(v) for k, v in error_samples.items()},
        frame_T_R_outcomes={k: dict(v) for k, v in paired_groups.items()},
        frame_mask_outcome_native_quality_bins=dict(frame_groups),
        frame_saved_pose_delta_distributions={c: {k: distribution(v) for k, v in data.items()} for c, data in delta_samples.items()},
        units=dict(line_counts='distinct frame/semantic-edge records; overlapping pool flags are not additive',
                   validator_point_counts='validator-record/corner incidences; reused frame corners can repeat across edge validators',
                   rejected_proxy_correct='agreement with saved geometric proxy <=8px, not a physical correctness certificate'))


def run(folder, output):
    folder, output = guard(folder, output)
    protocol_path = output / PROTOCOL
    protocol = read(protocol_path)
    require(protocol['policy'] == POLICY and protocol['inputs'] == inputs(folder), 'frozen audit code/input drift')
    require(all(not (output / name).exists() for name in OUTPUTS), 'preserve completed/interrupted audit attempt')
    write(output / OUTPUTS[0], dict(protocol=binding(protocol_path), actual_saved_arithmetic_runs=1,
          new_models_PnP_optimizers_rays_RGB_training_GT_pose_scoring_calls=0))
    start, complete, failure = time.monotonic(), False, None
    joined, summary = [], {}
    try:
        seal, score = read(folder / 'GEOMETRY_SEAL.json'), read(folder / 'SCORING_RECEIPT.json')
        require(seal['complete'] and seal['GT_read_allowed'] is False and seal['frames'] == 245 and
                seal['rows'] == 980 and seal['fixed_rows'] == 490, 'completed GT-free geometry missing')
        require(score['complete'] and score['scored_method_rows'] == 980 and score['fixed_control_rows'] == 490 and
                score['GT_access_only_after_complete_seal'] and score['new_pose_fits'] == 0,
                'completed post-seal scoring missing')
        require(read(folder / 'INFERENCE_RECEIPT.json')['complete'] and read(folder / 'V5_CONTROL_PARITY.json')['passed'],
                'successful completed inference/control parity missing')
        validated = read(folder / 'VALIDATION_CHECKS.json')
        require(validated['passed'] and validated['complete'], 'independent scalar validation checks missing')
        for name in ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'SCORING_RECEIPT.json',
                     'OBSERVATIONS.jsonl.gz', 'PREDICTIONS.jsonl.gz', 'FIXED_GEOMETRY_SEALED.jsonl.gz'):
            bound(folder / name, validated['inputs'][name])
        for name, saved in (('PROTOCOL.json', seal['protocol']), ('OBSERVATIONS.jsonl.gz', seal['observations']),
                            ('GEOMETRY_SEAL.json', score['geometry_seal']), ('PREDICTIONS.jsonl.gz', score['predictions']),
                            ('FIXED_PREDICTIONS.jsonl.gz', score['fixed_predictions'])):
            bound(folder / name, saved)
        observations = {}
        for row in rows(folder / 'OBSERVATIONS.jsonl.gz'):
            require(row['id'] not in observations and row['GT_input'] is False, 'source population/GT contract differs')
            observations[row['id']] = {k: row[k] for k in ('id', 'session', 'native_N3_points', 'corners', 'lines')}
        require(len(observations) == 245, '245 observations required')
        fixed, references = {}, {}
        for row in rows(folder / 'FIXED_PREDICTIONS.jsonl.gz'):
            key = row['method'], row['id']
            require(row['method'] in ('BASE', 'N3_SUBPIX') and row['id'] in observations and key not in fixed,
                    'fixed control population differs')
            fixed[key] = dict(pose=saved_scores(row), cf_extents=row['actual_pose'].get('cf_extents'))
            if row['method'] == 'N3_SUBPIX':
                require(row['native_points'] == observations[row['id']]['native_N3_points'], 'fresh native-N3 identity differs')
                references[row['id']] = row['evaluation_reference']
        require(len(fixed) == 490 and len(references) == 245, 'complete fixed controls required')
        primary_rows, controls, population = {}, {}, set()
        for row in rows(folder / 'PREDICTIONS.jsonl.gz'):
            key = row['method'], row['id']
            require(row['method'] in METHODS and row['id'] in observations and key not in population,
                    'new method population differs')
            population.add(key)
            if row['method'] == CONTROL:
                controls[row['id']] = dict(pose=saved_scores(row), cf_extents=row['actual_pose'].get('cf_extents'),
                                           status=row['output_status'])
            elif row['method'] == PRIMARY:
                ref = row['evaluation_reference']
                for key in ('phase_control', 'native_points_px', 'valid_native_ids', 'matched'):
                    require(ref[key] == references[row['id']][key], 'primary/fixed reference phase differs')
                primary_rows[row['id']] = primary(row, observations[row['id']])
        require(population == {(m, fid) for m in METHODS for fid in observations} and
                set(primary_rows) == set(controls) == set(observations), 'complete980/common245 population required')
        for fid in sorted(observations):
            row = primary_rows[fid]
            n3, control = fixed[('N3_SUBPIX', fid)], controls[fid]
            row.update(vs_N3=paired(row['pose'], n3['pose']), vs_V5=paired(row['pose'], control['pose']),
                extent_relation_vs_N3=extents_relation(row['actual_cf_extents'], n3['cf_extents']),
                extent_relation_vs_V5=extents_relation(row['actual_cf_extents'], control['cf_extents']),
                unchanged_v5_control_status=control['status'])
            for line in row['source_lines']:
                gate = line['validation']
                if gate is not None:
                    gate.update(extent_relation_vs_N3=extents_relation(gate['actual_cf_extents'], n3['cf_extents']),
                                extent_relation_vs_V5=extents_relation(gate['actual_cf_extents'], control['cf_extents']))
            joined.append(row)
        summary = summarize(joined)
        require(inputs(folder) == protocol['inputs'], 'completed input/code bytes changed')
        complete = True
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    with (output / OUTPUTS[1]).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for row in joined:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    result = dict(schema='completed_saved_v6_endpoint_gate_quality_audit_v1', complete=complete, passed=complete,
        failure=failure, protocol=binding(protocol_path), started=binding(output / OUTPUTS[0]),
        rows=binding(output / OUTPUTS[1]), row_count=len(joined), summary=summary,
        actual_saved_arithmetic_runs=1, actual_wall_seconds=time.monotonic() - start,
        new_models_PnP_optimizers_rays_RGB_training_GT_pose_scoring_calls=0,
        real_physical_boundary_ownership_certified=False, fully_statistically_independent=False,
        local_rank_labels_recomputed=False, deployed_policy_changed=False,
        limitations=['Proxy line quality is normal-distance agreement of a fixed semantic endpoint pair to an infinite observed line.',
            'It does not establish actual visible physical ownership, shared-corner identity, support along a finite physical edge, or independent GT authenticity.',
            'Available/checkable native consensus may be wrong. Point-quality labels are posthoc proxy errors, not solver inputs.',
            'H and Base proposals remain conditional dependencies. Existing raw labels, UNVERIFIED decisions and every deployed result are unchanged.',
            'All outcomes use saved operational pose scores; no new pose scoring, fitting, model execution or parameter selection occurs.'])
    write(output / OUTPUTS[2], result)
    print('GATE_QUALITY_COMPLETE', complete, 'rows', len(joined), 'failure', failure, flush=True)
    if not complete:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC, help='completed saved v6 result directory')
    parser.add_argument('--output', type=Path, default=DOC, help='new exclusive audit directory')
    args = parser.parse_args()
    {'freeze': freeze, 'run': run}[args.stage](args.input, args.output)


if __name__ == '__main__':
    main()
