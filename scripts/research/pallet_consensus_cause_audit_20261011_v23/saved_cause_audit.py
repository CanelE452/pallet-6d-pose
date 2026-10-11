"""Join frozen records only. No solver, projection, model, or new pose scoring."""
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = {
    'routing': Path('/tmp/pallet-consensus-routing-private-20261011-v22/audit_labels_ready/ROWS.jsonl.gz'),
    'routing_summary': Path('/tmp/pallet-consensus-routing-private-20261011-v22/audit_labels_ready/SUMMARY.json'),
    'geometry': Path('/tmp/pallet-whole-u-closure-private-20261011-v21/geometry/GEOMETRY.jsonl.gz'),
    'posthoc': Path('/tmp/pallet-whole-u-closure-private-20261011-v21/evaluation/POSTHOC_ROWS.jsonl.gz'),
}


def bind(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def candidate(c, good):
    ids = c.get('fit_input_ids', [])
    inliers = c.get('final_inliers', [])
    return {k: c.get(k) for k in (
        'candidate_index', 'proposal_index', 'logical_start_index', 'dimension_index',
        'proposal_kind', 'accepted', 'rejected', 'score_count', 'truncated_sse_px2',
        'fit_input_ids', 'final_inliers', 'fit_pool_all_final8px_inliers',
    )} | {'fit_proxy_good_ids': sorted(set(ids) & good),
         'inlier_proxy_good_ids': sorted(set(inliers) & good)}


def main():
    bindings = {name: bind(path) for name, path in INPUTS.items()}
    bindings['producer'] = bind(Path(__file__))
    write('PROTOCOL.json', {
        'schema': 'saved_observation_cause_join_v23', 'inputs': bindings,
        'scope': 'existing Clean153+Moderate92; Severe excluded; all 245 retained',
        'quality': 'existing POSTHOC CORRECT_WITHIN8PX is annotation proxy, not physical Rt or ownership truth',
        'operation': 'ID joins and recorded scalar/count comparisons only',
        'new_training_updates': 0, 'new_RGB': 0, 'new_PnP': 0,
        'new_pose_accuracy_evaluations': 0, 'threshold_changes': 0,
        'alternative_pose_accuracy_is_not_inferred': True,
    })
    routing = {r['id']: r for r in rows(INPUTS['routing'])}
    posthoc = {r['id']: r for r in rows(INPUTS['posthoc'])}
    assert len(routing) == len(posthoc) == 245
    assert set(routing) == set(posthoc)
    selected = {k for k, r in routing.items() if r['routing'] == 'ALL_STANDARD_CANDIDATES_REJECTED'}
    declared = json.loads(INPUTS['routing_summary'].read_text())['strata']['combined']['routing_frame_ids']['ALL_STANDARD_CANDIDATES_REJECTED']
    assert selected == set(declared) and len(selected) == 59
    seen, audited = set(), []
    overall = collections.Counter()
    with (ROOT / 'ROWS.jsonl').open('w') as out:
        for g in rows(INPUTS['geometry']):
            ident = g['id']; assert ident not in seen; seen.add(ident)
            r, p, s = routing[ident], posthoc[ident], g['solver']
            adopted = list(g['selected_corner_ids'])
            boundary = p['boundary_candidates']
            overall['frames'] += 1
            overall['frames_with_any_adopted_boundary'] += bool(adopted)
            overall['adopted_boundary_points'] += len(adopted)
            loo_records = g['observation_contract'].get('cornerwise_records', [])
            overall['original_LOO_accepted'] += sum(bool(a['accepted']) for a in loo_records)
            overall['original_LOO_accepted_but_not_adopted'] += sum(a['accepted'] and a['id'] not in adopted for a in loo_records)
            if ident not in selected:
                continue
            used = r['pools']['used']; good = set(used['proxy_correct_ids'])
            known_good = set(used['known_visibility_proxy_correct_ids'])
            assert set(s['used']) == set(used['ids'])
            assert set(s['used']).isdisjoint(g['hidden_initial'])
            corner = {c['id']: c for c in p['corners']}
            assert good == {k for k in used['ids'] if corner[k]['input_quality'] == 'CORRECT_WITHIN8PX'}
            expected = {(d['index'], ids) for d in s['registered_dimensions'] for ids in itertools.combinations(sorted(good), 4)}
            recorded = {(a['dimension_index'], tuple(a['generator_ids'])): a for a in s['direct_point_generation']}
            assert expected <= set(recorded)
            good_generation = [recorded[k] for k in sorted(expected)]
            good_proposals = [a for a in s['proposal_solutions'] if a['proposal_kind'] == 'DIRECT_FINITE_FOUR_POINT_ROOT' and set(a['point_generator_ids']) <= good]
            proposal_ids = {a['proposal_index'] for a in good_proposals}
            good_candidates = [candidate(c, good) for c in s['all_candidate_solutions'] if c.get('proposal_index') in proposal_ids]
            good_attempts = [a for a in s['optimizer_attempts'] if a.get('proposal_index') in proposal_ids]
            best = s.get('best_candidate')
            item = {
                'id': ident, 'session': g['session'], 'routing': r['routing'],
                'new_pose_estimated': g['new_pose_estimated'], 'fallback_used': g['fallback_used'],
                'output_status': g['output_status'], 'known_mask_state': r['known_mask_state'],
                'saved_outcome_vs_N3': r['saved_T_R_outcome_vs_controls']['N3_SUBPIX'],
                'saved_errors': {k: r['saved_errors'][k] for k in ('N3_SUBPIX', 'HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE')},
                'hidden_initial': g['hidden_initial'], 'selected_boundary_corner_ids': adopted,
                'pools': r['pools'], 'used_proxy_good_count': len(good),
                'used_known_visibility_proxy_good_count': len(known_good),
                'input_corner_errors_px': [{k: corner[i][k] for k in ('id','human_state','input_quality','N3_error_px','input_error_px')} for i in used['ids']],
                'all_proxy_good_four_ID_generation': good_generation,
                'all_proxy_good_four_ID_proposals': [{k: a.get(k) for k in ('proposal_index','dimension_index','point_generator','point_generator_ids','point_solution_index','whole_pool_score')} for a in good_proposals],
                'all_proxy_good_four_ID_refit_attempts': good_attempts,
                'all_proxy_good_four_ID_refit_candidates': good_candidates,
                'selected_candidate': None if best is None else candidate(best, good),
                'live_standard_candidates': r['live_standard_candidates'],
                'boundary_candidates': boundary,
                'cornerwise_LOO_records': g['observation_contract'].get('cornerwise_records', []),
                'physical_ownership_truth': 'UNKNOWN',
                'all_good_four_ID_expected_and_recorded': len(expected),
                'all_good_four_ID_valid_root_count': sum(a['returned_valid_roots'] for a in good_generation),
                'all_good_four_ID_geometry_proposal_count': len(good_proposals),
                'all_good_four_ID_accepted_refit_count': sum(bool(a['accepted']) for a in good_candidates),
                'all_good_four_ID_selected': best is not None and best.get('proposal_index') in proposal_ids,
            }
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
            audited.append(item)
    assert seen == set(routing) and len(audited) == 59
    counts = collections.Counter()
    for a in audited:
        counts['frames'] += 1
        counts['NEW'] += a['new_pose_estimated']
        counts['FALLBACK'] += a['fallback_used']
        counts['used_proxy_good_at_least4'] += a['used_proxy_good_count'] >= 4
        counts['used_known_visibility_proxy_good_at_least4'] += a['used_known_visibility_proxy_good_count'] >= 4
        counts['fit_known_visibility_proxy_good_at_least4'] += len(a['pools']['approved_fit']['known_visibility_proxy_correct_ids']) >= 4
        counts['unknown_used_frames'] += bool(a['pools']['used']['proxy_unknown_ids'])
        counts['any_boundary_adopted'] += bool(a['selected_boundary_corner_ids'])
        counts['adopted_boundary_points'] += len(a['selected_boundary_corner_ids'])
        counts['any_all_good_four_ID_valid_root'] += a['all_good_four_ID_valid_root_count'] > 0
        counts['any_all_good_four_ID_accepted_refit'] += a['all_good_four_ID_accepted_refit_count'] > 0
        counts['all_good_four_ID_selected'] += a['all_good_four_ID_selected']
        counts['missing_all_good_four_ID_requests'] += 0
    cross = collections.Counter((a['used_proxy_good_count'] >= 4, a['saved_outcome_vs_N3']) for a in audited)
    subset_outcomes = [dict(used_proxy_good_at_least4=k[0], saved_outcome_vs_N3=k[1], frames=v) for k,v in sorted(cross.items())]
    write('SUMMARY.json', {
        'schema': 'saved_observation_cause_join_summary_v23', 'complete': True,
        'all245_counts': dict(overall), 'rejected59_counts': dict(counts),
        'rejected59_good_supply_vs_saved_outcome': subset_outcomes,
        'four_ID_request_coverage_proved_only_for_recorded_proxy_good_subsets': True,
        'four_proxy_good_points_do_not_prove_exact_correspondence_or_global_pose_accuracy': True,
        'no_alternative_pose_accuracy_claim': True, 'physical_ownership_truth': 'UNKNOWN',
        'new_PnP': 0, 'new_training_updates': 0, 'new_pose_accuracy_evaluations': 0,
        'rows': bind(ROOT / 'ROWS.jsonl'),
    })
    assert bindings == {name: bind(Path(value['path'])) for name, value in bindings.items()}
    write('CHECKS.json', {'schema': 'saved_observation_cause_join_checks_v23', 'passed': True,
        'source_inputs_unchanged': True, 'all245_ID_join_exact': True, 'declared59_IDs_exact': True,
        'proxy_good_labels_match_existing_posthoc': True, 'H_used_disjoint_all59': True,
        'all_proxy_good_four_ID_requests_present': True, 'new_PnP': 0,
        'new_projection': 0, 'new_pose_accuracy_evaluations': 0})
    print(json.dumps({'complete': True, 'all245': dict(overall), 'rejected59': dict(counts), 'cross': subset_outcomes}, ensure_ascii=False))


if __name__ == '__main__':
    main()
