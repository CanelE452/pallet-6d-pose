"""Four-way, per-field witness agreement. No historical labels or model outputs."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json

V = Path(__file__).resolve().parents[1]
O = Path(__file__).resolve().parent
FIELDS = ['selected_instance_correspondence', 'native_part_visibility', 'query_relation', 'part_cover_readiness']
PATHS = {
    'A': V / 'reviewer_a/review/BLIND_CROSS40_NATIVE_PART_WITNESS.jsonl',
    'B': V / 'reviewer_b/CROSS40_NATIVE_PART_WITNESS.jsonl',
    'C': V / 'reviewer_c/BLIND_CROSS40_OBSERVATIONS.jsonl',
    'ROOT': V / 'root_blind_review/ROOT_BLIND_CROSS40_OBSERVATIONS.jsonl',
}

def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    rootseal = V / 'root_blind_review/ROOT_BLIND_CROSS40_SEALED.json'
    seal = json.loads(rootseal.read_text())
    assert seal['sealed_before_any_agent_votes_read']
    mpath = V / 'evidence_recovered_v3/BLIND_CROSS40_ALL_REVIEWERS.json'
    manifest = json.loads(mpath.read_text())
    identities = {r['case_id']: r for r in manifest['cases']}
    schema = json.loads((V / 'reviewer_a/preparation/SHARED_OBSERVATION_ROW_SCHEMA.json').read_text())
    reviews = {}
    for who, p in PATHS.items():
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        assert len(rows) == len({r['case_id'] for r in rows}) == 40
        assert {r['case_id'] for r in rows} == set(identities)
        for r in rows:
            c = identities[r['case_id']]
            assert r['reviewer'] == who
            for key in ['global_index', 'query_id', 'native_slot', 'native_name']:
                assert r[key] == c[key]
            for key in FIELDS:
                assert r[key] in schema['enums'][key]
            for key, value in schema['required_constant_fields'].items():
                assert r[key] == value
            view = r['actually_viewed']
            assert view['manifest'] == bind(mpath) and view['viewed'] and view['tool'] == 'tools.view_image'
            allowed = [c['images']['card']] + [x['image'] for x in manifest['contacts'] if r['case_id'] in x['case_ids']]
            assert view['images'] and all(b in allowed and bind(Path(b['path'])) == b for b in view['images'])
            if r['selected_instance_correspondence'] != 'CLEAR':
                assert r['native_part_visibility'] in ['UNKNOWN', 'BOUNDARY_OR_GAP_AMBIGUOUS']
                assert r['query_relation'] not in ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET', 'OTHER_SELECTED_PALLET_PART']
            candidate = r['selected_instance_correspondence'] == 'CLEAR' and r['native_part_visibility'] == 'VISIBLE_SUPPORTED' and r['query_relation'] in ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET']
            assert (r['part_cover_readiness'] == 'WEAK_CANDIDATE_PART_VISIBLE_AND_Q_RELATED') == candidate
        reviews[who] = {r['case_id']: r for r in rows}
    joined = []
    for cid, c in identities.items():
        votes = {w: reviews[w][cid] for w in PATHS}
        consensus = {}
        for field in FIELDS:
            values = [r[field] for r in votes.values()]
            consensus[field] = {'unanimous': len(set(values)) == 1, 'value': values[0] if len(set(values)) == 1 else None, 'votes': {w: votes[w][field] for w in PATHS}}
        joined.append({'case_id': cid, 'global_index': c['global_index'], 'native_slot': c['native_slot'], 'source_id': c['source_id'], 'consensus': consensus, 'original_votes': votes, 'native_target': -1, 'quality_target': -1, 'training_admitted': False})
    summary = {'completed': True, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'count': 40,
        'root_seal_before_vote_read': bind(rootseal), 'inputs': {w: bind(p) for w, p in PATHS.items()},
        'per_field': {f: {'unanimous_count': sum(r['consensus'][f]['unanimous'] for r in joined), 'disagreement_count': sum(not r['consensus'][f]['unanimous'] for r in joined), 'unanimous_values': dict(Counter(r['consensus'][f]['value'] for r in joined if r['consensus'][f]['unanimous']))} for f in FIELDS},
        'per_native': {str(k): {'N': sum(r['native_slot'] == k for r in joined), 'part_visibility_unanimous': sum(r['native_slot'] == k and r['consensus']['native_part_visibility']['unanimous'] for r in joined)} for k in range(8)},
        'unanimous_visible_part_and_q_background_case_ids': [r['case_id'] for r in joined if r['consensus']['native_part_visibility']['value'] == 'VISIBLE_SUPPORTED' and r['consensus']['query_relation']['value'] == 'BACKGROUND_OR_GAP'],
        'unanimous_visible_part_and_q_other_part_case_ids': [r['case_id'] for r in joined if r['consensus']['native_part_visibility']['value'] == 'VISIBLE_SUPPORTED' and r['consensus']['query_relation']['value'] == 'OTHER_SELECTED_PALLET_PART'],
        'disagreement_case_ids': {f: [r['case_id'] for r in joined if not r['consensus'][f]['unanimous']] for f in FIELDS},
        'no_majority_or_geometry_override': True, 'all_unknown_or_disagreement_retained': True,
        'historical_D_E_owner_or_model_reads': 0, 'primary390_still_pending': True,
        'assistant_weak_observations_only': True, 'all_new_native_quality_targets': -1, 'training_admitted': False, 'fully_solved': False}
    with (O / 'CROSS40_FOUR_WAY_WITNESSES.jsonl').open('x') as f:
        for row in joined:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with (O / 'CROSS40_AGREEMENT_BEFORE_HISTORICAL_LABEL_JOIN.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
