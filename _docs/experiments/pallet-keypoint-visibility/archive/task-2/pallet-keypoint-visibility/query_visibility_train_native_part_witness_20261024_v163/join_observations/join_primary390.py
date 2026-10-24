"""Validate and retain all fixed source witnesses. No old labels/models are read."""
from pathlib import Path
from collections import Counter
import argparse
import datetime
import hashlib
import json

V = Path(__file__).resolve().parents[1]
O = Path(__file__).resolve().parent
FIELDS = ['selected_instance_correspondence', 'native_part_visibility', 'query_relation', 'part_cover_readiness']

def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def readjl(p):
    return [json.loads(l) for l in p.read_text().splitlines()]

def write(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    ap = argparse.ArgumentParser()
    for who in 'ABC':
        ap.add_argument('--' + who, required=True, type=Path)
        ap.add_argument('--' + who + '-seal', required=True, type=Path)
    args = vars(ap.parse_args())
    manifests = {w: V / 'evidence_recovered_v3' / ('REVIEW_INPUT_GROUP_' + w + '.json') for w in 'ABC'}
    crosspath = V / 'evidence_recovered_v3/BLIND_CROSS40_ALL_REVIEWERS.json'
    cross = json.loads(crosspath.read_text())
    crossids = {c['case_id'] for c in cross['cases']}
    joinedcrosspath = O / 'CROSS40_FOUR_WAY_WITNESSES_V2.jsonl'
    joinedcross = {r['case_id']: r for r in readjl(joinedcrosspath)}
    assert set(joinedcross) == crossids and len(crossids) == 40
    schema = json.loads((V / 'reviewer_a/preparation/SHARED_OBSERVATION_ROW_SCHEMA.json').read_text())
    allmeta = readjl(V / 'evidence_recovered_v3/ALL390_BLINDED_EVIDENCE.jsonl')
    assert len(allmeta) == 390
    primary = {}
    receipts = {}
    for who in 'ABC':
        path, sealpath = args[who].resolve(), args[who + '_seal'].resolve()
        assert path.is_relative_to(V) and sealpath.is_relative_to(V)
        seal = json.loads(sealpath.read_text())
        assert seal.get('completed', seal.get('PASS', False)) is True
        manifest = json.loads(manifests[who].read_text())
        identities = {r['case_id']: r for r in manifest['cases']}
        rows = readjl(path)
        assert len(rows) == len({r['case_id'] for r in rows}) == 130
        assert {r['case_id'] for r in rows} == set(identities)
        for r in rows:
            cid = r['case_id']; c = identities[cid]
            assert cid not in primary and r['reviewer'] == who and r['review_set'] == 'PRIMARY130'
            for key in ['global_index', 'query_id', 'native_slot', 'native_name']:
                assert r[key] == c[key]
            for key in ['query_xy', 'L', 'frame_index', 'source_id', 'split', 'selected_gt_instance_index']:
                if key in r:
                    assert r[key] == c[key]
            for key in FIELDS:
                assert r[key] in schema['enums'][key]
            for key, value in schema['required_constant_fields'].items():
                assert r[key] == value
            for key in ['reason_KO', 'ambiguity_KO']:
                assert isinstance(r[key], str) and r[key].strip()
            if r['selected_instance_correspondence'] != 'CLEAR':
                assert r['native_part_visibility'] in ['UNKNOWN', 'BOUNDARY_OR_GAP_AMBIGUOUS']
                assert r['query_relation'] not in ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET', 'OTHER_SELECTED_PALLET_PART']
            candidate = r['selected_instance_correspondence'] == 'CLEAR' and r['native_part_visibility'] == 'VISIBLE_SUPPORTED' and r['query_relation'] in ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET']
            assert (r['part_cover_readiness'] == 'WEAK_CANDIDATE_PART_VISIBLE_AND_Q_RELATED') == candidate
            view = r['actually_viewed']
            assert view['viewed'] and view['tool'] == 'tools.view_image'
            allowed_manifests = [bind(manifests[who])] + ([bind(crosspath)] if cid in crossids else [])
            assert view['manifest'] in allowed_manifests
            allowed_images = list(c['images'].values())
            allowed_images += [x['image'] for x in manifest['contacts'] if cid in x['case_ids']]
            allowed_images += [x['image'] for x in cross['contacts'] if cid in x['case_ids']]
            assert view['images'] and all(b in allowed_images and bind(Path(b['path'])) == b for b in view['images'])
            if cid in joinedcross:
                original = joinedcross[cid]['original_votes'][who]
                for key in FIELDS + ['reason_KO', 'ambiguity_KO', 'part_support_extent']:
                    assert r[key] == original[key], (who, cid, key, 'overlap must reuse same observation')
            primary[cid] = r
        receipts[who] = {'observations': bind(path), 'seal': bind(sealpath), 'manifest': bind(manifests[who]), 'count': 130, 'cross_overlap_count': len(set(identities) & crossids)}
    assert set(primary) == {c['case_id'] for c in allmeta}
    combined = []
    for c in allmeta:
        cid = c['case_id']
        p = primary[cid]
        consensus = joinedcross[cid]['consensus'] if cid in joinedcross else None
        combined.append({
            'case_id': cid, 'global_index': c['global_index'], 'query_id': c['query_id'],
            'native_slot': c['native_slot'], 'native_name': c['native_name'], 'frame_index': c['frame_index'],
            'source_id': c['source_id'], 'split': c['split'], 'query_xy': c['query_xy'], 'L': c['L'],
            'selected_gt_instance_index': c['selected_gt_instance_index'],
            'primary_observation': p, 'review_support': 'FOUR_WAY_CROSS' if consensus else 'SINGLE_REVIEWER_WEAK',
            'four_way_consensus': consensus,
            'effective_observation': {f: consensus[f]['value'] if consensus else p[f] for f in FIELDS},
            'all_new_native_visibility_target': -1, 'all_new_point_quality_target': -1,
            'training_admitted': False, 'original_labels_modified': False,
        })
    def counts(rows):
        return {'N': len(rows), 'primary_observations': {f: dict(Counter(r['primary_observation'][f] for r in rows)) for f in FIELDS},
            'effective_observations_with_disagreement_null': {f: dict(Counter('DISAGREEMENT' if r['effective_observation'][f] is None else r['effective_observation'][f] for r in rows)) for f in FIELDS},
            'primary_joint_instance_visibility_query': dict(Counter(' | '.join(r['primary_observation'][f] for f in FIELDS[:3]) for r in rows))}
    dest = O / 'ALL390_PRIMARY_AND_CROSS_WITNESSES.jsonl'
    with dest.open('x') as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    summary = {'completed': True, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'count': 390,
        'primary_observation_count': 390, 'single_review_weak_count': 350, 'cross40_count': 40, 'cross_has_four_votes_no_duplicate_overlap': True,
        'reviewer_receipts': receipts, 'cross_witnesses': bind(joinedcrosspath), 'all390': bind(dest),
        'all390_summary': counts(combined), 'single350_summary': counts([r for r in combined if r['review_support'] == 'SINGLE_REVIEWER_WEAK']),
        'cross40_summary': counts([r for r in combined if r['review_support'] == 'FOUR_WAY_CROSS']),
        'per_native': {str(k): counts([r for r in combined if r['native_slot'] == k]) for k in range(8)},
        'per_scene': {scene: counts([r for r in combined if r['source_id'].split(':')[0] == scene]) for scene in sorted({r['source_id'].split(':')[0] for r in combined})},
        'historical_D_E_owner_or_model_reads': 0, 'old_label_comparison_barrier_satisfied': True,
        'four_way_disagreement_kept_not_majority_resolved': True,
        'all_new_native_quality_targets': -1, 'training_admitted': False, 'original_labels_modified': False,
        'missing_real_materials': ['SQUARE', 'non-Euro-WOOD'], 'native6_count': 2,
        'full_scope_WOOD_SQUARE_native8_unchanged': True, 'assistant_weak_observation_only': True, 'fully_solved': False}
    write(O / 'ALL390_REVIEW_BARRIER_AND_SUMMARY.json', summary)
    print(json.dumps({'barrier': bind(O / 'ALL390_REVIEW_BARRIER_AND_SUMMARY.json'), 'all390_summary': summary['all390_summary']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
