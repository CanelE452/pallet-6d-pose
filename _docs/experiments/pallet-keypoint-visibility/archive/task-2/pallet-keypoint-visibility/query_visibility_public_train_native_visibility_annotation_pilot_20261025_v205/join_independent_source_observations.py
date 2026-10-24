"""Join two sealed SOURCE TRAIN weak observations; never grant fit admission."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json

HERE = Path(__file__).absolute().parent
RULE = HERE / 'ROOT_PROSPECTIVE_INDEPENDENT_OBSERVATION_JOIN_RULE.json'
RULE_SHA = 'b2a2818587d5f710bb520942f3f4f03a54b691031a88b2418e208fd162b3b5fb'
STATE = 'part_visibility_D_E_S_UNKNOWN'
RELATION = 'query_relation_on_near_off_UNKNOWN'
IDENTITY = ('target_instance_identified', 'intended_part_identified')


def safe(path):
    path = Path(path).absolute()
    assert 'v112' not in str(path).lower() and '..' not in path.parts
    assert HERE in path.parents
    for parent in (path, *path.parents):
        assert not parent.is_symlink()
    return path


def binding(path):
    path = safe(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(bound):
    assert binding(bound['path']) == bound
    return json.loads(safe(bound['path']).read_text())


def classification(available, a, c):
    if not available:
        return 'ORIGINAL_INPUT_U', None
    for x in (a, c):
        assert x[STATE] in ('D', 'E', 'S', 'UNKNOWN')
        assert x[RELATION] in ('ON', 'NEAR', 'OFF', 'UNKNOWN')
        assert x['location_error_possible'] in ('YES', 'NO', 'UNKNOWN')
        assert all(x[k] in ('YES', 'NO', 'UNKNOWN') for k in IDENTITY)
    if any(x[RELATION] == 'OFF' or x['location_error_possible'] == 'YES' for x in (a, c)):
        return 'LOCATION_ERROR_OR_OFF', None
    if a[STATE] == c[STATE] == 'S':
        return 'SELF_CONSENSUS', None
    if a[STATE] == c[STATE] and a[STATE] in ('D', 'E'):
        aligned = (
            all(x[k] == 'YES' for x in (a, c) for k in IDENTITY)
            and a[RELATION] == c[RELATION] and a[RELATION] in ('ON', 'NEAR')
            and all(x['location_error_possible'] == 'NO' and x['short_visual_evidence'].strip()
                    for x in (a, c))
            and (a[STATE] != 'E' or all(x['external_occluder_evidence'].strip() for x in (a, c)))
        )
        return ('ALIGNED_WEAK_CONSENSUS_CANDIDATE', a[STATE]) if aligned else ('INSUFFICIENT_ALIGNED_EVIDENCE', None)
    if a[STATE] == c[STATE] == 'UNKNOWN':
        return 'UNKNOWN_CONSENSUS', None
    return 'DISAGREEMENT_OR_UNRESOLVED', None


def fixture_checks():
    base = {STATE: 'D', RELATION: 'ON', 'target_instance_identified': 'YES',
            'intended_part_identified': 'YES', 'location_error_possible': 'NO',
            'short_visual_evidence': 'identified corner visible', 'external_occluder_evidence': ''}
    tests = [
        (False, base, base, ('ORIGINAL_INPUT_U', None)),
        (True, base, base, ('ALIGNED_WEAK_CONSENSUS_CANDIDATE', 'D')),
        (True, {**base, STATE: 'E'}, {**base, STATE: 'E'}, ('INSUFFICIENT_ALIGNED_EVIDENCE', None)),
        (True, {**base, RELATION: 'OFF'}, base, ('LOCATION_ERROR_OR_OFF', None)),
        (True, base, {**base, RELATION: 'NEAR'}, ('INSUFFICIENT_ALIGNED_EVIDENCE', None)),
        (True, {**base, STATE: 'S'}, {**base, STATE: 'S'}, ('SELF_CONSENSUS', None)),
        (True, base, {**base, STATE: 'UNKNOWN'}, ('DISAGREEMENT_OR_UNRESOLVED', None)),
    ]
    for available, a, c, expected in tests:
        assert classification(available, a, c) == expected
    return {'completed': True, 'PASS': True, 'invented_only': True, 'count': len(tests),
            'real_observation_or_image_reads': 0}


def run(entry_path):
    entry = json.loads(safe(entry_path).read_text())
    assert entry['approved_saved_observation_join'] is True and entry['training_admission'] is False
    assert entry['code'] == binding(__file__)
    assert entry['rule'] == binding(RULE) and entry['rule']['sha256'] == RULE_SHA
    manifest = read(entry['preview_manifest'])
    docs = [read(entry[k]) for k in ('observer_a', 'observer_c')]
    expected = manifest['slots48']
    assert len(expected) == len(docs[0]['rows']) == len(docs[1]['rows']) == 48
    maps = [{r['row_id']: r for r in d['rows']} for d in docs]
    assert all(len(m) == 48 for m in maps)
    assert all(set(m) == {r['row_id'] for r in expected} for m in maps)
    rows = []
    for slot in expected:
        a, c = [m[slot['row_id']] for m in maps]
        for x, observer in ((a, 'A'), (c, 'C')):
            assert x['observer'] == observer and x['training_admission'] is False
            assert x['authority'].startswith('INDEPENDENT_ASSISTANT_SOURCE_TRAIN_WEAK')
            for k in ('row_id', 'native_slot', 'query_xy', 'L', 'original_query_available', 'frame_key'):
                assert x[k] == slot[k]
            if not slot['original_query_available']:
                assert x[STATE] == x[RELATION] == 'UNKNOWN'
        category, state = classification(slot['original_query_available'], a, c)
        rows.append({**{k: slot[k] for k in ('row_id', 'frame_key', 'native_slot', 'query_xy', 'L', 'original_query_available')},
                     'A_state': a[STATE], 'C_state': c[STATE], 'A_query_relation': a[RELATION],
                     'C_query_relation': c[RELATION], 'category': category, 'consensus_state_candidate': state,
                     'training_admission': False, 'authority': 'TWO_ASSISTANT_SOURCE_TRAIN_WEAK_NOT_GOLD'})
    output = safe(entry['output'])
    assert not output.exists()
    output.mkdir()
    result = {'completed': True, 'PASS': True, 'code': binding(__file__), 'entry': binding(entry_path),
              'rule': binding(RULE), 'preview_manifest': entry['preview_manifest'],
              'observer_a': entry['observer_a'], 'observer_c': entry['observer_c'],
              'all48_original_slots_retained': True, 'original_U': sum(not r['original_query_available'] for r in rows),
              'category_counts': dict(Counter(r['category'] for r in rows)),
              'consensus_candidate_state_counts': dict(Counter(r['consensus_state_candidate'] for r in rows
                                                              if r['consensus_state_candidate'])),
              'observer_A_state_counts': dict(Counter(r['A_state'] for r in rows)),
              'observer_C_state_counts': dict(Counter(r['C_state'] for r in rows)),
              'new_fit_model_forward_or_training_admission': False,
              'protected_eval_or_source_body_reads': 0, 'goal_solved': False, 'rows': rows}
    (output / 'JOINED_OBSERVATIONS48.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (output / 'JOINED_OBSERVATIONS48.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--invented-checks', action='store_true')
    group.add_argument('--entry', type=Path)
    args = parser.parse_args()
    print(json.dumps(fixture_checks())) if args.invented_checks else run(args.entry)
