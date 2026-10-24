"""Seal root's actual RGB observations; never reads another review or old label."""
from pathlib import Path
import datetime
import hashlib
import json

V = Path(__file__).resolve().parents[1]
O = Path(__file__).resolve().parent
M = V / 'evidence_recovered_v3' / 'BLIND_CROSS40_ALL_REVIEWERS.json'

def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def write_new(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    schema = json.loads((V / 'ROOT_BLIND_CROSS40_ROW_SCHEMA.json').read_text())
    manifest = json.loads(M.read_text())
    notes = json.loads((O / 'ROOT_RAW_NOTES_BLIND.json').read_text())
    assert len(manifest['cases']) == len(notes) == 40
    assert [n['ordinal'] for n in notes] == list(range(1, 41))
    assert all(n['actual_view_tool'] == 'tools.view_image' for n in notes)
    contacts = {c['page']: c for c in manifest['contacts']}
    rows = []
    for n, c in zip(notes, manifest['cases']):
        page = (n['ordinal'] - 1) // 2 + 1
        contact = contacts[page]
        assert c['case_id'] in contact['case_ids']
        b = contact['image']
        assert bind(Path(b['path'])) == b
        row = {k: c[k] for k in schema['required_identity'] if k not in ['reviewer', 'review_set']}
        row.update(reviewer='ROOT', review_set='ROOT_BLIND_CROSS40')
        for key in ['selected_instance_correspondence', 'native_part_visibility', 'query_relation']:
            assert n[key] in schema['enums'][key]
            row[key] = n[key]
        for key in ['reason_KO', 'ambiguity_KO']:
            assert isinstance(n[key], str) and n[key].strip()
            row[key] = n[key]
        supported = row['selected_instance_correspondence'] == 'CLEAR'
        if row['native_part_visibility'] in ['VISIBLE_SUPPORTED', 'EXTERNAL_BLOCK_SUPPORTED', 'SELF_HIDE_SUPPORTED']:
            assert supported
        if row['query_relation'] in ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET', 'OTHER_SELECTED_PALLET_PART']:
            assert supported
        candidate = supported and row['native_part_visibility'] == 'VISIBLE_SUPPORTED' and row['query_relation'] in ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET']
        row['part_cover_readiness'] = 'WEAK_CANDIDATE_PART_VISIBLE_AND_Q_RELATED' if candidate else 'UNKNOWN_NOT_ADMITTED'
        row['part_support_extent'] = None
        row['actually_viewed'] = {'manifest': bind(M), 'images': [b], 'tool': 'tools.view_image', 'viewed': True}
        row.update(schema['required_constant_fields'])
        rows.append(row)
    dest = O / 'ROOT_BLIND_CROSS40_OBSERVATIONS.jsonl'
    with dest.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_new(O / 'ROOT_BLIND_CROSS40_SEALED.json', {
        'completed': True, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'ROOT', 'count': 40, 'manifest': bind(M),
        'schema': bind(V / 'ROOT_BLIND_CROSS40_ROW_SCHEMA.json'), 'notes': bind(O / 'ROOT_RAW_NOTES_BLIND.json'),
        'observations': bind(dest), 'actual_contact_images_viewed': 20,
        'sealed_before_any_agent_votes_read': True, 'old_labels_or_model_outputs_read_for_review': False,
        'assistant_weak_observation_only': True, 'native_coordinate_gold': False,
        'training_admitted': False, 'original_labels_modified': False,
    })
    print(json.dumps(bind(O / 'ROOT_BLIND_CROSS40_SEALED.json')))

if __name__ == '__main__':
    main()
