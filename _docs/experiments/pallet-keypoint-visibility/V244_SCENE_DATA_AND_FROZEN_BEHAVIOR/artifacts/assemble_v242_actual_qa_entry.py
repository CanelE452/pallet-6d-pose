"""Bind the completed SOURCE scene data to the independent all-row replay."""
from pathlib import Path
import hashlib
import json

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(s in str(p).casefold() for s in ('v112', '__pycache__', '.pyc'))
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink()
    return p


def binding(p):
    p = safe(p)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def short(b):
    safe(b['path'])
    return {k: b[k] for k in ('path', 'sha256', 'size_bytes')}


def save(p, obj):
    with safe(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    cpath = R / 'V242_ROOT_SCENE_CF/scene_pairs_v1/COMPLETE.json'
    c = json.loads(safe(cpath).read_bytes())
    assert c['completed'] and c['PASS']
    assert c['counts']['attempted'] == 5406
    assert c['code']['sha256'] == '83a84fb22e177b0ea012b04f5c622e0ac71f9c5bdd612b9b63e4f1bae76538ca'
    assert c['training_executed'] is False and c['classifier_or_encoder_executed'] is False
    h = R / 'V242_SCENE_DATA_QA_C'
    ready = h / 'FINAL_READY.json'
    assert binding(ready)['sha256'] == 'a3c1a97fc67db6cdf304e4605b3380bd40cf9433424aeb988517bc35184fced0'
    e = json.loads(safe(h / 'ROOT_ENTRY_FALSE_TEMPLATE.json').read_bytes())
    assert e['approved_independent_SOURCE_scene_dataset_QA'] is False
    assert e['code']['sha256'] == '000a9d31b7626b73405df3e97f08618e7e121a2339fbca33f78a163eab6ecdcb'
    for key in ('code', 'math_code', 'io_code', 'protocol'):
        actual = binding(e[key]['path'])
        assert actual['sha256'] == e[key]['sha256']
    e['dataset_complete'] = binding(cpath)
    e['pair_metadata'] = short(c['outputs']['PAIR_METADATA.jsonl'])
    for name in e['outputs']:
        e['outputs'][name] = short(c['outputs'][name])
    e['array_whitelist'] = list(e['outputs'].values())
    assert len(e['array_whitelist']) == 11
    base_doc = json.loads(safe(e['parent_image_whitelist']['path']).read_bytes())
    e['base_image_whitelist'] = [row['image'] for row in base_doc['images']]
    assert len(e['base_image_whitelist']) == 804
    for b in e['base_image_whitelist']:
        safe(b['path'])  # Lexical and ancestor guard only, no payload body/header.
    scope = {'schema': 'V242_ROOT_INDEPENDENT_NUMERIC_DATA_QA_SCOPE',
             'approved_independent_SOURCE_scene_dataset_QA': True,
             'dataset_complete': e['dataset_complete'], 'code': e['code'],
             'math_code': e['math_code'], 'io_code': e['io_code'],
             'parents': e['parents'], 'base_image_whitelist': e['base_image_whitelist'],
             'array_whitelist': e['array_whitelist'], 'pair_metadata': e['pair_metadata'],
             'output_directory': e['output_directory'],
             'all5406_reconstructed': True, 'all_constructed_local8_and_context': True,
             'extra_donor_role_or_mask_file_access': False,
             'rawGT_or_model_or_private_current_held_access': False,
             'q_opacity_is_diagnostic_only_no_relabel_or_selection': True,
             'automatic_retry': False, 'training_or_candidate_admission': False}
    scope_path = R / 'V242_ROOT_SCENE_CF/READ_SCOPE_BEFORE_INDEPENDENT_QA.json'
    save(scope_path, scope)
    e['root_scope'] = binding(scope_path)
    e['metadata_whitelist'] += [e['root_scope'], e['dataset_complete'], e['pair_metadata']]
    assert not safe(e['output_directory']).exists()
    e['approved_independent_SOURCE_scene_dataset_QA'] = True
    e['Root_review'] = 'Exact code/independent lift/taps and streaming sampler reviewed; data-only all-row replay. No numeric access during assembly.'
    out = R / 'V242_ROOT_SCENE_CF/ACTUAL_INDEPENDENT_DATA_QA_ENTRY.json'
    save(out, e)
    print(json.dumps({'completed': True, 'PASS': True, 'entry': binding(out),
                      'base_images': 804, 'array_files': 11,
                      'new_numeric_payload_body_or_header_access_during_assembly': 0}))


if __name__ == '__main__':
    main()
