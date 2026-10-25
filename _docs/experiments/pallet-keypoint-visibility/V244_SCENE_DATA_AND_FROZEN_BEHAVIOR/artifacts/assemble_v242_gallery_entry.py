"""Bind completed all-row QA to the prospectively fixed pre-fit gallery."""
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


def save(p, obj):
    with safe(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    cpath = R / 'V242_ROOT_SCENE_CF/scene_pairs_v1/COMPLETE.json'
    c = json.loads(safe(cpath).read_bytes())
    assert c['completed'] and c['PASS']
    assert binding(cpath)['sha256'] == '12f973806eccdbeae25afea8f6e23fb43da738f2d412d7dbdbb82ea8d087d8bc'
    qpath = R / 'V242_SCENE_DATA_QA_C/actual_review_v1/REVIEW.json'
    q = json.loads(safe(qpath).read_bytes())
    assert q['completed'] and q['PASS']
    assert q['dataset_complete']['sha256'] == binding(cpath)['sha256']
    h = R / 'V242_SCENE_GALLERY_A'
    assert binding(h / 'FINAL_READY_V2.json')['sha256'] == '825c16dbd669cf6db3bd95edbca7117056ba768f8e226dbf2828e265d21fbff9'
    t = h / 'ENTRY_FALSE_TEMPLATE_V2.json'
    assert binding(t)['sha256'] == 'd26bfb6237e46256342319d55296b337964c9bf97eefaccaf63a66834d6829d3'
    e = json.loads(safe(t).read_bytes())
    assert e['approved_V242_SOURCE_gallery'] is False
    assert binding(e['code']['path'])['sha256'] == 'eaf24a38ec9bf4f49f7cc0532b51fdb42a49dd1436e4a63de6e3713d7bd7666c'
    e['dataset_complete'] = binding(cpath)
    e['data_QA'] = binding(qpath)
    e['pair_metadata'] = {k: c['outputs']['PAIR_METADATA.jsonl'][k] for k in ('path', 'sha256', 'size_bytes')}
    names = ('CF_NODE_RGB96.npy', 'CF_CONTEXT_RGB224_F32.npy', 'GROUP_NODE_VALID96.npy',
             'GROUP_CONTEXT_VALID224.npy', 'PAIR_LAYER_RGB96.npy', 'PAIR_ALPHA_PACKBITS.npy')
    e['payload_whitelist'] = [{k: c['outputs'][n][k] for k in ('path', 'sha256', 'size_bytes')} for n in names]
    e['save_selected_native_PNGs'] = True
    scope = {'schema': 'V242_FIXED_PREFIT_GALLERY_ROOT_SCOPE', 'approved_V242_SOURCE_gallery': True,
             'dataset_complete': e['dataset_complete'], 'data_QA': e['data_QA'],
             'code': e['code'], 'implementation_protocol': e['implementation_protocol'],
             'source_image_whitelist': e['source_image_whitelist'],
             'payload_whitelist': e['payload_whitelist'],
             'selection': 'Minimum parent_sample_index per original family x renewed proof PASS/UNKNOWN; empty bucket retained; maximum4. Seal caseIDs before any pixel access.',
             'selected_native_images_maximum': 12, 'model_output_or_ranking_used': False,
             'teacher_reference_overlay_display_only': True,
             'current_or_held_payloads_authorized': False, 'training_or_model_forward': False,
             'original_q_L_native_U_unchanged': True, 'automatic_retry': False}
    scope_path = R / 'V242_ROOT_SCENE_CF/READ_SCOPE_BEFORE_FIXED_PREFIT_GALLERY.json'
    save(scope_path, scope)
    e['root_scope'] = binding(scope_path)
    e['metadata_whitelist'] += [e['root_scope'], e['dataset_complete'], e['data_QA'], e['pair_metadata']]
    assert not safe(e['output_directory']).exists()
    e['approved_V242_SOURCE_gallery'] = True
    out = R / 'V242_ROOT_SCENE_CF/ACTUAL_FIXED_PREFIT_GALLERY_ENTRY.json'
    save(out, e)
    print(json.dumps({'completed': True, 'PASS': True, 'entry': binding(out),
                      'actual_pixels_read_during_assembly': 0}))


if __name__ == '__main__':
    main()
