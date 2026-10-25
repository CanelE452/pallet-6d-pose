"""Bind independent saved SOURCE feature verification; no array/model body reads."""
from pathlib import Path
import hashlib
import json
import datetime

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V239_FEATURE_PACKET_QA_C'


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).lower() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation'))
    assert not p.is_symlink()
    return p


def binding(p):
    p = safe(p); b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def write(p, d):
    with safe(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')
    return binding(p)


def main():
    ready = binding(H / 'FINAL_READY.json')
    assert ready['sha256'] == '00565da413d4fea76d57d7e31a0eacd54f4f9948483b58cb2261345fad95b8ad'
    rd = json.loads(Path(ready['path']).read_bytes())
    assert rd['completed'] and rd['PASS'] and rd['prepared_only']
    for name in ('code', 'protocol', 'feature_false_entry'):
        assert binding(rd[name]['path'])['sha256'] == rd[name]['sha256']
    cb = binding(R / 'V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/COMPLETE.json')
    assert cb['sha256'] == 'e2e796ab5c0580b95204292f1f4ecb90d30842060d9b048eab6c41f942cfc392'
    packet = json.loads(Path(cb['path']).read_bytes())
    assert packet['completed'] and packet['PASS'] and len(packet['features']) == 7
    e = json.loads(Path(rd['feature_false_entry']['path']).read_bytes())
    assert e['approved_independent_SOURCE_feature_packet_QA'] is False
    scope = {'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_only': True, 'feature_complete': cb, 'C_preparation': ready,
             'allowed': 'Exact saved7 SOURCE feature arrays + original samples/geometry nonpixel + all15040 map; opaque SHA/header and byte/numeric sharing/support/role QA only.',
             'not_allowed': 'RGB/valid caches, original raw images, encoder/official weights, head/optimizer, current/held payload or new selection.',
             'original_q_L_native_U_and_FIT_DEV_preserved': True,
             'new_current_or_held_payload': 0, 'model_adopted': False,
             'automatic_retry': False}
    sb = write(R / 'V239_ROOT_CANONICAL_CONTEXT/ACTUAL_INDEPENDENT_FEATURE_QA_SCOPE.json', scope)
    e['approved_independent_SOURCE_feature_packet_QA'] = True
    e['root_scope'] = sb; e['feature_complete'] = cb; e['features'] = packet['features']
    e['metadata_whitelist'] += [cb, sb, packet['canonical_sharing_QA_binding']]
    e['payload_whitelist'] += list(packet['features'].values())
    assert not safe(e['output_directory']).exists()
    eb = write(R / 'V239_ROOT_CANONICAL_CONTEXT/ACTUAL_INDEPENDENT_FEATURE_QA_ENTRY.json', e)
    print(json.dumps({'completed': True, 'actual_entry': eb, 'actual_payload_read': False}))


if __name__ == '__main__':
    main()
