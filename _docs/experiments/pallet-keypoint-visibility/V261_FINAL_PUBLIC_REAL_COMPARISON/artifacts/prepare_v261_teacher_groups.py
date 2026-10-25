"""Root metadata-only group entries for the already fixed public284 scope."""
from pathlib import Path
import argparse
import hashlib
import json

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V261_ROOT_PUBLIC_BATCH_TEACHER'
EXPECTED = {'0.55m_hall': 28, '0.55m_lab': 52, '1.23m_hall': 48,
            '1.23m_lab': 60, '1.92m_hall': 40, '1.92m_lab': 56}


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).casefold() for x in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        assert not a.is_symlink(), str(a)
    return p


def binding(p):
    b = safe(p).read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def read_exact(p, expected):
    p = safe(p)
    b = p.read_bytes()
    assert hashlib.sha256(b).hexdigest() == expected
    return json.loads(b)


def basic(b):
    p = safe(b['path'])
    assert len(b['sha256']) == 64 and type(b['size_bytes']) is int
    return {'path': str(p), 'sha256': b['sha256'], 'size_bytes': b['size_bytes']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    a = ap.parse_args()
    assert len(a.groups) == len(set(a.groups)) and set(a.groups) <= set(EXPECTED)
    receipt = read_exact(a.receipt, a.sha256)
    downloads = receipt['downloads'] if receipt.get('PASS') else receipt['completed_downloads']
    files = {x['file_id']: x for x in downloads}
    assert len(files) == len(downloads)
    metadata = read_exact(R / 'V260_PUBLIC_BATCH_METADATA_C/metadata_v1/ALL284_FOUR_WAY_METADATA_MANIFEST.json',
                          'a6ee4661700af0d3d4962e2aa3b978bd4f42d676779b4328955ba0bcbaa607ef')
    split = read_exact(R / 'V260_ROOT_PUBLIC_BATCH/PROSPECTIVE_BATCH_SCOPE_AND_SPLIT.json',
                      '55ef69da9044616761452a3db17b346777ea061d6a3242e068769006fa60ca68')
    held = read_exact(R / 'V258_PUBLIC_HELD_DIGEST_IDENTITY_B/HELD45_ORIGINAL_RGB_BYTE_DIGEST_ALLOWLIST.json',
                     'eebb382e9512b1471ba443d17f42ae0c90d6d7adf5dfd65d18aeace39c10fabd')
    exclusions = set(held['exact_encoded_byte_SHA256_exclusion_set'])
    assert len(exclusions) == 45 and all(type(s) is str and len(s) == 64 for s in exclusions)
    template = read_exact(R / 'V260_PUBLIC_BATCH_TEACHER_A/ENTRY_FALSE_TEMPLATE.json',
                         'dde5f0b351c22ecb74d57d57da212ce91282ca876e02a3dbc92a57021674e6cf')
    mesh = binding(R / 'V257_ROOT_PUBLIC_PALLET_GEOMETRY/pallox.obj')
    assert mesh['sha256'] == '2c44a016ec657955af2d195876af1234fab143dc1651203d13bad1c44c836e0f'
    safe(H).mkdir(exist_ok=True)
    results = []
    for group in a.groups:
        rows = [r for r in metadata['complete_bundles'] if r['coarse_capture_group_id'] == group]
        assert len(rows) == EXPECTED[group]
        frames = []
        for row in rows:
            inputs = {}
            for role, official in [*row['files'].items(), ('K', row['K'])]:
                f = files[official['file_id']]
                assert f['publisher_relative_path'] == official['publisher_relative_path']
                assert f['publisher_MD5_verified'] == official['publisher_MD5']
                assert f['size_bytes'] == official['size_bytes']
                inputs[role] = basic(f)
            assert inputs['RGB']['sha256'] not in exclusions, 'Protected exact encoded-byte identity hit; stop, do not filter.'
            inputs['OBJ'] = mesh
            frames.append({'frame_id': row['publisher_frame_path'], 'inputs': inputs,
                           'provenance': {'publisher_frame_path': row['publisher_frame_path'],
                                          'coarse_group_id': group,
                                          'source_role': split['adopted_group_roles'][group],
                                          'raw_receipt': binding(a.receipt),
                                          'encoded_identity_hits': 0,
                                          'reencoded_capture_disjointness_certified': False}})
        entry = dict(template)
        entry['approved_public_surface_batch_teacher'] = True
        entry['frames'] = frames
        stem = group.replace('.', 'p')
        entry['output_directory'] = str(H / ('group_' + stem + '_v1'))
        entry['Root_chunking_scope'] = 'Compute-only chunks of the fixed284; all six groups must be present before features/training. No data/model/pose/threshold selection.'
        p = safe(H / ('GROUP_' + stem + '_ENTRY.json'))
        with p.open('x') as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
            f.write('\n')
        results.append({'group': group, 'frames': len(frames), 'entry': binding(p),
                        'output_directory': entry['output_directory']})
    print(json.dumps({'prepared': results, 'same_frozen_teacher_policy': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
