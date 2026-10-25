"""Explicit V242 data / V243 prep / V244 frozen behavior publication.

Only the sealed inventory and the named diagnostic directories are copied.
Huge SOURCE image caches are excluded using saved bindings, before leaf access.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
OUT = R / 'V244_PUBLICATION_PACKAGE'
EXTRA = ('V242_PUBLICATION_INVENTORY_C', 'V244_ROOT_FROZEN_BEHAVIOR',
         'V244_OWN_BRANCH_DIAGNOSTIC_A', 'V244_OWN_BRANCH_CODE_REVIEW_C',
         'V244_V235_PAIR_DIAGNOSTIC_B', 'V244_V235_PAIR_CODE_REVIEW_A',
         'V244_SAVED_OUTPUT_QA_C', 'V244_PUBLICATION_CODE_REVIEW_B')
INVENTORY_SHA = '59b087b117f29fa3a56793482168fce3f571c088609747f69d911b598dd8f04e'


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(v in str(p).casefold() for v in ('v112', '__pycache__', '.pyc'))
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink()
    return p


def sha(b):
    return hashlib.sha256(b).hexdigest()


def leaves(root):
    for name in sorted(os.listdir(safe(root))):
        if any(v in name.casefold() for v in ('v112', '__pycache__', '.pyc')):
            continue
        p = safe(root / name)
        if p.is_dir():
            yield from leaves(p)
        else:
            assert p.is_file()
            yield p


def write(rel, b):
    relpath = PurePosixPath(rel)
    assert not relpath.is_absolute() and '..' not in relpath.parts
    p = safe(OUT / rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(b)


def jsave(rel, obj):
    write(rel, (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode())


def main():
    inventory_bytes = safe(R / 'V242_PUBLICATION_INVENTORY_C/CANDIDATE_FILES.json').read_bytes()
    assert sha(inventory_bytes) == INVENTORY_SHA
    inventory = json.loads(inventory_bytes)
    status = json.loads(safe(R / 'V244_ROOT_FROZEN_BEHAVIOR/PUBLICATION_STATUS.json').read_bytes())
    assert status['V242_data_QA_PASS'] and status['V244_both_frozen_diagnostics_complete']
    assert status['new_training_updates'] == 0 and status['new_current_or_held_payloads'] == 0
    assert status['model_adopted'] is False
    assert not safe(OUT).exists()
    OUT.mkdir()
    source_map = []
    copied = set()

    def copy(src, dst, expected=None, size=None):
        assert dst not in copied
        copied.add(dst)
        b = safe(src).read_bytes()
        assert len(b) < 95_000_000
        if expected:
            assert sha(b) == expected
        if size is not None:
            assert len(b) == size
        row = {'path': dst, 'source_path_provenance_only': str(src),
               'source_sha256': sha(b), 'source_bytes': len(b), 'encoding': 'identity'}
        # These leaf names are not Markdown/HTML link targets in original reports.
        if src.name in ('PAIR_METADATA.jsonl', 'ALL7617_OWN_BRANCH_ROWS.jsonl'):
            row['path'] += '.gz'
            row['encoding'] = 'gzip'
            b = gzip.compress(b, compresslevel=9, mtime=0)
        write(row['path'], b)
        source_map.append(row)

    for row in inventory['candidates']:
        rel = row['R_relative_path']
        assert str(safe(R / rel)) == row['path']
        copy(R / rel, row['proposed_package_relative_path'], row['sha256'], row['size_bytes'])
    # The invented fixture image was deliberately not read during inventory.
    toy = R / 'V242_SCENE_GALLERY_A/INVENTED_FIRST_ATTEMPT_CONSTRUCTED_UNKNOWN_CARD.png'
    copy(toy, 'artifacts/' + toy.relative_to(R).as_posix())
    for top in EXTRA:
        for src in leaves(R / top):
            copy(src, 'artifacts/' + src.relative_to(R).as_posix())
    for src, dst in ((Path(__file__), 'build_v244_publication.py'),
                     (R / 'publish_v244_main.py', 'publish_v244_main.py'),
                     (R / 'verify_v244_saved_outputs_portable.py', 'verify_saved_outputs.py'),
                     (R / 'V233_PUBLICATION_VERIFY_A/verify_package.py', 'verify_package.py')):
        copy(src, dst)
    notice = safe(R / 'V240_PUBLICATION_NOTICES.md').read_bytes()
    write('THIRD_PARTY_NOTICES.md', notice + b'\nV242 adds four fixed SOURCE scene examples (synthetic/YCB), native-scene derivative composites, exact code and QA. V244 adds frozen SOURCE diagnostic outputs/features; no new training and no official DINO weights redistribution. Prior notice counts describe the earlier package.\n')
    write('README_KO.md', safe(R / 'V244_ROOT_FROZEN_BEHAVIOR/PUBLICATION_README_KO.md').read_bytes())
    jsave('STATUS.json', status)
    jsave('LARGE_CACHE_EXCLUSIONS.json', json.loads(safe(R / 'V242_PUBLICATION_INVENTORY_C/EXCLUSIONS.json').read_bytes()))
    jsave('PUBLICATION_SOURCE_MAP.json', {'files': source_map,
          'absolute_pointers_are_provenance_only': True,
          'huge_excluded_cache_leaf_access': 0,
          'private_or_current_or_held_payloads': 0,
          'old_frozen_sources_changed': False})
    rows = []
    for p in leaves(OUT):
        b = p.read_bytes()
        rows.append({'path': p.relative_to(OUT).as_posix(), 'sha256': sha(b), 'bytes': len(b)})
    jsave('MANIFEST.json', {'schema': 'V244_DATA_AND_FROZEN_DIAGNOSIS_PUBLICATION',
          'files': rows, 'file_count': len(rows), 'total_bytes': sum(r['bytes'] for r in rows),
          'scientific_status': status, 'publication_is_not_model_success': True})
    print(json.dumps({'completed': True, 'PASS': True, 'files': len(rows),
          'bytes': sum(r['bytes'] for r in rows), 'manifest_sha256': sha((OUT / 'MANIFEST.json').read_bytes())}))


if __name__ == '__main__':
    main()
