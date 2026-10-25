"""Copy and stage an exact public-only non-frame geometry audit allowlist."""
from pathlib import Path
import hashlib
import json
import stat
import subprocess

R = Path(__file__).absolute().parent
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST = Path('_docs/experiments/pallet-keypoint-visibility/V257_PUBLIC_GEOMETRY_FORMAT')


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(a.lstat().st_mode), str(a)
        except FileNotFoundError:
            pass
    return p


def read(p):
    return safe(p).read_bytes()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*args, inp=None):
    return subprocess.run(['git', *args], cwd=safe(REPO), input=inp,
                          capture_output=True, check=True).stdout


def create(p, b):
    p = safe(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(b)


def run(entry_path):
    eb = read(entry_path)
    e = json.loads(eb)
    assert e['approved_public_nonframe_geometry_publication'] is True
    assert e['code_sha256'] == sha(read(Path(__file__)))
    assert e['training_or_new_evaluation'] is False
    assert git('rev-parse', 'HEAD').decode().strip() == e['parent']
    assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == e['parent']
    assert not git('status', '--porcelain', '-uno')
    pkg = safe(Path(e['output_directory']))
    assert not pkg.exists() and not safe(REPO / DEST).exists()
    pkg.mkdir()
    rows, contents = [], {}
    for item in e['files']:
        rel = Path(item['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        assert str(rel) not in contents
        b = read(Path(item['source']['path']))
        assert sha(b) == item['source']['sha256'] and len(b) == item['source']['size_bytes']
        assert len(b) < 90 * 1024 * 1024
        contents[str(rel)] = b
        create(pkg / rel, b)
        rows.append({'relative_path': str(rel), 'sha256': sha(b), 'size_bytes': len(b),
                     'original_source_binding_inert': item['source']})
    manifest = {'completed': True, 'PASS': True, 'files': rows,
                'root_entry_SHA256': sha(eb), 'root_entry_original_UTF8': eb.decode(),
                'all_source_and_packaged_bytes_equal': True,
                'array_model_or_frame_decode': 0, 'training_or_new_evaluation': False}
    mb = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    create(pkg / 'MANIFEST.json', mb)
    contents['MANIFEST.json'] = mb
    changed = {str(DEST / rel): b for rel, b in contents.items()}
    refs = {
        'readme.md': str(DEST / 'README_KO.md'),
        '_docs/experiments/pallet-keypoint-visibility/README.md': 'V257_PUBLIC_GEOMETRY_FORMAT/README_KO.md',
        '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md': 'V257_PUBLIC_GEOMETRY_FORMAT/README_KO.md',
    }
    preserved = {}
    for rel, url in refs.items():
        old = read(REPO / rel)
        prefix = ('> [V257：공개 pallet 보정값·mesh 형식 확인](' + url + ') — 공식 K·OBJ 원본과 독립 검산을 보존했습니다. 새 학습·정확도 평가 0이며, native 대응·투영·point 가림 라벨은 아직 미확정입니다. 모델 비교는 아래 V256 링크에 있습니다.\n\n').encode()
        changed[rel] = prefix + old
        preserved[rel] = {'old_sha256': sha(old), 'old_size_bytes': len(old),
                          'prefix_size_bytes': len(prefix)}
    for rel, b in changed.items():
        if rel in refs:
            safe(REPO / rel).write_bytes(b)
        else:
            create(REPO / rel, b)
    for rel, v in preserved.items():
        tail = read(REPO / rel)[v['prefix_size_bytes']:]
        assert sha(tail) == v['old_sha256'] and len(tail) == v['old_size_bytes']
    allowed = sorted(changed)
    git('add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul',
        inp=b'\0'.join(p.encode() for p in allowed) + b'\0')
    actual = git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0')
    assert sorted(actual) == allowed
    for rel, b in changed.items():
        assert git('show', ':' + rel) == b
    # Raw publisher calibration/OBJ/README/API are byte-preserved archives.
    style_paths = [p for p in allowed if '/assets/' not in p and '/references/' not in p]
    git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *style_paths)
    receipt = {'completed': True, 'PASS': True, 'parent': e['parent'],
               'staged_tree': git('write-tree').decode().strip(),
               'files': [{'relative_path': p, 'sha256': sha(b), 'size_bytes': len(b)}
                         for p, b in sorted(changed.items())],
               'registry_preserved_suffixes': preserved,
               'all_staged_blobs_equal_exact_allowlist_bytes': True,
               'style_checked_paths': style_paths,
               'publisher_archive_bytes_not_normalized': True,
               'commit_or_push_performed': False, 'force_push': False}
    create(R / 'V257_STAGED_BLOB_QA.json',
           (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({'completed': True, 'PASS': True, 'staged_files': len(allowed),
                      'package_files': len(contents), 'staged_tree': receipt['staged_tree']}, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', type=Path, required=True)
    run(ap.parse_args().entry)
