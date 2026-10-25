"""Replace only the owned, unstaged V240 package with its verified V2 superset."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def safe(rel):
    assert not any(x in rel.lower() for x in ('v112', '__pycache__', '.pyc'))
    p = PurePosixPath(rel)
    assert not p.is_absolute() and '..' not in p.parts
    return rel


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def main():
    pb = (R / 'V240_MAIN_PREPARATION.json').read_bytes()
    assert sha(pb) == '331d675b205b6f8bf39cb6cb49dd5b4fdcb8c2d80370aa0e28571770a78f1f98'
    previous = json.loads(pb)
    assert git('rev-parse', 'HEAD').decode().strip() == previous['parent_commit']
    assert not git('diff', '--cached', '--name-only', '-z')
    prior_by_path = {safe(f['path']): f for f in previous['copied']}
    for row in previous['copied'] + previous['entry_points']:
        p = REPO / safe(row['path'])
        assert not p.is_symlink()
        b = p.read_bytes()
        assert (len(b), sha(b)) == (row['bytes'], row['sha256'])
    package = R / 'V240_PUBLICATION_PACKAGE_V2'
    mb = (package / 'MANIFEST.json').read_bytes()
    assert sha(mb) == '3204602a3a2abe306a02e65e95fd13c7759bf6cac40679dd37dc6a1e414df5fa'
    manifest = json.loads(mb)
    dest = Path(previous['destination'])
    assert dest == REPO / '_docs/experiments/pallet-keypoint-visibility/V240_CANONICAL_CONTEXT_ACTUAL_RESULTS'
    copied = []
    for row in manifest['files'] + [{'path': 'MANIFEST.json', 'bytes': len(mb), 'sha256': sha(mb)}]:
        rel = safe(row['path']); source = package / rel
        assert not source.is_symlink()
        b = source.read_bytes()
        assert (len(b), sha(b)) == (row['bytes'], row['sha256'])
        target = dest / rel; public_rel = target.relative_to(REPO).as_posix()
        if target.exists():
            assert public_rel in prior_by_path and not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b)
        copied.append({'path': public_rel, 'sha256': sha(b), 'bytes': len(b)})
    assert set(prior_by_path) <= {r['path'] for r in copied}
    out = {**previous, 'package': str(package), 'copied': copied,
           'manifest_sha256': sha(mb), 'publication_revision': 2,
           'first_preparation_sha256': sha(pb), 'entrypoints_changed_after_first_preparation': False,
           'original_failed_package_preserved': True, 'no_unlisted_path_removed_or_staged': True}
    p = R / 'V240_MAIN_PREPARATION_V2.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'copied': len(copied),
                      'entrypoints': len(out['entry_points']), 'receipt_sha256': sha(p.read_bytes())}))


if __name__ == '__main__':
    main()
