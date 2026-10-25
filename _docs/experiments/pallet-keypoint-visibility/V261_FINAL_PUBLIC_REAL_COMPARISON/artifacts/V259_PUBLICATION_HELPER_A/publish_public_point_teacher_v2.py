"""Opaque allowlisted public audit packaging/staging. No loaders, commit or push."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import stat
import subprocess
import sys

R = Path(__file__).absolute().parent.parent
PACKAGE = R / 'V259_PUBLIC_POINT_TEACHER_PACKAGE'
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST = '_docs/experiments/pallet-keypoint-visibility/V259_PUBLIC_DEPTH_POINT_TEACHER_PILOT'
PARENT = '3a60c7571019ed59c864736b7e7f09c01513e8e2'
MAX_LEAF = 90 * 1024 * 1024
CHUNK = 1024 * 1024
REGISTRIES = ('readme.md', '_docs/experiments/pallet-keypoint-visibility/README.md',
              '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md')


def safe(value):
    p = Path(value)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for ancestor in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(ancestor.lstat().st_mode), str(ancestor)
        except FileNotFoundError:
            pass
    return p


def relative(value):
    assert type(value) is str and value and '\\' not in value and '\0' not in value
    assert not any(t in value.casefold() for t in ('v112', 'cache', '.pyc'))
    p = PurePosixPath(value)
    assert not p.is_absolute() and '..' not in p.parts and '.' not in p.parts
    assert str(p) == value and value != '.'
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def write_new(path, data):
    p = safe(path)
    safe(p.parent).mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)


def validate_binding(value):
    assert isinstance(value, dict) and {'path', 'sha256', 'size_bytes'} <= set(value)
    p = safe(value['path'])
    assert type(value['size_bytes']) is int and 0 <= value['size_bytes'] <= MAX_LEAF
    assert type(value['sha256']) is str and len(value['sha256']) == 64
    assert all(c in '0123456789abcdef' for c in value['sha256'])
    return p


def measure(path):
    p = safe(path)
    h, n = hashlib.sha256(), 0
    with p.open('rb') as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': n}


def copy_bound(binding, destination):
    source, target = validate_binding(binding), safe(destination)
    safe(target.parent).mkdir(parents=True, exist_ok=True)
    h, n = hashlib.sha256(), 0
    with source.open('rb') as src, target.open('xb') as dst:
        while chunk := src.read(CHUNK):
            n += len(chunk)
            assert n <= binding['size_bytes'], str(source)
            h.update(chunk)
            dst.write(chunk)
    assert n == binding['size_bytes'] and h.hexdigest() == binding['sha256'], str(source)


def validated_entry(path):
    p = safe(path)
    raw = p.read_bytes()
    entry = json.loads(raw)
    assert entry['RootapprovedPublicPointTeacherPublication'] is True
    assert entry['helper_sha256'] == sha(safe(Path(__file__).absolute()).read_bytes())
    assert entry['output_directory'] == str(PACKAGE)
    assert entry['repo_directory'] == str(REPO)
    assert entry['destination_relative_path'] == DEST
    assert entry['parent_commit'] == PARENT
    safe(entry['stage_receipt_directory'])
    rows = entry['source_files']
    assert isinstance(rows, list) and rows
    seen = set()
    for row in rows:
        rel = relative(row['relative_path'])
        assert str(rel) not in ('MANIFEST.json', 'FAILURE.json', 'STARTED.json')
        assert str(rel) not in seen
        seen.add(str(rel))
        source = validate_binding(row['source'])
        assert source != p and not source.is_relative_to(PACKAGE)
    assert 'README_KO.md' in seen, 'Registry link target must be an explicit source.'
    return entry, raw, {'path': str(p), 'sha256': sha(raw), 'size_bytes': len(raw)}


def build(entry, raw, entry_binding):
    out = safe(entry['output_directory'])
    assert not out.exists(), 'Fresh package required; preserve existing output.'
    out.mkdir()
    try:
        write_new(out / 'STARTED.json', json_bytes({'entry': entry_binding, 'mode': 'opaque package'}))
        files = []
        for row in entry['source_files']:
            rel, source = row['relative_path'], row['source']
            copy_bound(source, out / relative(rel))
            files.append({'relative_path': rel, 'sha256': source['sha256'],
                          'size_bytes': source['size_bytes'], 'original_source': source})
        started = measure(out / 'STARTED.json')
        files.append({'relative_path': 'STARTED.json', 'sha256': started['sha256'],
                      'size_bytes': started['size_bytes'], 'generated_metadata': True})
        manifest = {'completed': True, 'PASS': True, 'schema': 'V259_OPAQUE_PUBLIC_PACKAGE_V1',
            'original_entry': {'binding': entry_binding, 'utf8': raw.decode('utf-8')},
            'files': files, 'source_file_count': len(entry['source_files']),
            'total_file_bytes': sum(row['size_bytes'] for row in files),
            'raw_bytes_preserved': True, 'payload_decoders_or_loaders_used': False,
            'training_or_model_forward': 0, 'commit_or_push_performed': False}
        write_new(out / 'MANIFEST.json', json_bytes(manifest))
        return {'completed': True, 'PASS': True, 'manifest': measure(out / 'MANIFEST.json'),
                'files': len(files), 'payload_decoders_or_loaders_used': False}
    except BaseException as exc:
        write_new(out / 'FAILURE.json', json_bytes({'completed': False, 'PASS': False,
            'mode': 'package', 'error_type': type(exc).__name__, 'error': str(exc),
            'partial_output_preserved': True, 'automatic_retry': False}))
        raise


def git(*args, inp=None):
    return subprocess.run(['git', '--literal-pathspecs', *args], cwd=safe(REPO), input=inp,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)


def verify_manifest(entry, raw, entry_binding):
    mb = safe(PACKAGE / 'MANIFEST.json').read_bytes()
    manifest = json.loads(mb)
    assert manifest['completed'] and manifest['PASS'] and manifest['schema'] == 'V259_OPAQUE_PUBLIC_PACKAGE_V1'
    assert manifest['original_entry'] == {'binding': entry_binding, 'utf8': raw.decode('utf-8')}
    expected = {r['relative_path']: r['source'] for r in entry['source_files']}
    rows = manifest['files']
    assert len(rows) == len(expected) + 1
    assert len({r['relative_path'] for r in rows}) == len(rows)
    assert {r['relative_path'] for r in rows} == {*expected, 'STARTED.json'}
    for row in rows:
        rel = relative(row['relative_path'])
        got = measure(PACKAGE / rel)
        assert got['sha256'] == row['sha256'] and got['size_bytes'] == row['size_bytes']
        if str(rel) in expected:
            b = expected[str(rel)]
            assert row['original_source'] == b
            assert row['sha256'] == b['sha256'] and row['size_bytes'] == b['size_bytes']
        else:
            expected_started = json_bytes({'entry': entry_binding, 'mode': 'opaque package'})
            assert str(rel) == 'STARTED.json' and row['generated_metadata'] is True
            assert got['sha256'] == sha(expected_started) and got['size_bytes'] == len(expected_started)
    assert manifest['source_file_count'] == len(expected)
    assert manifest['total_file_bytes'] == sum(r['size_bytes'] for r in rows)
    return manifest, mb


def registry_prefix(index):
    link = (DEST + '/README_KO.md') if index == 0 else 'V259_PUBLIC_DEPTH_POINT_TEACHER_PILOT/README_KO.md'
    return ('> [V259: 공개 depth point teacher pilot](' + link + ') — 공개 2프레임의 D/E/U 교사 라벨을 점검했습니다. '
            '모델 정확도 결과가 아니며 학습 0회입니다. 최신 모델 비교는 V256입니다.\n\n').encode('utf-8')


def stage(entry, raw, entry_binding):
    receipt_dir = safe(entry['stage_receipt_directory'])
    assert not receipt_dir.exists(), 'Fresh stage receipt directory required.'
    assert not receipt_dir.is_relative_to(PACKAGE) and not receipt_dir.is_relative_to(REPO)
    receipt_dir.mkdir(parents=True)
    try:
        write_new(receipt_dir / 'STARTED.json', json_bytes({'entry': entry_binding, 'mode': 'stage'}))
        assert git('rev-parse', 'HEAD').stdout.decode().strip() == PARENT
        remote = git('ls-remote', '--exit-code', 'origin', 'refs/heads/main').stdout.decode().splitlines()
        assert remote == [PARENT + '\trefs/heads/main'], remote
        assert git('status', '--porcelain', '--untracked-files=no').stdout == b''
        git('diff', '--quiet')
        git('diff', '--cached', '--quiet')
        manifest, mb = verify_manifest(entry, raw, entry_binding)
        destination = safe(REPO / DEST)
        assert not destination.exists(), 'Fresh destination required; no overwrite.'
        destination.mkdir(parents=True)
        expected = {}
        check_paths = list(REGISTRIES)
        for row in manifest['files']:
            rel = relative(row['relative_path'])
            binding = {'path': str(PACKAGE / rel), 'sha256': row['sha256'], 'size_bytes': row['size_bytes']}
            copy_bound(binding, destination / rel)
            expected[str(PurePosixPath(DEST) / rel)] = binding
            if len(rel.parts) == 1 or rel.parts[0] == 'tools':
                check_paths.append(str(PurePosixPath(DEST) / rel))
        write_new(destination / 'MANIFEST.json', mb)
        expected[DEST + '/MANIFEST.json'] = {'sha256': sha(mb), 'size_bytes': len(mb)}
        check_paths.append(DEST + '/MANIFEST.json')
        preserved = {}
        for index, rel in enumerate(REGISTRIES):
            p = safe(REPO / relative(rel))
            old = git('show', 'HEAD:' + rel).stdout
            assert p.read_bytes() == old
            prefix = registry_prefix(index)
            new = prefix + old
            p.write_bytes(new)
            assert p.read_bytes()[len(prefix):] == old
            expected[rel] = {'sha256': sha(new), 'size_bytes': len(new)}
            preserved[rel] = {'old_sha256': sha(old), 'old_size_bytes': len(old),
                              'prefix_size_bytes': len(prefix), 'new_sha256': sha(new)}
        allowlist = sorted(expected)
        git('add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul',
            inp=b'\0'.join(p.encode('utf-8') for p in allowlist) + b'\0')
        staged = git('diff', '--cached', '--name-only', '-z').stdout.decode().split('\0')
        assert staged[-1] == '' and sorted(staged[:-1]) == allowlist
        for rel, target in expected.items():
            current = measure(REPO / relative(rel))
            assert current['sha256'] == target['sha256'] and current['size_bytes'] == target['size_bytes']
            blob = git('cat-file', 'blob', ':' + rel).stdout
            assert sha(blob) == target['sha256'] and len(blob) == target['size_bytes']
            assert blob == safe(REPO / relative(rel)).read_bytes(), rel
        git('diff', '--cached', '--check', '--', *sorted(set(check_paths)))
        tree = git('write-tree').stdout.decode().strip()
        assert len(tree) == 40 and all(c in '0123456789abcdef' for c in tree)
        result = {'completed': True, 'PASS': True, 'entry': entry_binding,
            'package_manifest': {'path': str(PACKAGE / 'MANIFEST.json'), 'sha256': sha(mb), 'size_bytes': len(mb)},
            'parent_HEAD_and_remote_main': PARENT, 'staged_tree': tree,
            'staged_files': [{'relative_path': p, 'sha256': expected[p]['sha256'],
                              'size_bytes': expected[p]['size_bytes']} for p in allowlist],
            'all_staged_and_working_bytes_verified': True, 'registry_preserved_tails': preserved,
            'whitespace_checked_paths': sorted(set(check_paths)),
            'archived_artifacts_assets_references_preserved_without_whitespace_normalization': True,
            'commit_or_push_performed': False, 'force_push': False}
        write_new(receipt_dir / 'STAGED_QA.json', json_bytes(result))
        return {'completed': True, 'PASS': True, 'staged_tree': tree, 'staged_files': len(allowlist),
                'receipt': measure(receipt_dir / 'STAGED_QA.json'), 'commit_or_push_performed': False}
    except BaseException as exc:
        failure = {'completed': False, 'PASS': False, 'mode': 'stage',
            'error_type': type(exc).__name__, 'error': str(exc),
            'partial_copies_registry_or_index_state_preserved': True, 'automatic_retry': False,
            'commit_or_push_performed': False}
        if isinstance(exc, subprocess.CalledProcessError):
            failure.update(returncode=exc.returncode,
                stdout=exc.stdout.decode('utf-8', errors='replace'),
                stderr=exc.stderr.decode('utf-8', errors='replace'))
        write_new(receipt_dir / 'FAILURE.json', json_bytes(failure))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--mode', required=True, choices=('package', 'stage'))
    args = parser.parse_args()
    entry, raw, binding = validated_entry(args.entry)
    result = build(entry, raw, binding) if args.mode == 'package' else stage(entry, raw, binding)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
