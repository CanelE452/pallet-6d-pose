"""Exact declared package + three raw registry prefixes; stage only.

Based on V259 publish_public_point_teacher_v2.py. No commit, push, model or
payload parsing. Root must first fully restore and hash-check the package.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

R = Path(__file__).absolute().parent.parent
PACKAGE = R/'V261_FINAL_PUBLICATION_PACKAGE'
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST = '_docs/experiments/pallet-keypoint-visibility/V261_FINAL_PUBLIC_REAL_COMPARISON'
PARENT = 'ff18ff3c489cd49a64956aaeab25b01278519f65'
SCHEMA = 'V261_FINAL_PUBLIC_REAL_COMPARISON_OPAQUE_BYTES_V1'
REGISTRIES = ('readme.md', '_docs/experiments/pallet-keypoint-visibility/README.md',
              '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md')
CHUNK = 1024*1024
MAX_LEAF = 90*1024*1024


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(value, exists=True):
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts and
         not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc')), 'Path rejected before FS')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact artifact absent')
            continue
        need(not stat.S_ISLNK(mode), 'No artifact ancestor or leaf symlink')
    return p


def relative(value):
    need(type(value) is str and value and '\\' not in value and '\0' not in value and
         not any(t in value.casefold() for t in ('v112', 'cache', '.pyc')), 'Relative lexical path')
    p = PurePosixPath(value)
    need(not p.is_absolute() and '..' not in p.parts and p.as_posix() == value and value != '.',
         'Canonical package or Git relative path')
    return Path(*p.parts)


def open_read(path):
    return os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def measure(path):
    h, n = hashlib.sha256(), 0
    with open_read(path) as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
            n += len(chunk)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': n}


def read_bound(b):
    with open_read(b['path']) as f:
        data = f.read()
    need(len(data) == b['size_bytes'] and sha(data) == b['sha256'], 'Exact bound metadata')
    return data


def write_new(path, data):
    p = safe(path, False)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)


def json_bytes(v):
    return (json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')


def copy_bound(b, destination):
    need(type(b['size_bytes']) is int and 0 <= b['size_bytes'] <= MAX_LEAF, 'Physical leaf limit90MiB')
    target = safe(destination, False)
    target.parent.mkdir(parents=True, exist_ok=True)
    h, n = hashlib.sha256(), 0
    with open_read(b['path']) as src, target.open('xb') as dst:
        for chunk in iter(lambda: src.read(CHUNK), b''):
            n += len(chunk)
            need(n <= b['size_bytes'], 'Source larger than declared physical leaf')
            h.update(chunk)
            dst.write(chunk)
    need(n == b['size_bytes'] and h.hexdigest() == b['sha256'], 'Opaque copy exact SHA/size')


def git(*args, inp=None):
    return subprocess.run(['git', '--literal-pathspecs', *args], cwd=safe(REPO), input=inp,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)


def validated_entry(path):
    with open_read(path) as f:
        raw = f.read()
    e = json.loads(raw)
    need(e.get('approved_actual_publication_stage') is True and
         e.get('automatic_retry') is False and e.get('commit_or_push') is False,
         'Root explicit stage-only entry')
    need(e['code'] == measure(Path(__file__).absolute()), 'Pinned V261 staging code')
    need(e['repo_directory'] == str(REPO) and e['package_directory'] == str(PACKAGE) and
         e['destination_relative_path'] == DEST and e['parent_commit'] == PARENT, 'Fixed V261 route/parent')
    scope = json.loads(read_bound(e['root_scope']))
    need(scope.get('approved_actual_publication_stage') is True and
         scope.get('entry_contract') == {k: v for k, v in e.items() if k != 'root_scope'},
         'Exact Root stage contract')
    need(set(e['registry_prefixes']) == set(REGISTRIES), 'Exactly the existing three registries')
    for rel, b in e['registry_prefixes'].items():
        value = b['utf8'].encode('utf-8')
        need(bool(value) and sha(value) == b['sha256'] and len(value) == b['size_bytes'],
             'Exact Root-supplied prefix bytes; no hardcoded scientific result')
        relative(rel)
    return e, {'path': str(path), 'sha256': sha(raw), 'size_bytes': len(raw)}


def verify_package_and_full_restore(e):
    need(e['package_manifest']['path'] == str(PACKAGE/'MANIFEST.json'), 'Fixed manifest path')
    mb = read_bound(e['package_manifest'])
    m = json.loads(mb)
    need(m.get('completed') is True and m.get('PASS') is True and m['schema'] == SCHEMA,
         'Completed sealed byte package')
    rows = m['files']
    need(len({r['relative_path'] for r in rows}) == len(rows) == m['file_count'] and
         sum(r['size_bytes'] for r in rows) == m['total_bytes'], 'Exact manifest accounting')
    for r in rows:
        rel = r['relative_path']
        need(rel != 'MANIFEST.json' and 0 <= r['size_bytes'] <= MAX_LEAF, 'Manifest self excluded/leaf limit')
        b = measure(PACKAGE/relative(rel))
        need(b['sha256'] == r['sha256'] and b['size_bytes'] == r['size_bytes'], 'Package physical bytes exact')
    qa = json.loads(read_bound(e['full_restore_QA']))
    need(qa.get('completed') is True and qa.get('PASS') is True and
         qa.get('package_manifest_sha256') == sha(mb) and
         qa.get('restored_original_bytes_exact') is True and
         qa.get('all_large_outputs_fully_restored') is True and
         qa.get('original_source_path_reads') == 0 and qa.get('array_or_model_decode') == 0,
         'Root full byte restoration before stage, no original fallback')
    expected = [{'relative_path': a['relative_path'], 'sha256': a['original_sha256'],
                 'size_bytes': a['original_size_bytes']} for a in m['artifacts']]
    need(qa['restored_artifacts'] == expected and qa['logical_artifacts_restored'] == len(expected) and
         qa['large_restored'] == sum(a['encoding'] == 'gzip_parts' for a in m['artifacts']),
         'Every identity and large original restored against this manifest')
    return m, mb


def stage(entry_path):
    e, entry_binding = validated_entry(entry_path)
    receipt_dir = safe(e['stage_receipt_directory'], False)
    need(not receipt_dir.exists() and not receipt_dir.is_relative_to(PACKAGE) and
         not receipt_dir.is_relative_to(REPO), 'Fresh stage receipt outside immutable roots')
    receipt_dir.mkdir(parents=True)
    try:
        write_new(receipt_dir/'STARTED.json', json_bytes({'entry': entry_binding, 'mode': 'stage_only'}))
        manifest, manifest_bytes = verify_package_and_full_restore(e)
        need(git('rev-parse', 'HEAD').stdout.decode().strip() == PARENT, 'Exact clone parent HEAD')
        remote = git('ls-remote', '--exit-code', 'origin', 'refs/heads/main').stdout.decode().splitlines()
        need(remote == [PARENT+'\trefs/heads/main'], 'Exact remote main before staging')
        need(git('status', '--porcelain', '--untracked-files=no').stdout == b'', 'Tracked checkout clean')
        git('diff', '--quiet')
        git('diff', '--cached', '--quiet')
        destination = safe(REPO/relative(DEST), False)
        need(not destination.exists(), 'New publication destination only')
        destination.mkdir(parents=True)
        expected = {}
        for row in manifest['files']:
            rel = relative(row['relative_path'])
            source = {'path': str(PACKAGE/rel), 'sha256': row['sha256'], 'size_bytes': row['size_bytes']}
            copy_bound(source, destination/rel)
            expected[(PurePosixPath(DEST)/rel.as_posix()).as_posix()] = source
        write_new(destination/'MANIFEST.json', manifest_bytes)
        expected[DEST+'/MANIFEST.json'] = {'sha256': sha(manifest_bytes), 'size_bytes': len(manifest_bytes)}
        preserved = {}
        for rel in REGISTRIES:
            p = safe(REPO/relative(rel))
            old = git('show', 'HEAD:'+rel).stdout
            with open_read(p) as f:
                need(f.read() == old, 'Registry working bytes equal original HEAD')
            prefix = e['registry_prefixes'][rel]['utf8'].encode('utf-8')
            new = prefix+old
            with p.open('wb') as f:
                f.write(new)
            with open_read(p) as f:
                actual = f.read()
            need(actual == new and actual[len(prefix):] == old, 'Preserved exact three registry tails')
            expected[rel] = {'sha256': sha(new), 'size_bytes': len(new)}
            preserved[rel] = {'old_sha256': sha(old), 'old_size_bytes': len(old),
                              'prefix_sha256': sha(prefix), 'prefix_size_bytes': len(prefix),
                              'new_sha256': sha(new)}
        allowlist = sorted(expected)
        git('add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul',
            inp=b'\0'.join(p.encode('utf-8') for p in allowlist)+b'\0')
        staged = git('diff', '--cached', '--name-only', '-z').stdout.decode('utf-8').split('\0')
        need(staged[-1] == '' and sorted(staged[:-1]) == allowlist, 'Exact NUL stage allowlist')
        for rel, b in expected.items():
            current = measure(REPO/relative(rel))
            need(current['sha256'] == b['sha256'] and current['size_bytes'] == b['size_bytes'], 'Working copy bytes exact')
            blob = git('cat-file', 'blob', ':'+rel).stdout
            need(sha(blob) == b['sha256'] and len(blob) == b['size_bytes'], 'Exact index blob SHA/size')
        checks = e['whitespace_check_relative_paths']
        need(checks and len(checks) == len(set(checks)) and set(checks) <= set(expected) and
             set(REGISTRIES) <= set(checks), 'Explicit core whitespace paths inside staged allowlist')
        git('diff', '--cached', '--check', '--', *checks)
        tree = git('write-tree').stdout.decode().strip()
        need(len(tree) == 40 and all(c in '0123456789abcdef' for c in tree), 'Staged tree SHA')
        result = {'completed': True, 'PASS': True, 'entry': entry_binding,
                  'package_manifest': e['package_manifest'], 'full_restore_QA': e['full_restore_QA'],
                  'parent_HEAD_and_remote_main': PARENT, 'destination': DEST, 'staged_tree': tree,
                  'staged_files': [{'relative_path': p, 'sha256': expected[p]['sha256'],
                                    'size_bytes': expected[p]['size_bytes']} for p in allowlist],
                  'all_staged_and_working_bytes_verified': True,
                  'registry_preserved_tails': preserved, 'whitespace_checked_paths': checks,
                  'archived_bytes_not_normalized': True, 'commit_or_push_performed': False,
                  'force_push': False}
        write_new(receipt_dir/'STAGED_QA.json', json_bytes(result))
        answer = {'completed': True, 'PASS': True, 'staged_tree': tree, 'staged_files': len(allowlist),
                  'receipt': measure(receipt_dir/'STAGED_QA.json'), 'commit_or_push_performed': False}
        print(json.dumps(answer, ensure_ascii=False, sort_keys=True), flush=True)
        return answer
    except BaseException as exc:
        failure = {'completed': False, 'PASS': False, 'mode': 'stage_only',
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'partial_copies_registry_and_index_preserved': True, 'automatic_retry': False,
                   'commit_or_push_performed': False}
        if isinstance(exc, subprocess.CalledProcessError):
            failure.update(returncode=exc.returncode,
                           stdout=exc.stdout.decode('utf-8', errors='replace'),
                           stderr=exc.stderr.decode('utf-8', errors='replace'))
        write_new(receipt_dir/'FAILURE.json', json_bytes(failure))
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    stage(parser.parse_args().entry)
