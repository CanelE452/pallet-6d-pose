"""Preserve the user's working files while integrating the audited main index.

Plan is read-only. Integrate changes only Git index/ref metadata, after actual
compute and ending verification. It never checks out upstream working files.
The caller reviews and stages the new namespace, then commits and pushes main.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def git(root, *args, input=None):
    return subprocess.run(['git', '-C', str(root), *args], input=input,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.new')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def plan(root, audit_path):
    root = Path(root).resolve()
    evidence = load(audit_path)
    summary = evidence['summary']
    old = summary['original_HEAD']
    target = summary['target_origin_MAIN']
    assert git(root, 'symbolic-ref', 'HEAD').decode().strip() == 'refs/heads/main'
    assert git(root, 'rev-parse', 'HEAD').decode().strip() == old
    assert not git(root, 'diff', '--cached', '--name-only')
    git(root, 'merge-base', '--is-ancestor', old, target)
    remote = git(root, 'ls-remote', 'origin', 'refs/heads/main').decode().split()
    assert remote == [target, 'refs/heads/main'], 'Remote changed: redo integration audit'
    index_flags = git(root, 'ls-files', '-v', '-z').split(b'\0')
    assert all(line[:1] == b'H' for line in index_flags if line), 'Existing special index flags require review'
    for record in evidence['records']:
        path = root / record['path']
        assert not record['protected_input_overlap'], record['path']
        if record['current_exists']:
            assert path.exists() and path.is_file(), record['path']
            assert path.stat().st_size == record['bytes'], record['path']
            assert digest(path) == record['current_sha256'], record['path']
        else:
            assert not path.exists() and not path.is_symlink(), record['path']
    for entry in summary['symlink_ancestor_groups']:
        assert os.readlink(root / entry['path']) == entry['link_target']
    absent = [r['path'] for r in evidence['records'] if not r['current_exists']]
    below_links = [r['path'] for r in evidence['records']
                   if r['category'] == 'untracked_below_symlink_ancestor']
    preserve = sorted(set(summary['user_tracked_dirty_paths'] +
                          [r['path'] for r in summary['conflicting_untracked_different']]))
    original = [dict(path=p, sha256=digest(root / p), bytes=(root / p).stat().st_size)
                for p in preserve]
    history = root / '_docs/history/2026-10-06.md'
    if history.exists() and str(history.relative_to(root)) not in preserve:
        original.append(dict(path=str(history.relative_to(root)), sha256=digest(history), bytes=history.stat().st_size))
    return dict(schema='main_index_only_publication_plan_v1', status='PASS',
                source_checkout=str(root), original_HEAD=old, target_main=target,
                audit_path=str(Path(audit_path).resolve()), audit_sha256=digest(audit_path),
                skip_worktree_paths=sorted(set(absent + below_links)),
                absent_paths=len(absent), symlink_descendants=len(below_links),
                original_file_bytes=original, symlinks=summary['symlink_ancestor_groups'],
                existing_changed_paths=sum(r['current_exists'] for r in evidence['records']),
                working_file_writes=0, branch='main', new_branches=0,
                time_unix=time.time())


def integrate(root, experiment, audit_path, cache):
    cache = Path(cache).resolve()
    receipt_path = cache / 'MAIN_INDEX_INTEGRATION.json'
    assert not receipt_path.exists(), 'Integration is already recorded; inspect before any replay'
    root = Path(root).resolve()
    doc = Path(experiment).resolve() / '_docs/experiments/pallet_pose_target_6d_20261006_v1'
    status = load(doc / 'STATUS.json')
    verify = load(doc / 'audit/INDEPENDENT_VERIFICATION.json')
    assert status['status'] == 'COMPUTE_AND_REPORT_DONE'
    assert verify['status'] == 'PASS' and verify['full_source_input_SHA_end'] == 'PASS'
    result = plan(root, audit_path)
    backup = cache / 'main-preservation'
    assert not backup.exists(), 'Do not overwrite preservation backup'
    backup.mkdir()
    index = Path(git(root, 'rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = root / index
    shutil.copy2(index, backup / 'index.before')
    for entry in result['original_file_bytes']:
        destination = backup / 'files' / entry['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / entry['path'], destination)
        assert digest(destination) == entry['sha256']
    result.update(status='ATTEMPT', index_backup=str(backup / 'index.before'),
                  index_backup_sha256=digest(backup / 'index.before'),
                  operation='read-tree without -u; compare-and-swap update-ref main; explicit skip-worktree')
    save(receipt_path, result)
    git(root, 'read-tree', result['target_main'])
    try:
        git(root, 'update-ref', 'refs/heads/main', result['target_main'], result['original_HEAD'])
    except Exception:
        shutil.copy2(backup / 'index.before', index)
        result.update(status='FAILED_REF_CAS_INDEX_RESTORED', end_unix=time.time())
        save(receipt_path, result)
        raise
    assert not git(root, 'diff', '--cached', '--name-only')
    assert git(root, 'write-tree').decode().strip() == git(root, 'rev-parse', result['target_main'] + '^{tree}').decode().strip()
    nul_paths = b''.join(os.fsencode(p) + b'\0' for p in result['skip_worktree_paths'])
    git(root, 'update-index', '-z', '--skip-worktree', '--stdin', input=nul_paths)
    for entry in result['original_file_bytes']:
        assert digest(root / entry['path']) == entry['sha256'], entry['path']
    for entry in result['symlinks']:
        assert os.readlink(root / entry['path']) == entry['link_target']
    flags = git(root, 'ls-files', '-v', '-z').split(b'\0')
    skip = sorted(os.fsdecode(line[2:]) for line in flags if line[:1] == b'S')
    assert skip == result['skip_worktree_paths']
    assert not git(root, 'diff', '--cached', '--name-only')
    result.update(status='PASS', index_after_sha256=digest(index),
                  actual_main_HEAD=git(root, 'rev-parse', 'HEAD').decode().strip(),
                  original_file_bytes_preserved=True, upstream_working_file_writes=0,
                  skip_worktree_count=len(skip), end_unix=time.time())
    save(receipt_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'integrate'), required=True)
    parser.add_argument('--source-checkout', type=Path, default=Path('/home/minjae/Documents/github/pallet-pose'))
    parser.add_argument('--experiment-root', type=Path, default=Path('/home/minjae/Documents/github/pallet-pose-target-6d-20261006'))
    parser.add_argument('--audit-json', type=Path, default=Path('/tmp/pallet-main-publication-readonly-audit-20261006.json'))
    parser.add_argument('--cost-cache', type=Path, default=Path('/tmp/pallet-pose-target-6d-cache'))
    args = parser.parse_args(argv)
    result = plan(args.source_checkout, args.audit_json) if args.stage == 'plan' else integrate(
        args.source_checkout, args.experiment_root, args.audit_json, args.cost_cache)
    print(json.dumps({k: v for k, v in result.items() if k not in
          ('skip_worktree_paths', 'original_file_bytes', 'symlinks')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
