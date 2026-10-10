"""Snapshot and compare existing repository bytes without changing old inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def source_state(root):
    root = Path(root)
    status = git(root, 'status', '--porcelain=v1', '-z')
    diff = git(root, 'diff', '--binary')
    names = git(root, 'diff', '--name-only', '-z').decode().split('\0')
    return dict(head=git(root, 'rev-parse', 'HEAD').decode().strip(),
                branch=git(root, 'branch', '--show-current').decode().strip(),
                status_sha256=hashlib.sha256(status).hexdigest(),
                tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                status_entries=len([part for part in status.split(b'\0') if part]),
                changed_tracked_sha256={name: sha(root / name) for name in names if name})


def snapshot(source_root):
    names = git(REPO, 'ls-files', '-z').decode().split('\0')
    sparse = {row[2:] for row in git(REPO, 'ls-files', '-v', '-z').decode().split('\0') if row.startswith('S ')}
    index = {}
    for row in git(REPO, 'ls-files', '--stage', '-z').decode().split('\0'):
        if row:
            metadata, name = row.split('\t', 1)
            index[name] = metadata.split()[1]
    bindings = []
    for name in names:
        if not name:
            continue
        path = REPO / name
        if path.is_symlink():
            bindings.append(dict(path=name, symlink=str(path.readlink())))
        elif path.is_file():
            bindings.append(dict(path=name, sha256=sha(path), bytes=path.stat().st_size))
        elif name in sparse:
            bindings.append(dict(path=name, not_materialized=True, git_blob=index[name]))
        else:
            raise RuntimeError('Previously tracked input missing: ' + name)
    return dict(schema='boundary_refiner_protection_v2',
                original_research_head=git(REPO, 'rev-parse', 'HEAD').decode().strip(),
                protected_repository_files=len(bindings), files=bindings,
                materialized_repository_files=sum(not b.get('not_materialized', False) for b in bindings),
                source_checkout=source_state(source_root))


def verify(data, source_root):
    failures = []
    index = {}
    for row in git(REPO, 'ls-files', '--stage', '-z').decode().split('\0'):
        if row:
            metadata, name = row.split('\t', 1)
            index[name] = metadata.split()[1]
    for bound in data['files']:
        path = REPO / bound['path']
        if bound.get('not_materialized'):
            good = not path.exists() and index.get(bound['path']) == bound['git_blob']
        elif 'symlink' in bound:
            good = path.is_symlink() and str(path.readlink()) == bound['symlink']
        else:
            good = path.is_file() and path.stat().st_size == bound['bytes'] and sha(path) == bound['sha256']
        if not good:
            failures.append(bound['path'])
    current = source_state(source_root)
    return dict(schema='boundary_refiner_protection_check_v2', passed=not failures and current == data['source_checkout'],
                protected_repository_files=data['protected_repository_files'], changed_protected_paths=failures,
                source_checkout_unchanged=current == data['source_checkout'], current_source_checkout=current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['snapshot', 'verify'])
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--snapshot', default=str(DOC / 'PROTECTION_BEFORE.json'))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.stage == 'snapshot':
        value = snapshot(args.source_root)
        destination = Path(args.snapshot)
    else:
        value = verify(json.loads(Path(args.snapshot).read_text()), args.source_root)
        destination = Path(args.output) if args.output else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: value[key] for key in value if key != 'files'}, ensure_ascii=False))
    if args.stage == 'verify' and not value['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
