"""Read-only actual main-ref and complete reviewed Git-blob verification.

The root agent performs fetch/commit/push. This checker never updates a Git
ref: the queried remote head must already exist in the local object database.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from . import publication_audit as A


def git(repo, arguments, data=None):
    result = subprocess.run(['git', *arguments], cwd=repo, input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout


def main(a):
    started = time.monotonic(); repo = a.worktree.resolve(); public = a.public_dir.resolve()
    if public != repo / '_docs/experiments' / A.NAMESPACE: raise ValueError('Exact new namespace required')
    for value in (a.result_commit, a.expected_head):
        if not re.fullmatch('[0-9a-f]{40}', value): raise ValueError('Literal full commit SHA required')
    origin = git(repo, ['remote', 'get-url', 'origin']).decode().strip()
    if not re.fullmatch(r'(?:https://github\.com/|git@github\.com:)CanelE452/pallet-6d-pose(?:\.git)?', origin):
        raise ValueError('Unexpected origin repository')
    remote_lines = git(repo, ['ls-remote', '--refs', 'origin', 'refs/heads/main']).decode().splitlines()
    if len(remote_lines) != 1: raise ValueError('Exact remote main ref required')
    remote, name = remote_lines[0].split()
    if name != 'refs/heads/main' or remote != a.expected_head: raise ValueError('Actual remote main differs from expected head')
    git(repo, ['cat-file', '-e', remote + '^{commit}'])
    git(repo, ['merge-base', '--is-ancestor', a.result_commit, remote])
    manifest_path = public / 'PUBLIC_PACK_MANIFEST.json'; audit_path = public / 'PUBLIC_PACK_AUDIT.json'
    manifest = A.read(manifest_path); audit = A.read(audit_path)
    if (audit['status'] != 'PASS' or audit['failure_count'] != 0 or audit['problems']
            or audit['manifest_sha256'] != A.sha(manifest_path)):
        raise ValueError('Exact reviewed immutable payload needs a zero-failure offline audit')
    declared = dict(manifest['source_files'])
    declared.update({str(public.relative_to(repo) / name): value for name, value in manifest['artifacts'].items()})
    declared.update({name: value['whole_current_file_sha256'] for name, value in manifest['pointer_files'].items()})
    for path, expected in declared.items():
        if A.sha(repo / A.relative(path)) != expected: raise ValueError('Local reviewed payload changed')
    for name in A.METADATA:
        path = public / name
        if path.exists(): declared[str(path.relative_to(repo))] = A.sha(path)
    roots = [str(public.relative_to(repo)), 'scripts/research/' + A.NAMESPACE]
    raw = git(repo, ['ls-tree', '-rz', '--name-only', remote, '--', *roots])
    scoped = {s.decode() for s in raw.split(b'\0') if s}
    if scoped != set(declared) - set(manifest['pointer_files']):
        raise ValueError('Remote namespace has missing or unreviewed extra files')
    tree = git(repo, ['ls-tree', '-rz', remote, '--', *sorted(declared)])
    entries = {}
    for record in tree.split(b'\0'):
        if not record: continue
        header, path = record.split(b'\t', 1); mode, kind, oid = header.decode().split()
        if kind != 'blob' or mode not in ('100644', '100755'): raise ValueError('Non-regular remote Git entry')
        entries[path.decode()] = oid
    if set(entries) != set(declared): raise ValueError('Remote tree is missing reviewed paths')
    names = sorted(entries)
    body = git(repo, ['cat-file', '--batch'], ''.join(entries[name] + '\n' for name in names).encode())
    position = 0; proof = []
    for path in names:
        end = body.index(b'\n', position); oid, kind, size = body[position:end].decode().split(); size = int(size)
        blob = body[end+1:end+1+size]
        if oid != entries[path] or kind != 'blob' or len(blob) != size or body[end+1+size:end+2+size] != b'\n':
            raise ValueError('Remote Git batch framing changed')
        if blob != (repo / path).read_bytes() or hashlib.sha256(blob).hexdigest() != declared[path]:
            raise ValueError('Actual remote blob differs from reviewed complete local bytes: ' + path)
        proof.append(dict(path=path, git_blob_oid=oid, bytes=size, sha256=declared[path])); position = end+size+2
    if position != len(body): raise ValueError('Unparsed Git batch data')
    if git(repo, ['ls-remote', '--refs', 'origin', 'refs/heads/main']).decode().splitlines() != remote_lines:
        raise ValueError('Remote main moved during verification')
    report = dict(schema='BI_OBSERVATION_ACTUAL_REMOTE_MAIN_BLOBS_VERIFICATION_V1', status='PASS',
        actual_remote_main=remote, expected_head=a.expected_head, result_commit=a.result_commit,
        result_commit_is_ancestor=True, files=len(proof), all_complete_blobs_equal=True,
        public_payload_manifest_sha256=A.sha(manifest_path), offline_audit_sha256=A.sha(audit_path),
        blobs=proof, git_mutations=0, push_or_fetch_performed=False,
        remote_ref_requeried_after_comparison=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic()-started,
        claim_scope='Actual remote main ref inclusion plus locally available fetched Git object bytes; no CDN or browser-render assertion')
    A.write_report(a.private_output, report, a.run_dir, immutable=True)
    if a.public_output:
        A.write_report(a.public_output, A.sanitized(report), a.run_dir)
    print(json.dumps({k: report[k] for k in ('status', 'actual_remote_main', 'files', 'all_complete_blobs_equal')}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser()
    for name in ('worktree', 'public-dir', 'run-dir', 'private-output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--public-output', type=Path); p.add_argument('--result-commit', required=True); p.add_argument('--expected-head', required=True)
    return p


if __name__ == '__main__': main(parser().parse_args())
