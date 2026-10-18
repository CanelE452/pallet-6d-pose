"""Actual remote-ref and fetched Git-blob verification; no working-tree mutation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def run(cmd, repo, data=None):
    p = subprocess.run(cmd, cwd=repo, input=data, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=True)
    return p.stdout


def main(a):
    started = time.monotonic()
    repo, public, private = map(Path, (a.repo, a.public, a.run_dir))
    output = private / a.output
    if output.exists(): raise FileExistsError('Preserve existing remote verification')
    for value in (a.result_commit, a.expected_head):
        if value is not None and not re.fullmatch('[0-9a-f]{40}', value):
            raise ValueError('Literal full commit SHA required')
    url = run(['git', 'remote', 'get-url', 'origin'], repo).decode().strip()
    if url != 'https://github.com/CanelE452/pallet-6d-pose.git': raise ValueError('Unexpected origin repository')
    run(['git', 'fetch', 'origin', 'main'], repo)
    lines = run(['git', 'ls-remote', '--refs', 'origin', 'refs/heads/main'], repo).decode().splitlines()
    if len(lines) != 1: raise ValueError('Exact remote main required')
    remote, name = lines[0].split()
    if name != 'refs/heads/main' or len(remote) != 40: raise ValueError('Unexpected remote response')
    fetched = run(['git', 'rev-parse', 'origin/main'], repo).decode().strip()
    if fetched != remote or (a.expected_head and remote != a.expected_head):
        raise ValueError('Remote moved or fetched ref differs; do not infer verification')
    run(['git', 'merge-base', '--is-ancestor', a.result_commit, remote], repo)
    manifest = json.loads((public / 'PUBLIC_PACK_MANIFEST.json').read_text())
    manifest_sha = hashlib.sha256((public/'PUBLIC_PACK_MANIFEST.json').read_bytes()).hexdigest()
    audit = json.loads((public / 'PUBLIC_PACK_AUDIT.json').read_text())
    if audit.get('status') != 'PASS' or audit.get('failure_count') != 0 or audit.get('problems') != [] or audit.get('manifest_sha256') != manifest_sha:
        raise ValueError('Current manifest is not bound to a zero-failure offline PASS')
    declared = dict(manifest['source_files'])
    declared.update({p:(v['whole_current_file_sha256'] if isinstance(v,dict) else v)
                     for p,v in manifest['pointer_files'].items()})
    declared.update({str(public.relative_to(repo)/n):v for n,v in manifest['artifacts'].items()})
    names = set(manifest['source_files']) | set(manifest['pointer_files'])
    names |= {str(public.relative_to(repo) / n) for n in manifest['artifacts']}
    names |= {str(public.relative_to(repo) / n) for n in
              ('PUBLIC_PACK_MANIFEST.json', 'PUBLIC_PACK_AUDIT.json', 'PUBLICATION_RECEIPT.json')
              if (public / n).exists()}
    names = sorted(names)
    for path, expected in declared.items():
        if hashlib.sha256((repo/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Local reviewed manifest payload differs: '+path)
    scope_roots = [str(public.relative_to(repo)), 'scripts/research/pallet_observation_contract_h1_v1']
    scoped = set(run(['git','ls-tree','-rz','--name-only',remote,'--']+scope_roots,repo).decode().strip('\0').split('\0'))
    if scoped != set(names)-set(manifest['pointer_files']):
        raise ValueError('Remote namespace inventory has extra or missing files')
    tree = run(['git', 'ls-tree', '-rz', remote, '--'] + names, repo)
    entries = {}
    for item in tree.split(b'\0'):
        if not item: continue
        header, path = item.split(b'\t', 1)
        mode, kind, oid = header.decode().split()
        if kind != 'blob' or mode not in ('100644', '100755'): raise ValueError('Non-regular Git entry')
        entries[path.decode()] = oid
    if set(entries) != set(names): raise ValueError('Remote tree has missing reviewed artifacts')
    body = run(['git', 'cat-file', '--batch'], repo,
               ''.join(entries[n] + '\n' for n in names).encode())
    pos = 0; proofs = []
    for path in names:
        end = body.index(b'\n', pos)
        oid, kind, size = body[pos:end].decode().split(); size = int(size)
        raw = body[end + 1:end + 1 + size]
        if len(raw) != size or oid != entries[path] or kind != 'blob': raise ValueError('Git batch binding differs')
        if body[end+1+size:end+2+size] != b'\n': raise ValueError('Git batch delimiter differs')
        if raw != (repo / path).read_bytes(): raise ValueError('Remote blob differs from reviewed file: ' + path)
        proofs.append(dict(path=path, git_blob_oid=oid, bytes=size,
                           sha256=hashlib.sha256(raw).hexdigest()))
        pos = end + size + 2
    if pos != len(body): raise ValueError('Unparsed Git data')
    final_lines = run(['git', 'ls-remote', '--refs', 'origin', 'refs/heads/main'], repo).decode().splitlines()
    if final_lines != lines: raise ValueError('Remote main moved during blob verification')
    value = dict(schema='ACTUAL_REMOTE_MAIN_AND_BLOBS_VERIFICATION_V1', status='PASS',
        actual_remote_main=remote, fetched_origin_main=fetched, actual_origin_url=url,
        required_result_commit=a.result_commit, result_commit_is_ancestor=True,
        checked_files=len(proofs), all_checked_files_exact=True,
        verification_method='Actual git fetch + independent ls-remote main query + fetched Git tree and whole blob byte comparison against reviewed files.',
        public_pack_manifest_sha256=manifest_sha,
        public_pack_audit_sha256=hashlib.sha256((public/'PUBLIC_PACK_AUDIT.json').read_bytes()).hexdigest(),
        audit_status=json.loads((public/'PUBLIC_PACK_AUDIT.json').read_text())['status'],
        blobs=proofs, wall_seconds=time.monotonic()-started,
        remote_ref_requeried_after_blob_comparison=True,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        claim_scope='Git ref inclusion and complete fetched blob bytes; not a browser render or CDN availability claim.')
    if value['audit_status'] != 'PASS': raise ValueError('Offline audit is not PASS')
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:value[k] for k in ('status','actual_remote_main','checked_files','all_checked_files_exact','wall_seconds')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--public',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--result-commit',required=True)
    p.add_argument('--expected-head');p.add_argument('--output',required=True);main(p.parse_args())
