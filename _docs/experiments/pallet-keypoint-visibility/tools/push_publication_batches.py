"""Push only a sealed file plan to existing main, without force or unrelated files.

The user explicitly authorized these publications. The plan is an internal
execution seal, not a request for another user confirmation. No source discovery.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time


def run(args, cwd, binary=False):
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(json.dumps(dict(command=args[:4], exit_code=result.returncode,
                                          stderr=result.stderr.decode(errors='replace')[-6000:])))
    return result.stdout if binary else result.stdout.decode().strip()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--state', required=True, type=Path)
    a = p.parse_args()
    plan_bytes = a.plan.read_bytes(); plan = json.loads(plan_bytes)
    seal = hashlib.sha256(plan_bytes).hexdigest()
    assert plan['authorized_main_push'] is True
    repo = Path(plan['repository'])
    assert run(['git', 'remote', 'get-url', 'origin'], repo) == plan['remote']
    assert plan['branch'] == 'main' and plan['batch_max_bytes'] <= 512*1024**2
    if a.state.exists():
        state = json.loads(a.state.read_text()); assert state['plan_sha256'] == seal
    else:
        assert not run(['git', 'diff', '--cached', '--name-only'], repo)
        state = dict(plan_sha256=seal, completed=[], pending=None,
                     expected_remote=plan['initial_remote_main'])
    def save():
        temporary = a.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(a.state)
    for i, batch in enumerate(plan['batches']):
        if i < len(state['completed']): continue
        assert sum(f['size_bytes'] for f in batch['files']) <= plan['batch_max_bytes']
        remote = run(['git', 'ls-remote', 'origin', 'refs/heads/main'], repo).split()[0]
        if state['pending'] is not None and remote == state['pending']['commit']:
            assert run(['git', 'rev-parse', 'HEAD'], repo) == state['pending']['commit']
            state['pending'].update(remote_verified=True, completed_unix=time.time(), resumed_after_remote_acceptance=True)
            state['completed'].append(state['pending']); state['pending'] = None
            state['expected_remote'] = remote; save()
            print(json.dumps(dict(event='PUSH_VERIFIED_ON_RESUME', **state['completed'][-1])), flush=True)
            continue
        assert remote == state['expected_remote'], 'Remote main changed; inspect and integrate without force'
        if state['pending'] is None:
            assert run(['git', 'rev-parse', 'HEAD'], repo) == state['expected_remote'], 'Unplanned local commits'
            assert not run(['git', 'diff', '--cached', '--name-only'], repo)
            expected_blobs = {}
            for f in batch['files']:
                name = f['path']
                assert name.startswith('_docs/experiments/pallet-keypoint-visibility/') or name == 'readme.md'
                assert 'v112' not in name.lower() and '..' not in Path(name).parts
                path = repo/name
                assert path.is_file() and not path.is_symlink()
                h = hashlib.sha256(); g = hashlib.sha1()
                g.update(('blob '+str(f['size_bytes'])+'\0').encode()); size = 0
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(4*1024**2), b''):
                        h.update(block); g.update(block); size += len(block)
                assert size == f['size_bytes'] and h.hexdigest() == f['sha256'], 'Sealed bytes changed'
                expected_blobs[name] = g.hexdigest()
            pathspec = a.state.parent/('PUSH_BATCH_%02d.paths'%i)
            pathspec.write_bytes(b''.join(f['path'].encode()+b'\0' for f in batch['files']))
            run(['git', '-c', 'core.autocrlf=false', 'add', '-f', '--pathspec-from-file='+str(pathspec),
                 '--pathspec-file-nul'], repo)
            staged = set(run(['git', 'diff', '--cached', '--name-only', '-z'], repo, True).decode().split('\0'))-{''}
            assert staged == set(expected_blobs), 'Unexpected or missing staged paths'
            index = run(['git', 'ls-files', '--stage', '-z', '--',
                         '_docs/experiments/pallet-keypoint-visibility', 'readme.md'], repo, True)
            index_map = {}
            for record in index.split(b'\0'):
                if record:
                    meta, name = record.decode().split('\t', 1)
                    mode, oid, stage = meta.split()
                    if name in expected_blobs:
                        assert mode == '100644' and stage == '0'
                        index_map[name] = oid
            assert index_map == expected_blobs, 'Git index changed original bytes'
            run(['git', '-c', 'gc.auto=0', 'commit', '--quiet', '-m', batch['message']], repo)
            committed_names = set(run(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', '-z', 'HEAD'], repo, True).decode().split('\0'))-{''}
            assert committed_names == set(expected_blobs), 'Commit hooks changed planned paths'
            tree = run(['git', 'ls-tree', '-r', '-z', 'HEAD', '--',
                        '_docs/experiments/pallet-keypoint-visibility', 'readme.md'], repo, True)
            committed_blobs = {}
            for record in tree.split(b'\0'):
                if record:
                    meta, name = record.decode().split('\t', 1)
                    mode, typ, oid = meta.split()
                    if name in expected_blobs:
                        assert mode == '100644' and typ == 'blob'
                        committed_blobs[name] = oid
            assert committed_blobs == expected_blobs, 'Commit differs from verified artifact bytes'
            state['pending'] = dict(batch=i+1, commit=run(['git', 'rev-parse', 'HEAD'], repo),
                                    files=len(batch['files']), stored_bytes=sum(f['size_bytes'] for f in batch['files']),
                                    git_index_byte_identity=True)
            save()
        print(json.dumps(dict(event='PUSH_START', **state['pending'])), flush=True)
        assert run(['git', 'rev-parse', 'HEAD'], repo) == state['pending']['commit'], 'Local pending HEAD changed'
        run(['git', '-c', 'pack.window=0', '-c', 'pack.compression=1', '-c', 'pack.threads=2',
             'push', 'origin', state['pending']['commit']+':refs/heads/main'], repo)
        remote = run(['git', 'ls-remote', 'origin', 'refs/heads/main'], repo).split()[0]
        assert remote == state['pending']['commit'], 'Remote commit verification failed'
        state['pending'].update(remote_verified=True, completed_unix=time.time())
        state['completed'].append(state['pending']); state['pending'] = None
        state['expected_remote'] = remote; save()
        print(json.dumps(dict(event='PUSH_VERIFIED', **state['completed'][-1])), flush=True)
    state['completed_all'] = True; save()
    print(json.dumps(dict(completed=True, batches=len(state['completed']), main=state['expected_remote'])), flush=True)


if __name__ == '__main__':
    main()
