"""Root read-only full-file and Git-state check before a separately run push."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def safe(p):
    p = Path(p)
    assert p.is_absolute() and 'v112' not in str(p).casefold()
    assert all(not n.is_symlink() for n in [p, *p.parents])
    return p


def binding(p):
    p = safe(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(safe(args.plan).read_text())
    repo = safe(plan['repository'])
    assert str(repo) == '/home/minjae/Documents/pallet-visibility-publication-20261024'
    assert plan['authorized_main_push'] is True and plan['branch'] == 'main'
    assert plan['remote'] == 'https://github.com/CanelE452/pallet-6d-pose.git'

    def git(*argv, input_data=None, allowed=(0,)):
        r = subprocess.run(['git', *argv], cwd=repo, input=input_data, capture_output=True)
        assert r.returncode in allowed, (argv, r.returncode)
        return r.stdout

    head = git('rev-parse', 'HEAD').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    assert head == remote == plan['initial_remote_main']
    assert not git('diff', '--cached', '--name-only', '-z')
    entries = [f for batch in plan['batches'] for f in batch['files']]
    paths = {f['path'] for f in entries}
    assert len(paths) == len(entries)
    total = 0
    for item in entries:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        p = safe(repo / relative)
        raw = p.read_bytes()
        assert len(raw) == item['size_bytes']
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        total += len(raw)
    raw_status = git('status', '--porcelain', '-z', '--untracked-files=all')
    status = raw_status.split(b'\0')
    normal = set()
    for entry in status:
        if not entry:
            continue
        assert len(entry) >= 4 and entry[2:3] == b' '
        state, path = entry[:2], entry[3:].decode()
        assert b'R' not in state and b'C' not in state and b'D' not in state
        normal.add(path)
    assert normal <= paths, sorted(normal - paths)
    missing = paths - normal
    ignored = set()
    if missing:
        data = b''.join(p.encode() + b'\0' for p in sorted(missing))
        ignored = {p.decode() for p in git('check-ignore', '--stdin', '-z', input_data=data).split(b'\0') if p}
        assert ignored == missing
    result = dict(completed=True, PASS=True, code=binding(__file__), plan=binding(args.plan),
                  files=len(entries), bytes=total, batches=len(plan['batches']),
                  every_file_sha256_and_size_exact=True, HEAD=head, remote_main=remote,
                  index_empty=True, normal_status_paths=len(normal),
                  independently_proven_ignored_plan_paths=len(ignored),
                  unexpected_normal_worktree_paths=0, Git_mutations=0,
                  excluded_source_bodies_or_array_decodes=0)
    with safe(args.receipt).open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(receipt=binding(args.receipt), files=len(entries), bytes=total,
                          normal=len(normal), ignored=len(ignored))))


if __name__ == '__main__':
    main()
