"""Publish only the two authorized namespaces, using ordinary Git pushes.

The original dirty checkout is read only. Publication uses its isolated clean
main worktree; no reset, stash, checkout, or force push is performed there.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[3]
PREFIXES=tuple(f'{top}/{name}/' for top in ('_docs/experiments','scripts/research')
    for name in ('pallet_wd_hypothesis_diag_20261010','pallet_square6d_manualpnp_20261010'))


def git(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd).decode().strip()


def preserve(private):
    private=Path(private)
    start=json.loads((private/'START.json').read_text())
    original=Path(start['original_root'])
    assert git('rev-parse','HEAD',cwd=original)==start['head']
    assert git('branch','--show-current',cwd=original)==start['branch']
    status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=original)
    assert status==(private/'original_status.bin').read_bytes(), 'Original changes must remain byte-identical'
    diff=subprocess.check_output(['git','diff','--binary'],cwd=original)
    assert diff==(private/'original_tracked.diff').read_bytes()
    for name,sha in start['changed_tracked_sha256'].items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest()==sha
    return dict(status='PASS',original_HEAD=start['head'],
        tracked_changed_files=len(start['changed_tracked_sha256']),
        original_status_entries=start['status_entries'],status_byte_identical=True,
        tracked_diff_byte_identical=True,changed_tracked_hashes_identical=True)


def run(main_tree,private,message):
    assert git('branch','--show-current')=='research/wd-square6d-20261010'
    before=preserve(private)
    files=[str(p.relative_to(ROOT)) for prefix in PREFIXES for p in (ROOT/prefix).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pending')]
    assert files and all((ROOT/p).stat().st_size<100_000_000 for p in files)
    subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],cwd=ROOT,check=True)
    subprocess.run(['git','add','-f','--',*files],cwd=ROOT,check=True)
    staged=git('diff','--cached','--name-only').splitlines()
    assert staged and all(name.startswith(PREFIXES) for name in staged), staged
    assert all(line[0]!='D' for line in git('diff','--cached','--name-status').splitlines())
    # Python's CSV writer uses standard CRLF. Accept that line ending while
    # retaining checks for actual trailing spaces and spaces before tabs.
    subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol',
                    'diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
    sha=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/research/wd-square6d-20261010'],cwd=ROOT,check=True)
    main_tree=Path(main_tree).resolve()
    assert git('branch','--show-current',cwd=main_tree)=='main'
    assert not git('status','--porcelain',cwd=main_tree), 'Publication main worktree must be clean'
    subprocess.run(['git','merge','--ff-only','research/wd-square6d-20261010'],cwd=main_tree,check=True)
    subprocess.run(['git','push','origin','main'],cwd=main_tree,check=True)
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    assert remote==sha
    receipt=dict(status='PASS',published_commit=sha,remote_main_sha=remote,
        force_push=False,original_checkout=preserve(private),files_changed=staged)
    (Path(private)/f'PUBLISH_{sha}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',commit=sha,files_changed=len(staged))),flush=True)
    return receipt


def check_links(sha,paths):
    checks=[]
    for path in paths:
        url=f'https://raw.githubusercontent.com/CanelE452/pallet-6d-pose/{sha}/{path}'
        with urllib.request.urlopen(url,timeout=45) as response:
            payload=response.read();status=response.status
        expected=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        actual=hashlib.sha256(payload).hexdigest()
        assert status==200 and actual==expected, path
        checks.append(dict(path=path,http_status=status,sha256=actual,remote_matches_local=True))
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-tree',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--message',required=True)
    args=parser.parse_args()
    run(args.main_tree,args.private_dir,args.message)
