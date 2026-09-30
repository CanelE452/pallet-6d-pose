"""Copy the reviewed manifest into an isolated checkout; never commit or push."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from . import common as C


def git(checkout,*args):
    return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkout',type=Path,default=Path('/tmp/pallet-pose-github-review-20260930'))
    args=parser.parse_args();checkout=args.checkout.resolve()
    assert checkout.is_relative_to(Path('/tmp')) and checkout != C.ROOT
    manifest=C.read(C.DOC/'PUBLICATION_MANIFEST.json')
    assert C.read(C.DOC/'PUBLICATION_VALIDATION.json')['PASS']
    C.verify(manifest['validation'])
    assert git(checkout,'rev-parse','HEAD')==manifest['publication_base_commit']
    assert git(checkout,'status','--porcelain')=='','Review existing checkout changes first'
    bindings=manifest['files']+[C.bind(C.DOC/'PUBLICATION_MANIFEST.json')]
    paths=[]
    for binding in bindings:
        C.verify(binding)
        path=Path(binding['path']);assert not path.is_absolute() and '..' not in path.parts
        destination=checkout/path;destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.is_symlink()
        shutil.copy2(C.ROOT/path,destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest()==binding['sha256']
        paths.append(str(path))
    subprocess.run(['git','-C',str(checkout),'add','--',*paths],check=True)
    staged=set(git(checkout,'diff','--cached','--name-only','--diff-filter=ACMRT').splitlines())
    assert staged and staged<=set(paths)
    assert not git(checkout,'diff','--name-only')
    assert not git(checkout,'ls-files','--others','--exclude-standard')
    C.save(C.RAW/'PUBLICATION_STAGED.json',dict(checkout=str(checkout),source_workspace_HEAD=git(C.ROOT,'rev-parse','HEAD'),
        base_commit=manifest['publication_base_commit'],manifest=C.bind(C.DOC/'PUBLICATION_MANIFEST.json'),
        copied_files=len(paths),changed_files=sorted(staged),all_copied_SHA256_verified=True,committed=False,pushed=False))
    print('PUBLICATION_STAGED',len(staged),'changed files;',len(paths),'verified copied files',flush=True)


if __name__=='__main__':main()
