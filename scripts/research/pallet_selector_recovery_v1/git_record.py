import argparse
import subprocess
from . import common as C

def main(n):
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True).split()[0]
    assert head==remote
    C.freeze(C.DOC/f'STAGE{n}_GIT.json',dict(stage=n,commit=head,remote_sha=remote,push='VERIFIED',branch='main',created_at=C.now(),
        note='Post-push receipt is included in the next stage/integrated commit to avoid self-referential commit hash.'))
    print('STAGE_PUSH_VERIFIED',n,head,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);main(p.parse_args().stage)
