import argparse
from . import common as C

def main(n):
    head=C.git('rev-parse','HEAD');remote=C.git('ls-remote','origin','refs/heads/main').split()[0];assert head==remote
    value=dict(created_at=C.now(),commit=head,branch=C.git('branch','--show-current'),remote_HEAD=remote,push_verified=True,
        status=C.git('status','--short','--branch'),task_status=C.git('status','--short','--',str(C.DOC.relative_to(C.ROOT)),str((C.ROOT/'scripts/research'/C.NAME).relative_to(C.ROOT))),
        receipt_note='Snapshot after this stage push; later receipt/report commits necessarily have a different HEAD.')
    C.save(C.OUT/f'STAGE{n}_GIT.json',value);C.save(C.DOC/f'STAGE{n}_GIT.json',value)
    print('PUSH_VERIFIED',n,head)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);main(p.parse_args().stage)
