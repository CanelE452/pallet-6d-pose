"""Main-only, scoped publication; never clean, stash, or force push user work."""
from env import *
SCOPES=[str(p.relative_to(ROOT)) for p in (HERE,DOC,PAPER)]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def run():
    assert git('branch','--show-current')=='main'
    assert (DOC/'FINAL_STATUS.json').exists(),'Audit must distinguish pending work before publication'
    assert complete('AUDIT_COMPLETE'),'Current report must be bound by an actual audit'
    blocked=DOC/'AUDIT_BLOCKED.json'
    if blocked.exists():assert read(DOC/'AUDIT_COMPLETE.json')['end']>read(blocked)['time'],'Resolve or explicitly report the latest failed audit before publication'
    git('fetch','origin','main')
    if git('rev-parse','HEAD')!=git('rev-parse','origin/main') and subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],cwd=ROOT).returncode==0:
        # Git itself refuses an overlap with dirty files; no stash or force checkout.
        git('merge','--ff-only','origin/main')
    staged=git('diff','--cached','--name-only').splitlines()
    assert all(any(p==s or p.startswith(s+'/') for s in SCOPES) for p in staged),'Unrelated user staging preserved; resolve explicitly before committing'
    # Never stage raw data, external weights, TeX fonts or generated build intermediates.
    git('add','--',*SCOPES)
    for p in PAPER.rglob('*'):
        if p.is_file() and p.suffix in ('.pdf','.png'):
            assert p.stat().st_size<10_000_000
            git('add','-f','--',str(p.relative_to(ROOT)))
    files=git('diff','--cached','--name-only').splitlines()
    assert all(any(p.startswith(s+'/') for s in SCOPES) for p in files)
    assert not any(Path(p).suffix in ('.pt','.npy','.npz','.ttf','.otf','.aux','.bbl','.blg','.log','.out') for p in files if p.startswith(SCOPES[2]))
    # Preserve hash-frozen math source bytes and the standard CSV writer's CRLF.
    # Only those two paths permit EOF blank/CRLF; all other whitespace checks stay on.
    byte_frozen=[str((HERE/'posefix_contract_math.py').relative_to(ROOT)),str((DOC/'CONFIRMATION_CAPTURE_TEMPLATE.csv').relative_to(ROOT))]
    git('diff','--cached','--check','--','.',*[':(exclude)'+p for p in byte_frozen])
    git('-c','core.whitespace=-blank-at-eof,cr-at-eol','diff','--cached','--check','--',*byte_frozen)
    print(git('diff','--cached','--stat'),flush=True)
    if files:print(git('commit','-m','research: complete fixed refinement comparisons and assemble Sensors manuscript'),flush=True)
    git('fetch','origin','main')
    # A remote advance is not permission to stash dirty work or resolve conflicts.
    ancestor=subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],cwd=ROOT).returncode
    assert ancestor==0,'Remote advanced: preserve training SHA and resolve unpublished integration without touching user changes'
    git('push','origin','main');git('fetch','origin','main')
    local=git('rev-parse','HEAD');remote=git('rev-parse','origin/main');advertised=git('ls-remote','origin','refs/heads/main').split()[0]
    assert local==remote==advertised,(local,remote,advertised)
    write(RAW/'PUBLISH_COMPLETE.json',dict(complete=True,time=now(),local_SHA=local,origin_main_SHA=remote,remote_advertised_SHA=advertised,worktree=git('status','--short'),not_submitted=True))
    print('PUSH_VERIFIED',local,flush=True)

if __name__=='__main__':run()
