"""Publish only this new result prefix using an isolated Git index/work tree.

Never checks out, stages, resets or stashes the dirty primary checkout.
Non-force push; any remote movement requires preparing anew, never overwriting.
"""
from common import *
import argparse,shutil

PREFIX='_docs/experiments/pallet_dope_thesis_analysis_batch1_v1'
PUB=OUT/'.publication'

def git(*args,env=None,binary=False):
    e=os.environ.copy();e['GIT_OPTIONAL_LOCKS']='0'
    if env:e.update(env)
    p=subprocess.run(['git','-C',str(REPO),*args],env=e,capture_output=True,check=True)
    return p.stdout if binary else p.stdout.decode().strip()

def primary():
    index=REPO/'.git/index'
    changed=git('diff','--name-only','-z',binary=True).decode().split('\0')
    return dict(HEAD=git('rev-parse','HEAD'),index_sha256=sha(index),
        tracked_status=git('status','--porcelain=v1','-uno'),
        modified_worktree_sha256={p:sha(REPO/p) for p in changed if p and (REPO/p).is_file()})

def files():
    result=[]
    for p in sorted(OUT.rglob('*')):
        rel=p.relative_to(OUT)
        if rel.parts[0] in ('.publication','publication') or '__pycache__' in rel.parts:continue
        if p.is_file():result.append((p,rel))
    return result

def publog(event,**kw):
    folder=OUT/'publication';folder.mkdir(exist_ok=True)
    with (folder/'PUBLISH_LOG.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=now(),event=event,**kw),ensure_ascii=False)+'\n')

def prepare():
    assert read(OUT/'COMPLETION.json')['analysis_complete'] is True
    assert read(OUT/'hashes/COMPARISON.json')['all_unchanged'] is True
    PUB.mkdir(exist_ok=True)
    assert not (PUB/'PREPARED.json').exists(),'Prepared commit already exists; inspect instead of replacing'
    before=primary()
    git('fetch','origin','main')
    base=git('rev-parse','origin/main')
    assert git('merge-base','--is-ancestor','79ba5b3f6f03e520499278b939412fddfdd57c30',base)==''
    assert not git('ls-tree','--name-only',base,PREFIX),'Target already exists remotely: stop'
    for ancestor in ('AGENTS.md','_docs/AGENTS.md','_docs/experiments/AGENTS.md'):
        assert not git('ls-tree','--name-only',base,ancestor),'Remote AGENTS needs review before publishing'
    stage=PUB/'worktree';target=stage/PREFIX;target.mkdir(parents=True,exist_ok=True)
    inventory={}
    for src,rel in files():
        assert src.stat().st_size<95_000_000,('Unexpected large deliverable',str(src))
        dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assert sha(src)==sha(dst)
        inventory[PREFIX+'/'+str(rel)]={'sha256':sha(dst),'bytes':dst.stat().st_size}
    assert {PREFIX+'/'+str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}==set(inventory), 'Stale isolated staging files'
    env={'GIT_INDEX_FILE':str(PUB/'index'),'GIT_WORK_TREE':str(stage)}
    git('read-tree',base,env=env)
    git('add','-f','--',PREFIX,env=env)
    names=git('diff','--cached','--name-status',base,env=env).splitlines()
    assert len(names)==len(inventory) and all(n.startswith('A\t'+PREFIX+'/') for n in names)
    # ReportLab's ASCII85 PDF is binary data but Git may classify it as text.
    # Check human text only. Preserve the frozen source's harmless EOF blank line.
    git('-c','core.whitespace=blank-at-eol,space-before-tab,-blank-at-eof,cr-at-eol','diff','--cached','--check',base,'--',
        PREFIX,':(exclude)*.pdf',':(exclude)*.png',env=env)
    tree=git('write-tree',env=env)
    message=PUB/'commit_message.txt'
    message.write_text('Analyze DOPE self-training Batch1 without retraining\n\nAdd frozen-cache bias and paired bootstrap analysis, verified filter diagnostics, PnP availability, and nine fixed qualitative cases. Preserve original experiments and report unavailable recording CIs and unproven R0 decoder explicitly.\n')
    commit=git('commit-tree',tree,'-p',base,'-F',str(message),env=env)
    assert primary()==before,'Primary checkout changed during isolated prepare'
    payload=dict(base=base,commit=commit,prefix=PREFIX,files=inventory,primary_before=before,
        source_folders_unchanged=True,source_integrity_comparison_sha256=sha(OUT/'hashes/COMPARISON.json'),
        code_sha256=sha(Path(__file__)),prepared_utc=now(),git_index=str(PUB/'index'))
    write(PUB/'PREPARED.json',payload)
    publog('prepared',commit=commit,base=base,files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),
           command=['git','commit-tree',tree,'-p',base,'-F',str(message)],isolated_primary_preserved=True)
    print('PREPARED',commit,'files',len(inventory),'bytes',sum(v['bytes'] for v in inventory.values()),flush=True)

def push():
    data=read(PUB/'PREPARED.json')
    assert primary()==data['primary_before'],'Primary checkout changed before push'
    assert git('ls-remote','origin','refs/heads/main').split()[0]==data['base'],'Remote main moved: stop'
    current=files()
    assert {PREFIX+'/'+str(rel) for _,rel in current}==set(data['files']),'Source deliverable set changed after review'
    for src,rel in current:
        expected=data['files'].get(PREFIX+'/'+str(rel))
        assert expected and expected['sha256']==sha(src),('Source deliverable changed after review',str(rel))
    commit=data['commit']
    command=['git','push','origin',commit+':refs/heads/main']
    publog('push_attempt',commit=commit,command=command,force=False)
    print(git('push','origin',commit+':refs/heads/main'),flush=True)
    verify()

def verify():
    data=read(PUB/'PREPARED.json');commit=data['commit']
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    assert remote==commit,('Remote main verification failed',remote,commit)
    tree=git('ls-tree','-r','-z',commit,'--',PREFIX,binary=True).split(b'\0')
    records={}
    for row in tree:
        if not row:continue
        left,name=row.split(b'\t',1);mode,typ,oid=left.split();name=name.decode()
        assert mode==b'100644' and typ==b'blob'
        blob=git('cat-file','blob',oid.decode(),binary=True)
        h=hashlib.sha256(blob).hexdigest();expected=data['files'][name]
        assert h==expected['sha256'] and len(blob)==expected['bytes'],('Published blob differs',name)
        records[name]=h
    assert set(records)==set(data['files'])
    after=primary();assert after==data['primary_before'],'Primary checkout changed during publication'
    result=dict(verified=True,commit=commit,parent=data['base'],branch='main',remote='origin',
        remote_url=git('remote','get-url','origin'),url='https://github.com/CanelE452/pallet-6d-pose/commit/'+commit,
        prefix=PREFIX,published_files=len(records),all_published_blobs_SHA256_match=True,
        all_blobs_sha256=records,primary_checkout_preserved=True,primary_before=data['primary_before'],
        primary_after=after,verified_utc=now(),force=False)
    write(OUT/'publication/REMOTE_VERIFIED.json',result)
    publog('push_verified',commit=commit,published_files=len(records),primary_checkout_preserved=True)
    print('PUSH_VERIFIED',commit,len(records),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=('prepare','push','verify'));s=a.parse_args()
    {'prepare':prepare,'push':push,'verify':verify}[s.command]()
