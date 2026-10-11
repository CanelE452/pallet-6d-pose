"""Publish explicit frozen byte groups; ROOT alone executes this helper.

No science runs, producer JSON changes, force pushes, or automatic retries.
Only declared NEW ordered parts may be reclaimed after remote/blob proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

PARENT='bb1c3a97c46703cd960220afe969e11da698a381'
MAIN='f565418e6a63acd5fb920a5741308d84eb7c131f'
BRANCH='refs/heads/research/observation-refiner-robust-pnp-20261009'
MAX_GROUP=1073741824
MAX_FILE=100*1024*1024
PROTOCOL='SELECTED_PUBLICATION_PROTOCOL.json'
LIMITS=dict(explicit_selected_files_only=True,science_model_GT_RGB_gzip_decode_calls=0,
    force_push_rebase_main_write=False,original_source_repository_writes=False,
    reclaim_only_explicit_new_parts_after_push_and_blob_proof=True,automatic_retry=False)

def require(value,message):
    if not value:raise ValueError(message)

def safe(value):
    p=Path(value).absolute()
    require('..' not in p.parts and not any(q.is_symlink() for q in (p,*p.parents)),'unsafe/symlink '+str(p))
    return p

def relative(value):
    require(isinstance(value,str) and '\0' not in value and '\n' not in value,'invalid relative declaration')
    p=Path(value)
    require(not p.is_absolute() and '..' not in p.parts and len(p.parts)>2
        and p.parts[:2] in (('_docs','experiments'),('scripts','research')),'outside allowed public paths')
    return p.as_posix()

def digest(value):
    p=safe(value);require(p.is_file(),'missing regular file '+str(p));h=hashlib.sha256();n=0
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block);n+=len(block)
    require(n==p.stat().st_size,'file changed while hashing');return dict(bytes=n,sha256=h.hexdigest())

def binding(value):return dict(path=str(safe(value)),**digest(value))
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def read(value):return json.loads(safe(value).read_bytes())

def write(value,record,raw=False):
    p=safe(value);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream:
        stream.write(record if raw else (json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())

def command(repo,*args):return ['git','--no-optional-locks','--literal-pathspecs','-C',str(repo),*args]
def git(repo,*args,data=None):
    return subprocess.run(command(repo,*args),input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},check=True).stdout

def state(repo):
    operations={'index_entries':('ls-files','--stage','-z'),'tracked_diff':('diff','--no-ext-diff','--binary'),
        'staged_diff':('diff','--cached','--no-ext-diff','--binary'),'status':('status','--porcelain=v1','-z','-uall')}
    return dict(HEAD=git(repo,'rev-parse','HEAD').decode().strip(),
        **{name+'_sha256':hashlib.sha256(git(repo,*args)).hexdigest() for name,args in operations.items()})

def protected(a):return {str(root):state(root) for root in dict.fromkeys((a.original,a.source))}
def head(a):return git(a.stage,'rev-parse','HEAD').decode().strip()
def empty_index(a):require(git(a.stage,'diff','--cached','--name-only','-z')==b'','stage index must be empty')

def remote(a):
    result={}
    for line in git(a.stage,'ls-remote','--refs','origin',BRANCH,'refs/heads/main').decode().splitlines():
        sha,name=line.split();require(re.fullmatch('[0-9a-f]{40}',sha) and name not in result,'ambiguous remote proof');result[name]=sha
    require(set(result)=={BRANCH,'refs/heads/main'} and result['refs/heads/main']==MAIN,'remote main or branch missing/changed')
    return dict(dedicated=result[BRANCH],main=result['refs/heads/main'])

def nul_list(value):
    data=safe(value).read_bytes();require(not data or data.endswith(b'\0'),'NUL list must end in NUL')
    paths=[relative(part.decode('utf-8')) for part in data.split(b'\0')[:-1]]
    require(len(paths)==len(set(paths)),'duplicate explicit path');return sorted(paths)

def blob(a,commit,path):
    process=subprocess.Popen(command(a.stage,'cat-file','blob',commit+':'+path),stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});h=hashlib.sha256();n=0
    try:
        for block in iter(lambda:process.stdout.read(1048576),b''):h.update(block);n+=len(block)
        message=process.stderr.read();require(process.wait()==0,'committed blob unavailable '+path+' '+message.decode(errors='replace'))
    finally:
        process.stdout.close();process.stderr.close()
        if process.poll() is None:process.kill();process.wait()
    return dict(bytes=n,sha256=h.hexdigest())

def group_names(index):return ('GROUP_%03d_STARTED.json'%index,'GROUP_%03d_CHECKS.json'%index,'GROUP_%03d_PATHS.nul'%index)

def reclaim_identity(a,path,item):
    stage=safe(a.stage/path);copied=safe(a.copied_root/path)
    require(same(digest(stage),item) and same(digest(copied),item),'part aliases changed before reclaim')
    left,right=stage.stat(),copied.stat()
    require((left.st_dev,left.st_ino)==(right.st_dev,right.st_ino) and left.st_nlink==right.st_nlink==2,
        'reclaim requires exactly the two declared hardlink aliases')
    return dict(device=left.st_dev,inode=left.st_ino,nlink_before=2)

def configuration(a):return {name:str(getattr(a,name)) for name in ('stage','original','source','copied_root','paths','reclaim_paths','output')}|dict(chunk_bytes=a.chunk_bytes)

def guard(a):
    for name in ('stage','original','source','copied_root','paths','reclaim_paths','output'):setattr(a,name,safe(getattr(a,name)))
    require(all(root.is_dir() for root in (a.stage,a.original,a.source,a.copied_root)),'existing roots required')
    require(0<a.chunk_bytes<=MAX_GROUP,'group limit must be positive and at most 1 GiB')
    roots=(a.stage,a.copied_root,a.output)
    require(all(not x.is_relative_to(y) and not y.is_relative_to(x) for i,x in enumerate(roots) for y in roots[i+1:]),'disjoint owned roots')
    require(all(not x.is_relative_to(y) and not y.is_relative_to(x) for x in roots for y in (a.original,a.source)),'owned/protected roots overlap')
    require(all(not p.is_relative_to(a.output) for p in (a.paths,a.reclaim_paths,safe(__file__))),'own output contains an input')
    require(git(a.stage,'rev-parse','--abbrev-ref','HEAD').decode().strip()!='main','never commit on local main')

def freeze(a):
    require(a.group is None and not(a.output/PROTOCOL).exists(),'freeze group/prefix');empty_index(a)
    inputs=dict(helper=binding(__file__),paths=binding(a.paths),reclaim_paths=binding(a.reclaim_paths))
    before=protected(a);require(head(a)==PARENT and remote(a)==dict(dedicated=PARENT,main=MAIN),'exact initial parent/remote')
    paths=nul_list(a.paths);reclaims=nul_list(a.reclaim_paths);require(paths and set(reclaims)<=set(paths),'nonempty selected paths/reclaim subset')
    files=[dict(path=path,**digest(a.stage/path)) for path in paths];lookup={item['path']:item for item in files}
    require(all(item['bytes']<=MAX_FILE and item['bytes']<=a.chunk_bytes for item in files),'file/group byte limit')
    for path in reclaims:
        require('archives' in Path(path).parts and re.search(r'\.part[0-9]{3,}$',path)
            and git(a.stage,'ls-tree','-z',PARENT,'--',path)==b'','reclaim path must be a newly generated ordered part')
        reclaim_identity(a,path,lookup[path])
    groups=[];current=[];size=0
    for item in files:
        if current and size+item['bytes']>a.chunk_bytes:groups.append(current);current=[];size=0
        current.append(item['path']);size+=item['bytes']
    if current:groups.append(current)
    require(protected(a)==before and head(a)==PARENT and remote(a)==dict(dedicated=PARENT,main=MAIN),'preflight authority changed')
    require(inputs==dict(helper=binding(__file__),paths=binding(a.paths),reclaim_paths=binding(a.reclaim_paths)),'preflight source/input changed')
    write(a.output/PROTOCOL,dict(schema='explicit_selected_git_publication_protocol_v21',configuration=configuration(a),
        **inputs,protected=before,parent=PARENT,expected_main=MAIN,branch=BRANCH,
        remote_at_freeze=dict(dedicated=PARENT,main=MAIN),files=files,groups=groups,reclaim=reclaims,limits=LIMITS))

def run(a):
    own=read(a.output/PROTOCOL);require(own['schema']=='explicit_selected_git_publication_protocol_v21'
        and own['configuration']==configuration(a) and own['helper']==binding(__file__)
        and own['paths']==binding(a.paths) and own['reclaim_paths']==binding(a.reclaim_paths)
        and protected(a)==own['protected'],'frozen source/input/protected authority differs')
    require(a.group is not None and 1<=a.group<=len(own['groups']),'one exact group required');expected=PARENT;reclaimed=set()
    for index in range(1,a.group):
        previous=read(a.output/group_names(index)[1]);require(previous['complete'] is True and previous['passed'] is True
            and previous['error'] is None and previous['protocol']==binding(a.output/PROTOCOL)
            and previous['helper']==own['helper'] and previous['group']==index and previous['parent']==expected
            and previous['paths']==own['groups'][index-1] and previous['committed_blob_hashes_verified']==len(previous['paths'])
            and previous['commit_command_completed'] is True and previous['push_command_completed'] is True
            and previous['remote_after']==dict(dedicated=previous['HEAD'],main=MAIN),'earlier group needs complete remote proof')
        require([item['path'] for item in previous['reclaim_actions']]==[path for path in previous['paths'] if path in own['reclaim']]
            and all(item['stage_unlinked'] is True and item['copied_unlinked'] is True for item in previous['reclaim_actions']),'earlier reclaim completion differs')
        expected=previous['HEAD'];reclaimed.update(item['path'] for item in previous['reclaim_actions'])
    empty_index(a);require(head(a)==expected and remote(a)==dict(dedicated=expected,main=MAIN),'prior HEAD/remote mismatch')
    lookup={item['path']:item for item in own['files']}
    for item in own['files']:
        if item['path'] in reclaimed:
            require(not safe(a.stage/item['path']).exists() and not safe(a.copied_root/item['path']).exists()
                and same(blob(a,expected,item['path']),item),'earlier reclaimed part/blob differs')
        else:require(same(digest(a.stage/item['path']),item),'frozen selected bytes changed')
    names=group_names(a.group);require(all(not(a.output/name).exists() for name in names),'preserve attempted group; no automatic retry')
    selected=own['groups'][a.group-1];protocol=binding(a.output/PROTOCOL)
    write(a.output/names[2],b''.join((path+'\0').encode() for path in selected),raw=True)
    write(a.output/names[0],dict(protocol=protocol,helper=own['helper'],group=a.group,parent=expected,paths=selected,limits=LIMITS))
    error=None;commit=None;remote_after=None;reclaim_actions=[];committed_verified=0;commit_attempted=False;commit_completed=False;push_attempted=False;push_completed=False
    try:
        git(a.stage,'add','--pathspec-from-file='+str(a.output/names[2]),'--pathspec-file-nul')
        staged=git(a.stage,'diff','--cached','--name-only','-z').decode().split('\0')[:-1]
        require(sorted(staged)==selected,'staged names must be exactly this group')
        for path in selected:require(same(digest(a.stage/path),lookup[path]),'group bytes changed after add')
        commit_attempted=True;git(a.stage,'-c','gc.auto=0','commit','-m','Publish selected pallet evidence group %d/%d'%(a.group,len(own['groups'])))
        commit_completed=True;commit=head(a);empty_index(a)
        changed=git(a.stage,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',expected,commit).decode().split('\0')[:-1]
        require(sorted(changed)==selected and protected(a)==own['protected'],'commit selection/protected state changed')
        for path in selected:require(same(blob(a,commit,path),lookup[path]),'committed blob differs before push')
        require(remote(a)==dict(dedicated=expected,main=MAIN),'remote changed before normal push')
        push_attempted=True;git(a.stage,'-c','pack.window=0','-c','pack.depth=0','-c','pack.threads=1','-c','gc.auto=0','push','origin','HEAD:'+BRANCH)
        push_completed=True;remote_after=remote(a);require(remote_after==dict(dedicated=commit,main=MAIN),'pushed remote proof differs')
        for path in selected:
            require(same(blob(a,commit,path),lookup[path]),'committed blob differs after push');committed_verified+=1
        require(protected(a)==own['protected'],'protected repositories changed')
        parts=[path for path in selected if path in own['reclaim']]
        identities={path:reclaim_identity(a,path,lookup[path]) for path in parts}
        if parts:
            git(a.stage,'update-index','--skip-worktree','-z','--stdin',data=b''.join((path+'\0').encode() for path in parts))
            flags=git(a.stage,'ls-files','-v','-z','--',*parts).decode().split('\0')[:-1]
            require(sorted(flags)==sorted('S '+path for path in parts),'skip-worktree flags not set exactly')
        for path in parts:
            action=dict(path=path,**identities[path],stage_unlinked=False,copied_unlinked=False);reclaim_actions.append(action)
            require(reclaim_identity(a,path,lookup[path])==identities[path],'part link identity changed')
            safe(a.stage/path).unlink();action['stage_unlinked']=True
            other=safe(a.copied_root/path);st=other.stat();require(st.st_nlink==1
                and (st.st_dev,st.st_ino)==(action['device'],action['inode']),'remaining owned alias changed')
            other.unlink();action['copied_unlinked']=True
        require(head(a)==commit and remote(a)==remote_after and protected(a)==own['protected'],'final remote/protected authority changed')
        require(own['helper']==binding(__file__) and own['paths']==binding(a.paths)
            and own['reclaim_paths']==binding(a.reclaim_paths) and binding(a.output/PROTOCOL)==protocol,'frozen code/inputs/protocol changed')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    checks=dict(schema='explicit_selected_git_publication_group_checks_v21',complete=error is None,passed=error is None,error=error,
        protocol=protocol,helper=own['helper'],group=a.group,parent=expected,HEAD=commit,remote_after=remote_after,
        paths=selected,pathspec=binding(a.output/names[2]),committed_blob_hashes_verified=committed_verified,
        commit_command_attempted=commit_attempted,commit_command_completed=commit_completed,
        push_command_attempted=push_attempted,push_command_completed=push_completed,reclaim_actions=reclaim_actions,stage_HEAD_at_receipt=head(a),
        protected_after=protected(a),limits=LIMITS,partial_failure_preserved=True)
    write(a.output/names[1],checks);print(json.dumps(dict(group=a.group,complete=checks['complete'],passed=checks['passed'],
        HEAD=commit,remote=remote_after,files=len(selected),reclaimed=len(reclaim_actions),error=error)),flush=True)
    if error:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','run'))
    for name in ('stage','original','source','copied-root','paths','reclaim-paths','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--group',type=int);p.add_argument('--chunk-bytes',type=int,default=MAX_GROUP)
    a=p.parse_args();guard(a);freeze(a) if a.mode=='freeze' else run(a)

if __name__=='__main__':main()
