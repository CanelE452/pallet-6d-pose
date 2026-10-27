"""Publish a new reviewed prefix from a wholly isolated Git repository.

Code-only scaffold until ROOT_PUBLICATION_GO. Never run Git in the dirty primary
repository, never check out its tracked files, and never modify its HEAD/index.
Private isolated index commit-tree and non-force push; maximum3 remote retries.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import publication_common as P

PUB=P.OUT/'publication/merged372_v1';PAYLOAD=PUB/'payload'
GO=P.OUT/'preflight/ROOT_PUBLICATION_GO.json'
PRIMARY=Path('/home/minjae/Documents/github/pallet-pose')


def primary_metadata():
    gitdir=PRIMARY/'.git';P.require(gitdir.is_dir() and not gitdir.is_symlink(),'Unexpected primary Git metadata layout; ROOT review required')
    paths=[gitdir/'HEAD',gitdir/'index',gitdir/'packed-refs',gitdir/'refs/heads/main']
    return {str(p):P.sha(p) if p.is_file() else None for p in paths}


def run(repo,args,env=None,input_bytes=None,check=True):
    P.require(repo.resolve().is_relative_to(PUB) and not repo.resolve().is_relative_to(PRIMARY),'Git cwd must be isolated')
    isolated_env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')};isolated_env['GIT_TERMINAL_PROMPT']='0'
    if env:
        P.require(set(env)=={'GIT_INDEX_FILE'} and Path(env['GIT_INDEX_FILE']).parent==PUB,'Unexpected Git environment override')
        isolated_env.update(env)
    result=subprocess.run(['git',*args],cwd=repo,env=isolated_env,input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check:P.require(result.returncode==0,'Isolated Git command failed: '+args[0]+'; '+result.stderr.decode(errors='replace')[:1000])
    return result


def text(repo,args,env=None):return run(repo,args,env).stdout.decode().strip()


def verify(repo,revision,prefix,inventory):
    raw=run(repo,['ls-tree','-r','-z','--name-only',revision,'--',prefix]).stdout
    paths={p.decode() for p in raw.split(b'\0') if p}
    P.require(paths=={prefix+'/'+p for p in inventory},'Published prefix inventory differs')
    for rel,item in inventory.items():
        blob=run(repo,['show',revision+':'+prefix+'/'+rel]).stdout
        import hashlib
        P.require(hashlib.sha256(blob).hexdigest()==item['sha256'] and len(blob)==item['bytes'],'Published blob differs: '+rel)


def main():
    io=P.Inputs();io.gates('publish');go=json.loads(GO.read_text());go_sha=P.sha(GO)
    P.require(go.get('authorized') is True and go.get('action')=='publish_new_primary372_prefix'
        and go['publication_inputs_sha256']==io.manifest_sha
        and go['publisher_sha256']==P.sha(Path(__file__).resolve())
        and go['helper_sha256']==P.sha(Path(P.__file__).resolve()),'Explicit reviewed ROOT publication GO missing/different')
    prefix=go['prefix'];P.safe_relative(prefix)
    P.require(prefix.startswith('_docs/experiments/pallet_dope_') and len(Path(prefix).parts)==3,'Unreviewed target prefix')
    P.require(go['branch']=='main' and go['remote_url']=='https://github.com/CanelE452/pallet-6d-pose.git','Unreviewed remote/branch')
    inventory=json.loads((PUB/'PAYLOAD_INVENTORY.json').read_text());prepare=json.loads((PUB/'PREPARE_QA.json').read_text())
    numerical=P.OUT/'preflight/FINAL_NUMERICAL_QA.json';nqa=json.loads(numerical.read_text())
    P.require(prepare['passed'] is True and prepare['complete'] is True and nqa['passed'] is True
        and prepare['publication_inputs_sha256']==io.manifest_sha and prepare['final_numerical_QA_sha256']==P.sha(numerical)
        and go['prepare_QA_sha256']==P.sha(PUB/'PREPARE_QA.json')
        and go['payload_inventory_sha256']==P.sha(PUB/'PAYLOAD_INVENTORY.json')
        and go['final_numerical_QA_sha256']==P.sha(numerical),'Publication QA authority changed')
    P.require(P.inventory(PAYLOAD)==inventory and len(inventory)==prepare['payload_files'],'Payload changed after preparation')
    required_sources={str(Path(__file__).resolve()),str(Path(P.__file__).resolve()),
        str(P.OUT/'code/prepare_publication.py'),str(P.OUT/'code/final_numerical_qa.py')}
    P.require(required_sources.issubset(go['source_sha256']),'Publisher/helper/preparation/numerical sources must be frozen')
    for path,digest in go['source_sha256'].items():
        p=Path(path);P.require(p.is_relative_to(P.OUT/'code') and P.sha(p)==digest,'Publisher source freeze changed')
    for path,digest in prepare['source_sha256'].items():
        if path==str(P.INPUTS):P.require(P.sha(path)==digest,'Publication contract changed')
        else:P.require(P.sha(io.path(path))==digest,'Prepared source changed')
    before=primary_metadata();isolated=PUB/'isolated_publish_checkout'
    P.require(not isolated.exists(),'Prior isolated publication preserved; ROOT review required before retry')
    isolated.mkdir();run(isolated,['init','--quiet'])
    run(isolated,['remote','add','origin',go['remote_url']])
    run(isolated,['config','user.name',go['git_author_name']]);run(isolated,['config','user.email',go['git_author_email']])
    target=isolated/prefix;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(PAYLOAD,target)
    index=PUB/'isolated_git_index';env={'GIT_INDEX_FILE':str(index)}
    attempts=[];pushed=None

    def frozen():
        io.unchanged();P.require(P.sha(GO)==go_sha and P.inventory(PAYLOAD)==inventory,'Publication inputs/payload changed')
        P.require(P.sha(PUB/'PREPARE_QA.json')==go['prepare_QA_sha256']
            and P.sha(PUB/'PAYLOAD_INVENTORY.json')==go['payload_inventory_sha256']
            and P.sha(numerical)==go['final_numerical_QA_sha256'],'Approved publication QA/inventory changed')
        P.require(primary_metadata()==before,'Primary Git HEAD/index metadata changed concurrently')
        for path,digest in go['source_sha256'].items():P.require(P.sha(path)==digest,'Source changed while publishing')

    for attempt in range(1,4):
        frozen();run(isolated,['fetch','--no-tags','origin','refs/heads/main']);base=text(isolated,['rev-parse','FETCH_HEAD'])
        for rel in ('AGENTS.md','_docs/AGENTS.md','_docs/experiments/AGENTS.md',prefix+'/AGENTS.md'):
            present=run(isolated,['cat-file','-e',base+':'+rel],check=False).returncode==0
            P.require(not present,'Remote AGENTS instructions require ROOT review before publishing: '+rel)
        P.require(not run(isolated,['ls-tree','--name-only',base,'--',prefix]).stdout,'Target prefix already exists; never overwrite it')
        run(isolated,['read-tree',base],env);run(isolated,['add','--',prefix],env)
        tree=text(isolated,['write-tree'],env)
        changes=text(isolated,['diff-tree','--no-commit-id','--name-status','-r',base,tree])
        P.require(set(changes.splitlines())=={'A\t'+prefix+'/'+rel for rel in inventory},'Commit includes changes outside exact additions')
        verify(isolated,tree,prefix,inventory)
        commit=run(isolated,['commit-tree',tree,'-p',base],env,input_bytes=(go['commit_message'].rstrip()+'\n').encode()).stdout.decode().strip()
        remote=text(isolated,['ls-remote','origin','refs/heads/main']).split()[0]
        entry=dict(attempt=attempt,base=base,tree=tree,commit=commit,remote_before_push=remote);attempts.append(entry)
        if remote!=base:entry['status']='remote_advanced_before_push';continue
        frozen();result=run(isolated,['push','origin',commit+':refs/heads/main'],check=False)
        entry['push_exit_code']=result.returncode
        observed=text(isolated,['ls-remote','origin','refs/heads/main']).split()[0]
        run(isolated,['fetch','--no-tags','origin','refs/heads/main']);head=text(isolated,['rev-parse','FETCH_HEAD'])
        ancestor=run(isolated,['merge-base','--is-ancestor',commit,head],check=False).returncode==0
        if ancestor:
            for rel in ('AGENTS.md','_docs/AGENTS.md','_docs/experiments/AGENTS.md',prefix+'/AGENTS.md'):
                P.require(run(isolated,['cat-file','-e',head+':'+rel],check=False).returncode!=0,'Final remote AGENTS requires ROOT review: '+rel)
            verify(isolated,head,prefix,inventory)
            P.require(text(isolated,['ls-remote','origin','refs/heads/main']).split()[0]==head,'Push reached remote but final verification raced; preserve artifacts for ROOT review')
            pushed=dict(commit=commit,observed_remote=observed,verified_remote=head);entry['status']='verified_published';break
        P.require(observed!=base,'Push failed without a remote race; preserve isolated artifacts for ROOT review')
        entry['status']='nonforce_remote_race_retry'
    P.require(pushed is not None,'Three non-force attempts exhausted; no overwrite/force fallback')
    frozen();P.write(PUB/'PUSH_VERIFIED.json',dict(complete=True,pushed=True,**pushed,prefix=prefix,attempts=attempts,
        payload_files=len(inventory),payload_inventory_sha256=P.sha(PUB/'PAYLOAD_INVENTORY.json'),
        ROOT_publication_GO_sha256=go_sha,primary_metadata_before=before,primary_metadata_after=primary_metadata(),
        primary_working_files_written=False,primary_working_tree_checkout_performed=False,primary_Git_commands_executed=0,
        isolated_checkout=str(isolated),all_public_blob_SHA_verified=True,source_sha256=go['source_sha256']))


if __name__=='__main__':main()
