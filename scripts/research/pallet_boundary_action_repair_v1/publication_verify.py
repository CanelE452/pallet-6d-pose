"""Verify a pushed result commit using actual network and immutable A blobs.

This tool never stages, commits, pushes, merges, resets, or edits a worktree.
Its only Git reference mutation is an explicit no-tags fetch of origin/main.
Raw command output stays in PRIVATE; canonical proofs contain hashes only.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

PUBLIC_ROOT='_docs/experiments/pallet_boundary_action_repair_v1'
CODE_ROOT='scripts/research/pallet_boundary_action_repair_v1'
INDEX_PATHS={'readme.md','README.md','_docs/README.md','_docs/INDEX.md','_docs/experiments/README.md','_docs/experiments/INDEX.md'}
HASH=re.compile(r'^[0-9a-f]{64}$');COMMIT=re.compile(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})$')


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write_new(path,data):
    with Path(path).open('xb')as f:f.write(data if isinstance(data,bytes)else encoded(data))
def require(value,message):
    if not value:raise ValueError(message)
def relative(value):
    require(isinstance(value,str)and value and not any(c in value for c in('\0','\n','\r',':','\\')),'Unsafe relative path')
    p=PurePosixPath(value)
    require(not p.is_absolute()and '..'not in p.parts and str(p)==value and value!='.','Non-canonical relative path')
    return value
def inside(root,value):
    result=(root/relative(value)).resolve()
    require(result.is_relative_to(root),'Path escaped its registered root')
    return result
def exit_zero(value):return type(value)is int and value==0


class Git:
    def __init__(self,repo,attempt):
        self.repo=repo;self.attempt=attempt;self.commands=[];self.env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def call(self,args,*,cwd=None,check=True,label=None):
        label=label or f'command_{len(self.commands):03d}'
        start=time.monotonic();result=subprocess.run(['git',*args],cwd=cwd or self.repo,env=self.env,capture_output=True)
        out=self.attempt/(label+'.stdout');err=self.attempt/(label+'.stderr')
        write_new(out,result.stdout);write_new(err,result.stderr)
        self.commands.append(dict(argv=['git',*args],cwd=str(cwd or self.repo),exit_code=result.returncode,
            stdout_path=str(out.name),stderr_path=str(err.name),stdout_sha256=sha(out),stderr_sha256=sha(err),wall_seconds=time.monotonic()-start))
        if check:require(result.returncode==0,'Git command failed; inspect the private command receipt: '+label)
        return result
    def text(self,args,**kwargs):return self.call(args,**kwargs).stdout.decode().strip()
    def ancestor(self,a,b):
        r=self.call(['merge-base','--is-ancestor',a,b],check=False)
        require(r.returncode in(0,1),'Git ancestor check did not complete')
        return r.returncode==0


def blob_hashes(git,commit,paths,capture=()):
    """Stream Git blobs through one bounded-memory, read-only cat-file child."""
    values={};payloads={};err=git.attempt/'git_blob_batch.stderr';start=time.monotonic()
    with err.open('xb')as errors:
        child=subprocess.Popen(['git','cat-file','--batch'],cwd=git.repo,env=git.env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        try:
            for path in sorted(paths):
                child.stdin.write((commit+':'+path+'\n').encode());child.stdin.flush()
                header=child.stdout.readline().decode().strip().split()
                require(len(header)==3 and header[1]=='blob','Expected an actual A blob: '+path)
                size=int(header[2]);remaining=size;digest=hashlib.sha256();chunks=[]
                while remaining:
                    block=child.stdout.read(min(remaining,1024*1024));require(bool(block),'Truncated Git blob')
                    digest.update(block);remaining-=len(block)
                    if path in capture:chunks.append(block)
                require(child.stdout.read(1)==b'\n','Malformed Git batch separator')
                values[path]=dict(sha256=digest.hexdigest(),git_blob_oid=header[0],bytes=size)
                if path in capture:payloads[path]=b''.join(chunks)
            child.stdin.close();require(child.stdout.read()==b'','Unexpected Git batch trailing output')
            code=child.wait();require(code==0,'Git blob batch failed')
        finally:
            if child.poll()is None:child.kill();child.wait()
            if not child.stdin.closed:child.stdin.close()
            child.stdout.close()
    git.commands.append(dict(argv=['git','cat-file','--batch'],exit_code=code,stderr_path=err.name,
        stderr_sha256=sha(err),blob_count=len(values),wall_seconds=time.monotonic()-start))
    return values,payloads


def preserved_original(git,run,prepush,start,attempt_id):
    original=Path(prepush['original_worktree_root']).resolve()
    require(original!=git.repo,'Original user worktree must be distinct from the research worktree')
    require(git.text(['rev-parse','--show-toplevel'],cwd=original)==str(original),'Original worktree root mismatch')
    head=git.text(['rev-parse','HEAD'],cwd=original)
    command=['git','status','--porcelain=v1','-z']
    status=git.call(command[1:],cwd=original,label='original_worktree_same_command_status').stdout
    protected={path:sha(inside(original,path))for path in start['protected_files_sha256']}
    receipt=dict(schema='ORIGINAL_WORKTREE_PUBLICATION_READONLY_RECHECK_V1',generated_utc=utc(),command=command,
        HEAD=head,HEAD_matches_start=head==start['original_HEAD'],status_sha256=sha_bytes(status),
        start_status_sha256=start['status_sha256'],status_bytes_match_start=sha_bytes(status)==start['status_sha256'],
        protected_sha256=protected,all_protected_bytes_match_start=protected==start['protected_files_sha256'],
        source_receipt_sha256=sha(run/'START_BEFORE_FETCH.json'),agent_writes_to_original_user_worktree=False,
        optional_git_index_refresh_disabled=True)
    path=run/('ORIGINAL_WORKTREE_RECHECK_AT_PUBLICATION_'+attempt_id+'.json');write_new(path,receipt)
    require(receipt['HEAD_matches_start']and receipt['status_bytes_match_start']and receipt['all_protected_bytes_match_start'],
        'Original user worktree differs from START; attribution is not inferred')
    return path,receipt


def verify(repo_root,run_dir,result_commit,prepush_receipt,push_receipt,stage_allowlist,*,GO=False):
    if not GO:raise PermissionError('--verification-GO is required for actual fetch and proof output')
    started=time.monotonic();repo=Path(repo_root).resolve();run=Path(run_dir).resolve()
    require(COMMIT.fullmatch(result_commit)is not None,'Use a full result commit SHA')
    require(not(run/'PUBLICATION_GIT_PROOF.json').exists()and not(run/'REMOTE_PUBLICATION_RECEIPT.json').exists(),
        'Preserve immutable canonical A proofs; use a separate later verification receipt')
    ident=uuid.uuid4().hex;attempt=run/'publication_verification_attempts'/ident;attempt.mkdir(parents=True)
    git=Git(repo,attempt)
    try:
        spec=read(run/'FINAL_PUBLICATION_PROOF_SCHEMA_B.json')
        require(spec['schema']=='FINAL_PUBLICATION_PROOF_SCHEMA_SPEC_B_V1'and spec['is_schema_spec_only']is True,'Unexpected canonical schema specification')
        require(spec['remote_receipt_expected_shape']['schema']=='REMOTE_RESULT_COMMIT_VERIFICATION_V1'and spec['git_proof_expected_shape']['schema']=='SCOPED_RESEARCH_PUBLICATION_GIT_PROOF_V1','Canonical schema mismatch')
        prepath=Path(prepush_receipt).resolve();pushpath=Path(push_receipt).resolve();allowpath=Path(stage_allowlist).resolve()
        for path in(prepath,pushpath,allowpath):require(path.is_relative_to(run),'Input receipts must be rooted in PRIVATE run-dir')
        pre=read(prepath);push=read(pushpath);allow=read(allowpath)
        require(pre['schema']=='RESULT_A_PREPUSH_RECEIPT_V1'and push['schema']=='RESULT_A_PUSH_RECEIPT_V1'and allow['schema']=='RESULT_A_STAGE_ALLOWLIST_V1','Input receipt schema mismatch')
        require(Path(pre['repo_root']).resolve()==Path(allow['repo_root']).resolve()==repo,'Receipt research root mismatch')
        require(pre['result_commit']==push['result_commit']==push['source_commit']==result_commit,'Result A binding mismatch')
        require(pre['stage_allowlist_sha256']==sha(allowpath),'Stage allowlist bytes changed')
        require(allow['public_root']==PUBLIC_ROOT and allow['allowed_code_roots']==[CODE_ROOT],'Publication roots must be the new namespaces')
        indices={relative(p)for p in allow['allowed_index_paths']};require(indices<=INDEX_PATHS,'Only explicit existing documentation indices are allowed')
        expected={}
        for item in allow['paths']:
            path=relative(item['path']);require(path not in expected and HASH.fullmatch(item['sha256'])is not None,'Duplicate path or invalid expected SHA256')
            require(path.startswith(PUBLIC_ROOT+'/')or path.startswith(CODE_ROOT+'/')or path in indices,'Path outside the explicit publication scopes')
            expected[path]=item['sha256']
        require(bool(expected),'Empty stage/artifact allowlist')
        require(git.text(['rev-parse','--show-toplevel'])==str(repo),'Research repository root mismatch')
        require(git.text(['rev-parse',result_commit+'^{commit}'])==result_commit,'Result A is not an existing exact commit')
        mid=read(run/'MIDRUN_GIT_STATE.json');start=read(run/'START_BEFORE_FETCH.json')
        base=pre['base_commit'];require(COMMIT.fullmatch(base)is not None and base==mid['execution_base'],'Execution base binding mismatch')
        require(git.ancestor(base,result_commit),'Result A does not descend from the actual execution base')
        before=pre['remote_main_before'];require(COMMIT.fullmatch(before)is not None and before==push['remote_main_before'],'Prepush remote binding mismatch')
        require(exit_zero(pre['fetch_exit_code'])and pre['fetch_argv']==['git','fetch','--no-tags','origin','main'],'Actual prepush no-tags fetch receipt required')
        require(all(pre[k]is True for k in('related_diff_reviewed','affected_rechecks_passed','failures_and_limitations_visible')),'Actual root review/recheck attestations required')
        require(git.ancestor(before,result_commit),'Result A is not a fast-forward descendant of prepush main')
        require(push['push_argv']==['git','push','--porcelain','origin',result_commit+':refs/heads/main']and exit_zero(push['exit_code'])and push['force_used']is False and push['normal_fast_forward']is True,'Actual normal non-force A-to-main push receipt required')
        for stream in('stdout','stderr'):
            require(sha(inside(run,push[stream+'_path']))==push[stream+'_sha256'],'Actual push output log hash mismatch')
        raw=inside(run,pre['staged_paths_raw_path']).read_bytes()
        require(sha_bytes(raw)==pre['staged_paths_stdout_sha256'],'Recorded cached-path bytes changed')
        require(not raw or raw.endswith(b'\0'),'Captured cached-path stream must retain its actual NUL terminator')
        staged=raw.rstrip(b'\0').decode().split('\0')if raw else[]
        require(len(staged)==len(set(staged))and set(staged)==set(pre['staged_paths'])and set(staged)<=set(expected),'Actual staged paths differ from the explicit allowlist')
        parent=git.text(['rev-parse',result_commit+'^1'])
        changed=git.call(['diff','--no-renames','--name-only','-z',parent,result_commit,'--']).stdout
        changed=set(changed.rstrip(b'\0').decode().split('\0'))if changed else set()
        require(changed==set(staged),'Actual result-A commit paths differ from captured cached paths')
        net=git.call(['diff','--no-renames','--name-only','-z',before,result_commit,'--']).stdout
        net=set(net.rstrip(b'\0').decode().split('\0'))if net else set()
        require(net<=set(expected),'Unpublished net changes include a protected/unallowlisted path')
        original_path,_=preserved_original(git,run,pre,start,ident)
        first=git.call(['ls-remote','origin','refs/heads/main'],label='network_ls_remote_before_fetch')
        fetched=git.call(['fetch','--no-tags','origin','main'],label='network_fetch_origin_main')
        fetched_sha=git.text(['rev-parse','FETCH_HEAD'])
        last=git.call(['ls-remote','origin','refs/heads/main'],label='network_ls_remote_after_fetch')
        lines=last.stdout.decode().strip().splitlines()
        require(len(lines)==1 and lines[0].split()[1]=='refs/heads/main','Actual remote main response missing or ambiguous')
        remote_sha=lines[0].split()[0];require(remote_sha==fetched_sha,'Remote main advanced during verification; preserve attempt and retry separately')
        require(git.ancestor(result_commit,remote_sha),'Actual remote main does not contain result A')
        tree=git.call(['ls-tree','-r','-z',result_commit,'--',PUBLIC_ROOT,CODE_ROOT,*sorted(indices)]).stdout
        modes={}
        for entry in tree.split(b'\0'):
            if not entry:continue
            info,path=entry.split(b'\t',1);mode,kind,oid=info.decode().split();path=path.decode()
            if path in expected:require(kind=='blob'and mode in('100644','100755'),'Symlink/non-blob public artifact forbidden');modes[path]=mode
        require(set(modes)==set(expected),'Allowlisted A artifact is absent or not a blob')
        cpath=PUBLIC_ROOT+'/PUBLICATION_CHECK.json';mpath=PUBLIC_ROOT+'/PUBLICATION_MANIFEST.json'
        require(cpath in expected and mpath in expected,'A must include its actual offline check and manifest')
        blobs,payloads=blob_hashes(git,result_commit,expected,capture=(cpath,mpath))
        artifacts=[dict(path=p,result_commit=result_commit,expected_sha256=expected[p],remote_blob_sha256=blobs[p]['sha256'],match=blobs[p]['sha256']==expected[p])for p in sorted(expected)]
        require(all(a['match']for a in artifacts),'Fetched A blob differs from the precommit expected file SHA256')
        check=json.loads(payloads[cpath]);manifest=json.loads(payloads[mpath])
        require(check['status']=='PASS'and check['manifest_sha256']==blobs[mpath]['sha256']and manifest['complete_preparation']is True,'A offline C1/M1 pair failed or is inconsistent')
        manifest_files={PUBLIC_ROOT+'/'+relative(a['path']):a['sha256']for a in manifest['artifacts']}
        require(len(manifest_files)==len(manifest['artifacts']),'Duplicate A manifest artifact path')
        all_public={entry.split(b'\t',1)[1].decode()for entry in tree.split(b'\0')if entry and entry.split(b'\t',1)[1].decode().startswith(PUBLIC_ROOT+'/')}
        require(set(manifest_files)==all_public-{cpath,mpath},'A manifest does not cover the complete public tree')
        require(all(p in expected and expected[p]==v for p,v in manifest_files.items()),'A manifest artifact differs from the rooted expected hashes')
        snapshot=run/'publication_snapshots/result_A';snapshot.mkdir(parents=True,exist_ok=True)
        for path,payload in[(snapshot/'PUBLICATION_CHECK.json',payloads[cpath]),(snapshot/'PUBLICATION_MANIFEST.json',payloads[mpath])]:
            require(not path.is_symlink(),'Immutable snapshot must not be a symlink')
            if path.exists():require(path.read_bytes()==payload,'Preserve existing immutable A snapshot; bytes differ')
            else:write_new(path,payload)
        head=git.text(['rev-parse','HEAD']);current=[]
        for path,value in expected.items():
            item=repo/path;actual=sha(item)if item.is_file()and not item.is_symlink()else None
            current.append(dict(path=path,current_sha256=actual,result_A_sha256=value,match=actual==value))
        if head==result_commit:require(all(r['match']for r in current),'Actual result-A worktree files differ from their committed expected bytes')
        proof=dict(schema='SCOPED_RESEARCH_PUBLICATION_GIT_PROOF_V1',prework_state_recorded=True,
            prework_receipt_sha256={name:sha(run/name)for name in('START_BEFORE_FETCH.json','MIDRUN_GIT_STATE.json')},
            explicit_stage_allowlist=sorted(expected),stage_only_allowlisted=True,result_commit=result_commit,base_commit=base,
            failures_and_limitations_visible=pre['failures_and_limitations_visible'],
            prepush=dict(fetch_exit_code=pre['fetch_exit_code'],remote_main_before=before,related_diff_reviewed=True,affected_rechecks_passed=True,
                receipt_sha256=sha(prepath),review_scope='Root actual review/recheck attestations, byte-bound here; this Git verifier does not re-read scientific accuracy results.'),
            push=dict(exit_code=push['exit_code'],force_used=False,normal_fast_forward=True,result_descends_remote_before=True,receipt_sha256=sha(pushpath)),
            preservation=dict(original_reports_or_failure_records_changed=False,operating_defaults_changed=False,papers_changed=False,agent_modified_original_user_worktree=False),
            preservation_evidence=[dict(path=str(original_path.relative_to(run)),sha256=sha(original_path),scope='Actual original HEAD/status --porcelain=v1 -z/protected-file hashes match START; verifier issued no original worktree mutation.')],
            stage_allowlist_sha256=sha(allowpath),actual_result_changed_paths=sorted(changed),network_attempt_directory=str(attempt.relative_to(run)),
            command_receipt_sha256=sha_bytes(encoded(git.commands)),source_sha256=sha(__file__),schema_spec_sha256=sha(run/'FINAL_PUBLICATION_PROOF_SCHEMA_B.json'),
            current_worktree_comparison=dict(HEAD=head,required_to_match_A=head==result_commit,files=current,scope='A blobs and immutable precommit hashes remain authoritative; later B metadata bytes are not required to equal A.'),
            verification_performed_utc=utc(),wall_seconds=time.monotonic()-started,reference_or_accuracy_reads=0)
        write_new(attempt/'COMMANDS.json',git.commands)
        proof_path=run/'PUBLICATION_GIT_PROOF.json';write_new(proof_path,proof)
        remote=dict(schema='REMOTE_RESULT_COMMIT_VERIFICATION_V1',status='PUSHED_MAIN_VERIFIED',result_commit=result_commit,base_commit=base,
            remote_main_sha=remote_sha,remote_main_contains_result_commit=True,network_ls_remote_exit_code=last.returncode,
            network_fetch_exit_code=fetched.returncode,remote_artifacts_verified=True,git_proof_sha256=sha(proof_path),verified_artifacts=artifacts,
            offline_check_snapshot=dict(result_commit=result_commit,path='publication_snapshots/result_A/PUBLICATION_CHECK.json',sha256=sha(snapshot/'PUBLICATION_CHECK.json'),
                manifest_path='publication_snapshots/result_A/PUBLICATION_MANIFEST.json',manifest_sha256=sha(snapshot/'PUBLICATION_MANIFEST.json'),status='PASS'),
            verified_utc=utc(),source_sha256=sha(__file__),reference_or_accuracy_reads=0)
        write_new(run/'REMOTE_PUBLICATION_RECEIPT.json',remote)
        write_new(attempt/'RESULT.json',dict(status='PASS',result_commit=result_commit,remote_main_sha=remote_sha,
            git_proof_sha256=sha(proof_path),remote_receipt_sha256=sha(run/'REMOTE_PUBLICATION_RECEIPT.json'),wall_seconds=time.monotonic()-started))
        return remote
    except BaseException as exc:
        write_new(attempt/'FAILURE.json',dict(schema='PUBLICATION_VERIFICATION_ATTEMPT_FAILURE_V1',status='FAIL',error=repr(exc),
            traceback=traceback.format_exc(),commands=git.commands,result_commit=result_commit,created_utc=utc(),wall_seconds=time.monotonic()-started,
            actual_remote_publication_verified=False,canonical_proofs_not_overwritten=True,reference_or_accuracy_reads=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--result-commit',required=True)
    p.add_argument('--prepush-receipt',required=True);p.add_argument('--push-receipt',required=True);p.add_argument('--stage-allowlist',required=True);p.add_argument('--verification-GO',action='store_true')
    a=p.parse_args();r=verify(a.repo_root,a.run_dir,a.result_commit,a.prepush_receipt,a.push_receipt,a.stage_allowlist,GO=a.verification_GO)
    print(json.dumps(dict(status=r['status'],result_commit=r['result_commit'],remote_main_sha=r['remote_main_sha'],artifacts=len(r['verified_artifacts']))))
