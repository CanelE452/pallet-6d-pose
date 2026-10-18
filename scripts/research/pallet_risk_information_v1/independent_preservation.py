"""Read-only original worktree and original manifest byte preservation audit."""
from __future__ import annotations
import argparse, hashlib, json, os, resource, subprocess, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(repo,run_dir,output,*,untracked_mode='all',prior_failure=None):
    started,cpu=time.monotonic(),time.process_time();repo,run_dir,output=map(lambda p:Path(p).resolve(),(repo,run_dir,output))
    if output.exists():raise FileExistsError('Preserve previous verification: '+str(output))
    start_path=run_dir/'START.json';status_before=run_dir/'ORIGINAL_STATUS.bin';manifest_path=run_dir/'independent_p0_v1/INPUTS_PRIVATE.json'
    start=json.loads(start_path.read_text());manifest=json.loads(manifest_path.read_text());commands=[];checks=[]
    def check(name,value,detail=None):checks.append(dict(check=name,PASS=bool(value),detail=detail))
    def git(*args):
        argv=['git','--no-optional-locks','-C',str(repo),*args]
        result=subprocess.run(argv,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        commands.append(dict(argv=argv,GIT_OPTIONAL_LOCKS='0',exit_code=result.returncode,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),stderr_utf8=result.stderr.decode(errors='replace')))
        if result.returncode:raise RuntimeError('Read-only Git command failed: '+repr(argv))
        return result.stdout
    index_path=Path(git('rev-parse','--git-path','index').decode().strip())
    if not index_path.is_absolute():index_path=repo/index_path
    index_before=sha(index_path)
    if untracked_mode not in('all','normal'):raise ValueError('Explicit supported status view required')
    # START used the default normal directory-summary view. The first audit
    # intentionally preserves the failed all-files comparison; recovery uses
    # the identical original invocation, without changing any worktree bytes.
    status_args=['status','--porcelain=v1','-z']
    if untracked_mode=='all':status_args.append('--untracked-files=all')
    actual_status=git(*status_args)
    actual_status_sha=hashlib.sha256(actual_status).hexdigest();before=status_before.read_bytes()
    check('START_status_bytes_binding',hashlib.sha256(before).hexdigest()==start['status_sha256'])
    check('exact_original_status_bytes',actual_status==before,dict(before_bytes=len(before),after_bytes=len(actual_status)))
    head=git('rev-parse','HEAD').decode().strip();main=git('rev-parse','refs/heads/main').decode().strip()
    check('original_HEAD_unchanged',head==start['original_head'],head)
    check('local_main_unchanged',main==start['original_head'],main)
    protected=[]
    for relative,digest in start['protected_hashes'].items():
        path=repo/relative;actual=sha(path);protected.append(dict(relative_path=relative,expected_sha256=digest,actual_sha256=actual));check('protected_file_exact',actual==digest,relative)
    check('six_protected_files',len(protected)==6)
    archived=[]
    for key,row in manifest['inputs'].items():
        path=Path(row['path']);actual=sha(path);same=actual==row['sha256']and path.stat().st_size==row['bytes']
        archived.append(dict(key=key,path=str(path),expected_sha256=row['sha256'],actual_sha256=actual,bytes=path.stat().st_size));check('original_P0_input_bytes_exact',same,key)
    check('all25_P0_inputs_checked',len(archived)==25)
    index_after=sha(index_path);check('original_index_not_modified',index_before==index_after)
    status_after=output.with_name('ORIGINAL_STATUS_AFTER.bin'if untracked_mode=='all'else'ORIGINAL_STATUS_NORMAL_MODE_AFTER.bin')
    with status_after.open('xb')as f:f.write(actual_status)
    result=dict(schema='RISK_ORIGINAL_PRESERVATION_VERIFICATION_V1',status='PASS'if all(x['PASS']for x in checks)else'FAIL',
        checks=checks,failure_count=sum(not x['PASS']for x in checks),original_repo=str(repo),
        START_sha256=sha(start_path),START_original_status_sha256=sha(status_before),P0_private_manifest_sha256=sha(manifest_path),
        actual_status_sha256=actual_status_sha,actual_status_file=str(status_after),original_HEAD=head,local_main=main,
        protected_files=protected,original_input_artifacts=archived,read_only_git_commands=commands,
        original_index_before_sha256=index_before,original_index_after_sha256=index_after,
        source_sha256=sha(__file__),real_reference_content_decodes=0,synthetic_reference_content_decodes=0,
        worktree_mutations=0,index_refresh_disabled=True,reset_stash_checkout_add_commit_process_kill_operations=0,
        status_untracked_mode=untracked_mode,status_command_matches_START_view=untracked_mode=='normal',
        prior_failed_comparison=dict(path=str(Path(prior_failure).resolve()),sha256=sha(prior_failure),
            actual_status_sha256=json.loads(Path(prior_failure).read_text())['actual_status_sha256'])if prior_failure else None,
        enumeration_scope_note='Default normal mode collapses untracked directories; all-files enumeration is different. A byte mismatch across these views does not establish changed files.',
        scope='Exact original status bytes in explicitly stated view, HEAD/main, six protected file hashes and all25 P0 input artifact bytes; reference files are hashed only, never decoded.',
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    with output.open('x')as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--untracked-mode',choices=('all','normal'),default='all');p.add_argument('--prior-failure',type=Path);a=p.parse_args()
    r=run(a.repo_root,a.run_dir,a.output,untracked_mode=a.untracked_mode,prior_failure=a.prior_failure);print(json.dumps({k:r[k]for k in('status','failure_count','original_HEAD','local_main','actual_status_sha256','status_untracked_mode','cost')}));raise SystemExit(r['status']!='PASS')
