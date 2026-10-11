"""Snapshot/verify original source bytes and all bound historical inputs.

Git is read with optional locks disabled. Existing private snapshots are never
replaced. Without an initial untracked-file hash manifest, full preservation of
untracked contents is explicitly not claimed. No private absolute path is emitted.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def sha_bytes(v):return hashlib.sha256(v).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def git(root,*args):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    return subprocess.check_output(['git','--no-optional-locks','-C',str(root),*args],env=env)
def manuscript_paths(root):
    return [Path(x.decode()) for x in git(root,'ls-files','-z').split(b'\0') if x and Path(x.decode()).suffix.lower() in ('.tex','.pdf','.bib','.sty','.cls')]

def bound_inputs(args):
    doc=Path(args.public_dir);private=Path(args.private_dir)
    roots={'historical_source':Path(args.source_root),'published_source':Path(args.published_root),'wd_published_source':Path(args.wd_published_root),'private_new':private,'new_worktree':Path(__file__).resolve().parents[3]}
    bindings=read(doc/'SOURCE_BINDINGS.json');checked=[];failures=[]
    for b in bindings['bindings']:
        owner=b['owner'];path=roots[owner]/b['path'];row=dict(owner=owner,path=b['path'],expected_sha256=b['sha256'],expected_bytes=b['bytes'])
        try:
            row.update(actual_sha256=sha(path),actual_bytes=path.stat().st_size,readable=True)
            row['unchanged']=row['actual_sha256']==b['sha256'] and row['actual_bytes']==b['bytes']
        except (OSError,KeyError):row.update(readable=False,unchanged=False)
        checked.append(row)
        if not row['unchanged']:failures.append(dict(owner=owner,path=b['path'],reason='unreadable_or_digest_size_changed'))
    additional=[]
    dependencies=doc/'REPRO_DEPENDENCIES.json'
    if dependencies.exists():
        known={r['path']:r for r in checked if r['owner']=='historical_source'}
        for b in read(dependencies)['expected_private_checkpoint_inputs']:
            if b['path'] in known:
                assert known[b['path']]['expected_sha256']==b['sha256'],('Conflicting checkpoint baseline',b['path'])
                continue
            path=Path(args.source_root)/b['path'];row=dict(owner='historical_source',path=b['path'],expected_sha256=b['sha256'],baseline_manifest='REPRO_DEPENDENCIES.json')
            try:row.update(actual_sha256=sha(path),actual_bytes=path.stat().st_size,readable=True);row['unchanged']=row['actual_sha256']==b['sha256']
            except OSError:row.update(readable=False,unchanged=False)
            additional.append(row)
            if not row['unchanged']:failures.append(dict(owner='historical_source',path=b['path'],reason='additional_frozen_checkpoint_unreadable_or_digest_changed'))
    return checked,additional,failures

def snapshot(args):
    start=time.monotonic();root=Path(args.source_root);private=Path(args.private_dir);doc=Path(args.public_dir)
    names=['original-status.bin','original-diff.bin','original-preservation-metadata.json','original-manuscript-hashes.json']
    assert not any((private/n).exists() for n in names),'Existing original snapshots are preserved; use verify'
    status=git(root,'status','--porcelain=v1','-z','--untracked-files=normal');diff=git(root,'diff','--binary','HEAD')
    paths=manuscript_paths(root);manuscripts=[dict(path=str(p),sha256=sha(root/p),bytes=(root/p).stat().st_size) for p in paths if (root/p).is_file()]
    private.mkdir(parents=True,exist_ok=True)
    for name,content in [('original-status.bin',status),('original-diff.bin',diff)]:
        with (private/name).open('xb') as f:f.write(content)
    write(private/'original-manuscript-hashes.json',dict(stage='PRE_EXPERIMENT_SNAPSHOT',tracked_files=manuscripts,untracked_files_hashed=False))
    write(private/'original-preservation-metadata.json',dict(HEAD=git(root,'rev-parse','HEAD').decode().strip(),status_command='git status --porcelain=v1 -z --untracked-files=normal',diff_command='git diff --binary HEAD',status_sha256=sha_bytes(status),diff_sha256=sha_bytes(diff)))
    write(doc/'SOURCE_SNAPSHOT_RECEIPT.json',dict(status='SNAPSHOT_COMPLETE',source_status_sha256=sha_bytes(status),source_diff_sha256=sha_bytes(diff),source_status_entries=sum(bool(r) for r in status.split(b'\0')),tracked_manuscript_snapshot_files=len(manuscripts),untracked_content_snapshot=False,private_absolute_paths_published=False,source_mutations=0,model_inferences=0,optimizer_updates=0,solver_calls=0,wall_seconds=time.monotonic()-start))

def verify(args):
    start=time.monotonic();root=Path(args.source_root);private=Path(args.private_dir);doc=Path(args.public_dir)
    before_status=(private/'original-status.bin').read_bytes();before_diff=(private/'original-diff.bin').read_bytes()
    status=git(root,'status','--porcelain=v1','-z','--untracked-files=normal');diff=git(root,'diff','--binary','HEAD')
    checked,additional,failures=bound_inputs(args);status_equal=status==before_status;diff_equal=diff==before_diff
    if not status_equal:failures.append(dict(owner='historical_source',path='git-status',reason='status_bytes_differ_from_initial_snapshot'))
    if not diff_equal:failures.append(dict(owner='historical_source',path='git-diff',reason='binary_HEAD_diff_bytes_differ_from_initial_snapshot'))
    paths=manuscript_paths(root);inventory=[];initial=private/'original-manuscript-hashes.json'
    before_manuscripts={r['path']:r for r in read(initial)['tracked_files']} if initial.exists() else None
    for path in paths:
        p=root/path
        if not p.is_file():continue
        row=dict(path=str(path),current_sha256=sha(p),current_bytes=p.stat().st_size)
        row['git_diff_from_HEAD_present']=bool(git(root,'diff','--name-only','HEAD','--',str(path)).strip())
        if before_manuscripts is not None:
            old=before_manuscripts.get(str(path));row['initial_full_byte_hash_available']=old is not None
            row['initial_full_byte_hash_matches']=old is not None and old['sha256']==row['current_sha256'] and old['bytes']==row['current_bytes']
            if not row['initial_full_byte_hash_matches']:failures.append(dict(owner='historical_source',path=str(path),reason='tracked_manuscript_initial_full_byte_hash_mismatch'))
        else:row['initial_full_byte_hash_available']=False
        inventory.append(row)
    checkpoints=[r for r in checked if r['path'].endswith('.pt')];all_checkpoints=checkpoints+additional
    result=dict(status='PASS' if not failures else 'FAIL',source_HEAD=git(root,'rev-parse','HEAD').decode().strip(),status_command='git status --porcelain=v1 -z --untracked-files=normal (GIT_OPTIONAL_LOCKS=0)',diff_command='git diff --binary HEAD (staged and unstaged tracked changes)',source_status_bytes_initial=len(before_status),source_status_bytes_final=len(status),source_status_entries_initial=sum(bool(x) for x in before_status.split(b'\0')),source_status_entries_final=sum(bool(x) for x in status.split(b'\0')),source_status_sha256_initial=sha_bytes(before_status),source_status_sha256_final=sha_bytes(status),source_status_bytes_identical=status_equal,source_binary_diff_bytes_initial=len(before_diff),source_binary_diff_bytes_final=len(diff),source_binary_diff_sha256_initial=sha_bytes(before_diff),source_binary_diff_sha256_final=sha_bytes(diff),source_binary_diff_bytes_identical=diff_equal,all_SOURCE_BINDINGS_readable_and_digest_unchanged=all(r['unchanged'] for r in checked),source_bindings_manifest_sha256=sha(doc/'SOURCE_BINDINGS.json'),bindings_checked=len(checked),bound_checkpoint_count=len(checkpoints),all_bound_original_checkpoints_unchanged=all(r['unchanged'] for r in checkpoints),source_binding_results=checked,tracked_manuscript_inventory=inventory,tracked_manuscript_git_state_preserved=status_equal and diff_equal,initial_tracked_manuscript_full_byte_snapshot_available=before_manuscripts is not None,initial_untracked_manuscript_content_hashes_available=False,full_untracked_content_equivalence_claimed=False,manuscript_limit='Existing start snapshots cover Git status/diff, not untracked file bytes. Current manuscript digests are an inventory, not invented initial hashes; no complete untracked manuscript byte-preservation claim.',failures=failures,private_absolute_paths_published=False,raw_RGB_or_reference_coordinates_published=False,original_status_and_diff_snapshots_overwritten=False,source_mutations=0,source_checkpoint_writes=0,manuscript_writes=0,model_inferences=0,optimizer_updates=0,solver_calls=0,wall_seconds=time.monotonic()-start)
    result.update(additional_repro_checkpoint_results=additional,all_expected_original_checkpoint_count=len(all_checkpoints),all_expected_original_checkpoints_unchanged=all(r['unchanged'] for r in all_checkpoints))
    write(doc/args.receipt_name,result);assert not failures,'SOURCE_PRESERVATION_FAILED: see public receipt'
    print('SOURCE_PRESERVATION_PASS',len(checked),'bindings',len(all_checkpoints),'checkpoints',len(inventory),'tracked manuscripts',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['snapshot','verify']);p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);p.add_argument('--published-root',required=True);p.add_argument('--wd-published-root',required=True);p.add_argument('--receipt-name',default='SOURCE_PRESERVATION.json');a=p.parse_args()
    assert Path(a.receipt_name).name==a.receipt_name and a.receipt_name.endswith('.json');(snapshot if a.stage=='snapshot' else verify)(a)
