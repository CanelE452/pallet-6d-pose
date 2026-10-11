"""Narrow preservation of existing inputs, original dirty changes and manuscripts."""
import argparse, hashlib, importlib.util, json, os, subprocess, time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def git(root,*args):return subprocess.check_output(['git','--no-optional-locks','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})

def run(args):
    root=Path(__file__).resolve().parents[3];old=root/'_docs/experiments/pallet_nonsevere_precision_20261011'
    doc=root/'_docs/experiments/pallet_rgb_only_cvrank_20261011';private=Path(args.private_dir)
    source=Path(args.source_root);prior=Path(args.prior_private_dir);start=time.monotonic()
    if args.stage=='snapshot':
        assert (private/'original-status.bin').exists() and (private/'original-diff.bin').exists()
        inventory=[]
        for directory in (root/'scripts/research/pallet_nonsevere_precision_20261011',old):
            for p in sorted(directory.rglob('*')):
                if p.is_file():inventory.append(dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size))
        snapshot=private/'prior_namespace_snapshot.json'
        if snapshot.exists():assert read(snapshot)==dict(files=inventory)
        else:write(snapshot,dict(files=inventory))
        prior_manuscripts=read(old/'SOURCE_PRESERVATION.json')['tracked_manuscript_inventory']
        assert all(sha(source/r['path'])==r['current_sha256'] for r in prior_manuscripts)
        write(doc/'PRESERVATION_START.json',dict(status='PASS',prior_namespace_files=len(inventory),
            prior_namespace_snapshot_sha256=sha(private/'prior_namespace_snapshot.json'),
            existing_original_dirty_status_diff_snapshots_preserved=True,
            tracked_manuscripts_prior_baseline_reverified=len(prior_manuscripts),
            untracked_full_content_hash_claim=False,source_mutations=0,wall_seconds=time.monotonic()-start))
    else:
        spec=importlib.util.spec_from_file_location('prior_preservation',root/'scripts/research/pallet_nonsevere_precision_20261011/preservation.py')
        p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
        adapted=argparse.Namespace(source_root=args.source_root,private_dir=args.prior_private_dir,public_dir=str(old),
            published_root=args.published_root,wd_published_root=args.wd_published_root)
        bound,extra,failures=p.bound_inputs(adapted)
        status=git(source,'status','--porcelain=v1','-z','--untracked-files=normal')
        diff=git(source,'diff','--binary','HEAD')
        assert status==(private/'original-status.bin').read_bytes() and diff==(private/'original-diff.bin').read_bytes()
        original=read(old/'SOURCE_PRESERVATION.json')['tracked_manuscript_inventory']
        assert all(sha(source/r['path'])==r['current_sha256'] and (source/r['path']).stat().st_size==r['current_bytes'] for r in original)
        previous=read(private/'prior_namespace_snapshot.json')['files']
        assert all(sha(root/r['path'])==r['sha256'] and (root/r['path']).stat().st_size==r['bytes'] for r in previous)
        assert not failures
        write(doc/'SOURCE_PRESERVATION.json',dict(status='PASS',source_HEAD=git(source,'rev-parse','HEAD').decode().strip(),
            source_status_entries=sum(bool(x) for x in status.split(b'\0')),
            source_status_bytes_exact=True,source_diff_bytes_exact=True,
            source_status_sha256=hashlib.sha256(status).hexdigest(),source_diff_sha256=hashlib.sha256(diff).hexdigest(),
            immutable_input_bindings=len(bound),original_checkpoints=sum(r['path'].endswith('.pt') for r in bound)+len(extra),
            all_input_bindings_and_checkpoints_exact=True,tracked_manuscript_files=len(original),tracked_manuscript_hashes_exact=True,
            existing_precision_namespace_files=len(previous),existing_precision_namespace_hashes_exact=True,
            source_writes=0,manuscript_writes=0,old_result_writes=0,untracked_content_equivalence_claimed=False,
            bound_source_manifest_sha256=sha(old/'SOURCE_BINDINGS.json'),wall_seconds=time.monotonic()-start))
    print('PRESERVATION',args.stage,'PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['snapshot','verify'])
    for name in ('source-root','private-dir','prior-private-dir','published-root','wd-published-root'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
