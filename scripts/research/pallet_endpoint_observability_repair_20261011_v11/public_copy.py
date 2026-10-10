"""Own-frozen public byte-copy fixture; no experiment or metric execution.

Copy the two new code/doc trees and their public immutable input closure.
Optionally reuse byte-identical files from the existing V9 review fixture.
This is a fresh destination of public files, not a fresh Git clone. No model,
RGB dataset, private GT, production import, solver or saved-metric arithmetic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

REPO=Path(__file__).resolve().parents[3]
V10='pallet_unused_line_endpoint_audit_20261011_v10';V11='pallet_endpoint_observability_repair_20261011_v11'
DOC=REPO/'_docs/experiments'/V11
PRIVATE=Path('/tmp/pallet-endpoint-observability-repair-private-20261011-v11')
ROOTS=tuple(str(Path(prefix)/name) for prefix in ('scripts/research','_docs/experiments') for name in (V10,V11))
FILES=('PUBLIC_COPY_PROTOCOL.json','PUBLIC_COPY_STARTED.json','PUBLIC_COPY_CHECKS.json','PUBLIC_FILE_MANIFEST.json')
MAX_NEW_BYTES=100*1024*1024
AUTHORITIES=(('_docs/experiments/'+V10,n) for n in ('AUDIT_PROTOCOL.json','VALIDATION_PROTOCOL.json','DIAGNOSTIC_SUMMARY_PROTOCOL.json'))
AUTHORITIES=tuple(str(Path(p)/n) for p,n in AUTHORITIES)+tuple(str(Path('_docs/experiments')/V11/n) for n in
    ('PROTOCOL.json','METRICS.json','STATISTICS_VERIFICATION_PROTOCOL.json','REVIEW_VALIDATION_PROTOCOL.json','FORWARD_RUNTIME_CHECK_PROTOCOL.json','runtime_forward_review/RUNTIME_PROTOCOL.json'))
POLICY=dict(source_GT_RGB_dataset_weights_model_production_import_fit_metric_calls=0,
    new_Git_clone=False,gzip_recompressed=False,no_existing_file_overwrite=True,
    original_V9_fixture_prior_provenance='192 published dependencies and 10 restored V9 original gzip files; actual selected reuse is recounted',
    max_new_Git_file_bytes=MAX_NEW_BYTES,inherited_restored_gzip_has_no_Git_size_guard=True,
    publication_own_late_outputs_excluded=list(FILES),automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def reject(p):require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink '+str(p))
def read(p):
    reject(p)
    with p.open(encoding='utf-8') as f:return json.load(f)
def binding(p,relative):
    reject(p);require(p.is_file(),'missing public file '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=relative,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))
def write(p,v):
    reject(p)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def allowed(relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:return False
    if str(p).startswith('scripts/research/'):return p.suffix=='.py'
    return str(p).startswith('_docs/experiments/') and (p.suffix in ('.md','.json','.gz','.csv','.png','.txt') or p.name=='.gitignore' or re.search(r'\.part\d{3}$',p.name) is not None)
def tree(root):
    reject(root);require(root.is_dir(),'missing public directory '+str(root))
    for p in sorted(root.rglob('*')):
        reject(p)
        if p.is_file() and '__pycache__' not in p.parts:yield p
def selection(args):
    excluded={str(Path('_docs/experiments')/V11/n) for n in FILES}|set(args.exclude)
    require(all(allowed(p) and any(p.startswith(r+'/') for r in ROOTS) for p in excluded),'late exclusion must belong to the two new trees')
    selected={};required={};omitted=[]
    for root in ROOTS:
        for p in tree(REPO/root):
            rel=str(p.relative_to(REPO))
            if rel in excluded:continue
            require(allowed(rel),'unexpected new public file format '+rel);require(p.stat().st_size<MAX_NEW_BYTES,'new Git file requires declared original-byte archive before copy: '+rel)
            selected[rel]=dict(binding=binding(p,rel),copy_from='current_repository',new_code_or_docs=True)
    for rel in AUTHORITIES:
        document=read(REPO/rel);require(isinstance(document.get('inputs'),dict),'public authority inputs missing '+rel)
        for value in document['inputs'].values():
            if not isinstance(value,dict) or not {'path','bytes','sha256'}<=set(value):continue
            if value.get('origin') not in (None,'public_repository'):continue
            path=value['path']
            if not allowed(path):
                require(value.get('origin')!='public_repository','invalid declared public input '+path);continue
            require(path not in excluded,'required immutable input cannot be excluded '+path)
            require((REPO/path).is_file(),'missing immutable public input; restore its original gzip from ARCHIVE_MANIFEST before freeze if needed: '+path)
            b=binding(REPO/path,path);require(same(b,value),'mismatched immutable public input; restore its original gzip from ARCHIVE_MANIFEST if needed: '+path)
            require(path not in required or same(required[path],b),'conflicting immutable input '+path);required[path]=b
    if args.old_fixture:
        for p in tree(args.old_fixture):
            rel=str(p.relative_to(args.old_fixture))
            if not allowed(rel):omitted.append(dict(path=rel,reason='outside public code/document whitelist'));continue
            old=binding(p,rel);current=REPO/rel
            now=binding(current,rel) if current.is_file() else None
            if now is None or not same(old,now):
                require(rel not in required,'stale prior fixture intersects required immutable closure '+rel)
                omitted.append(dict(path=rel,reason='missing current public file' if now is None else 'current bytes differ',prior=old,current=now));continue
            if rel not in selected:selected[rel]=dict(binding=now,copy_from='existing_V9_fixture',new_code_or_docs=False)
    for rel,b in required.items():selected.setdefault(rel,dict(binding=b,copy_from='current_repository',new_code_or_docs=False))
    archives={}
    for rel,item in list(selected.items()):
        if item['new_code_or_docs'] or item['binding']['bytes']<MAX_NEW_BYTES:continue
        p=REPO/rel;manifest=p.parent/'ARCHIVE_MANIFEST.json';require(p.name.endswith('.gz') and manifest.is_file(),'large inherited file requires original archive authority '+rel)
        m=read(manifest);matches=[r for r in m['files'] if r['original']['path']==p.name and same(r['original'],item['binding'])]
        require(len(matches)==1,'large inherited original archive binding differs '+rel);mr=str(manifest.relative_to(REPO));archives[rel]=binding(manifest,mr)
        selected.setdefault(mr,dict(binding=archives[rel],copy_from='current_repository',new_code_or_docs=False))
    require(set(required)<=set(selected),'required public closure missing')
    return dict(files={k:selected[k] for k in sorted(selected)},required_public_inputs=required,excluded_prior_fixture_files=omitted,
        declared_late_exclusions=sorted(excluded),large_inherited_original_archive_bindings=archives)
def guard(args,stage):
    for p in (args.output,args.target):reject(p)
    require(args.output.is_dir() and (args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE)),'existing V11 receipt directory required')
    require(args.target.resolve().is_relative_to(PRIVATE) and not args.target.exists(),'new private public-fixture destination must be absent')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)
def freeze(args):
    guard(args,'freeze');data=selection(args)
    write(args.output/FILES[0],dict(schema='public_byte_copy_fixture_protocol_v11',policy=POLICY,selection=data,target=str(args.target.absolute()),
        old_fixture=None if args.old_fixture is None else str(args.old_fixture.absolute()),frozen_before_copy=True))
def run(args):
    guard(args,'run');own=read(args.output/FILES[0]);data=selection(args);pb=binding(args.output/FILES[0],str((args.output/FILES[0]).relative_to(REPO)) if args.output.is_relative_to(REPO) else FILES[0])
    require(own['selection']==data and own['policy']==POLICY and own['target']==str(args.target.absolute()) and own['old_fixture']==(None if args.old_fixture is None else str(args.old_fixture.absolute())) and own['frozen_before_copy'] is True,'frozen source selection differs')
    write(args.output/FILES[1],dict(protocol=pb,policy=POLICY,public_copy_runs=1));created=[];error=None
    try:
        args.target.mkdir(parents=True,exist_ok=False)
        for rel,item in data['files'].items():
            source=(args.old_fixture if item['copy_from']=='existing_V9_fixture' else REPO)/rel;target=args.target/rel
            target.parent.mkdir(parents=True,exist_ok=True);reject(target)
            with source.open('rb') as src,target.open('xb') as dst:
                for b in iter(lambda:src.read(1048576),b''):dst.write(b)
                dst.flush();os.fsync(dst.fileno())
            require(binding(target,rel)==item['binding'],'copied bytes differ '+rel);created.append(rel)
        require(selection(args)==data,'public source bytes changed while copying')
        for rel,b in data['required_public_inputs'].items():require(binding(args.target/rel,rel)==b,'materialized immutable input differs '+rel)
        write(args.output/FILES[3],dict(schema='public_code_docs_and_materialized_fixture_manifest_v11',protocol=pb,policy=POLICY,selection=data,
            source_new_code_docs_files=[item['binding'] for item in data['files'].values() if item['new_code_or_docs']],fixture_files=[item['binding'] for item in data['files'].values()]))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None
    write(args.output/FILES[2],dict(schema='public_byte_copy_fixture_receipt_v11',complete=complete,passed=complete,protocol=pb,policy=POLICY,error=error,
        copied_files=len(created),selected_files=len(data['files']),existing_V9_fixture_files_reused=sum(v['copy_from']=='existing_V9_fixture' for v in data['files'].values()),
        excluded_prior_fixture_files=data['excluded_prior_fixture_files'],target=str(args.target.absolute()),preserved_partial_file_paths=[str(p.relative_to(args.target)) for p in tree(args.target)] if args.target.exists() else [],
        manifest=binding(args.output/FILES[3],FILES[3]) if complete else None,first_failure_preserved=True,saved_metrics_verified=False))
    if not complete:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--old-fixture',type=Path)
    p.add_argument('--target',type=Path,default=PRIVATE/'public-fresh-v11');p.add_argument('--output',type=Path,default=DOC);p.add_argument('--exclude',action='append',default=[],help='explicit late relative public path')
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
