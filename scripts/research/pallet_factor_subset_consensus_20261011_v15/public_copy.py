"""Freeze then copy exact public V15 dependencies into an initially absent bundle.

Only public protocol paths and the V15 code/document trees are copied. No Git,
private GT/RGB/weights, solver, metric calculation or original file mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
PRIVATE=Path('/tmp/pallet-factor-subset-consensus-private-20261011-v15')
FILES=('PUBLIC_COPY_PROTOCOL.json','PUBLIC_COPY_STARTED.json','PUBLIC_COPY_CHECKS.json','PUBLIC_COPY_MANIFEST.json')

def require(value,message):
    if not value:raise ValueError(message)

def safe(path):
    path=Path(path).absolute();require(not any(p.is_symlink() for p in (path,*path.parents)),'symlink '+str(path));return path

def read(path):
    with safe(path).open(encoding='utf-8') as f:return json.load(f)

def bind(path):
    p=safe(path);require(p.is_file(),'missing '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))

def write(path,value):
    with safe(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def public(name):
    p=Path(name)
    if p.is_absolute():require(p.is_relative_to(REPO),'external asset refused');p=p.relative_to(REPO)
    require('..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'unsafe public path');return safe(REPO/p)

def snapshot(args):
    proto=read(args.public_protocol);check=read(args.public_protocol.with_name('PUBLIC_REPORT_CHECKS.json'))
    require(proto['schema']=='factor_subset_report_public_protocol_v15' and check['schema']=='factor_subset_report_public_checks_v15' and check['complete'] is True and check['passed'] is True
        and check['inputs']==proto['inputs'] and same(check['protocol'],bind(args.public_protocol)),'actual public PASS required')
    require(check['table_rows']['moment_scalars']==810 and check['table_rows']['CI_slots']==81
        and check['timing_reported_statistic_slots']==56 and check['existing_statistics_rows']==1225
        and check['limits']['new_raw_metric_or_CI_arithmetic_calls']==0
        and check['limits']['external_declared_metadata_is_not_current_asset_verification'] is True
        and proto['scope']['classification_uses_file_existence_heuristic'] is False,'actual report transcription and public-origin authority')
    review=proto['scope']['public_origin_review']
    require(check['scope']['public_origin_review']==review and review['transcription_arithmetic_AST_changes']==0
        and review['legacy_classification_requires_origin_path_SHA_and_bytes'] is True
        and review['missing_file_heuristic_added'] is False and review['original_failed_before_protocol_and_STARTED'] is True
        and review['original_checker_or_fixed_reports_changed'] is False,'actual explicit legacy-origin review authority')
    for key,name in (('original_checker','public_review.py'),('review_checker','public_review_ready.py')):
        require(same(review[key],bind(CODE/name)),'public classifier source differs '+key)
    require(same(review['original_failure'],bind(DOC/'FIRST_PUBLIC_METADATA_FAILURE.json')),'preserved public metadata failure differs')
    excluded={safe(args.receipts/n) for n in FILES}|{safe(args.late_report)}
    require(safe(args.late_report).parent==DOC and safe(args.late_report).suffix=='.md','late report must be a new V15 document')
    files={}
    for alias,value in proto['inputs'].items():
        p=public(value['path']);require(p not in excluded,'excluded required public dependency '+alias);require(same(bind(p),value),'public frozen dependency differs '+alias);files[str(p.relative_to(REPO))]=p
    for tree in (CODE,DOC):
        for p in sorted(tree.rglob('*')):
            safe(p)
            if '__pycache__' in p.parts or p.suffix=='.pyc':continue
            if p.is_file() and p not in excluded:
                allowed=p.suffix=='.py' if tree==CODE else (p.suffix in ('.json','.md','.gz','.csv','.png','.txt')
                    or p.name=='.gitignore' or bool(re.fullmatch(r'.+\.part\d{3}',p.name)))
                require(allowed,'undeclared asset in V15 tree '+str(p.relative_to(REPO)))
                files[str(p.relative_to(REPO))]=p
    for p in excluded:
        if p.is_relative_to(REPO):files.pop(str(p.relative_to(REPO)),None)
    provenance={}
    for name,p in list(files.items()):
        if p.suffix!='.gz' or p.stat().st_size<50*1024*1024:continue
        manifest=p.parent/'ARCHIVE_MANIFEST.json';data=read(manifest)
        matches=[x for x in data['files'] if x['original']['path']==p.name]
        require(data['original_files_preserved'] is True and len(matches)==1 and same(bind(p),matches[0]['original']),
            'large inherited gzip needs exact archive provenance '+name)
        files[str(manifest.relative_to(REPO))]=manifest
        for part in matches[0]['parts']:
            relative=Path(part['path']);require(not relative.is_absolute() and '..' not in relative.parts,'unsafe archive part')
            q=safe(p.parent/relative);require(same(bind(q),part),'archive part differs');files[str(q.relative_to(REPO))]=q
        entry=matches[0];parts=entry['parts'];declared=entry.get('storage')
        if declared is None:
            require(bool(parts),'legacy archive must retain ordered parts')
            storage='direct_original' if len(parts)==1 and parts[0]['path']==p.name and same(parts[0],entry['original']) else 'ordered_parts'
        else:
            require(declared in ('direct_original','ordered_parts'),'unknown archive storage');storage=declared
        provenance[name]=dict(manifest=bind(manifest),manifest_format='explicit_storage' if declared is not None else 'legacy_parts_without_storage',
            effective_storage=storage,declared_storage=declared,parts=parts)
    files=dict(sorted(files.items()));bound={k:bind(p) for k,p in files.items()}
    return files,bound,provenance,sorted(str(p.relative_to(REPO)) for p in excluded if p.is_relative_to(REPO)),len(proto['inputs'])

def guard(args,stage):
    target=safe(args.output);receipts=safe(args.receipts)
    require(target==PRIVATE/'public-fresh-v15' and not target.exists(),'fresh fixed bundle must be initially absent')
    require(target.parent.is_dir() and receipts==DOC,'existing scratch parent and new V15 receipts required')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not (receipts/n).exists(),'preserve '+n)
    return target,receipts

def freeze(args):
    target,receipts=guard(args,'freeze');_,b,provenance,excluded,count=snapshot(args)
    write(receipts/FILES[0],dict(schema='fresh_public_dependency_copy_protocol_v15',inputs=b,
        public_protocol=bind(args.public_protocol),target=str(target),target_initially_absent=True,
        excluded_late_paths=excluded,archive_provenance=provenance,declared_public_input_entries=count,
        sorted_merge_of_public_protocol_and_full_V15_trees=True,new_private_asset_or_numeric_calls=0,automatic_retry=False))
    print('PUBLIC_COPY_FROZEN',len(b),flush=True)

def run(args):
    target,receipts=guard(args,'run');own=read(receipts/FILES[0]);files,b,provenance,excluded,count=snapshot(args)
    require(own['schema']=='fresh_public_dependency_copy_protocol_v15' and own['inputs']==b and own['archive_provenance']==provenance
        and own['excluded_late_paths']==excluded and own['target']==str(target) and own['target_initially_absent'] is True
        and same(own['public_protocol'],bind(args.public_protocol)),'own frozen copy differs')
    write(receipts/FILES[1],dict(own_protocol=bind(receipts/FILES[0]),inputs=b,target_initially_absent=True,new_private_asset_or_numeric_calls=0))
    began=time.perf_counter();result=dict(schema='fresh_public_dependency_copy_checks_v15',complete=False,passed=False,
        own_protocol=bind(receipts/FILES[0]),inputs=b,target=str(target),target_initially_absent=True,
        copied=[],new_private_asset_or_numeric_calls=0,source_files_mutated=False,automatic_retry=False)
    try:
        target.mkdir(exist_ok=False)
        for name,p in files.items():
            q=target/name;q.parent.mkdir(parents=True,exist_ok=True)
            with p.open('rb') as source,q.open('xb') as out:shutil.copyfileobj(source,out,8*1024*1024);out.flush();os.fsync(out.fileno())
            require(same(bind(q),b[name]),'copied bytes differ '+name);result['copied'].append(name)
        require(snapshot(args)[1:]==(b,provenance,excluded,count),'source closure changed during copy')
        manifest=dict(schema='fresh_public_dependency_copy_manifest_v15',files=b,initially_absent=True,
            target=str(target),excluded_late_paths=excluded,archive_provenance=provenance,
            final_mutable_public_report_excluded=True,new_private_asset_or_numeric_calls=0)
        write(receipts/FILES[3],manifest)
        result.update(complete=True,passed=True,manifest=bind(receipts/FILES[3]),copied_files=len(b),
            copied_bytes=sum(x['bytes'] for x in b.values()),declared_public_input_entries=count)
    except Exception as error:result['error']=dict(type=type(error).__name__,message=str(error))
    result['elapsed_seconds']=time.perf_counter()-began;write(receipts/FILES[2],result)
    print('PUBLIC_COPY',result['passed'],len(result['copied']),flush=True)
    if not result['passed']:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--public-protocol',type=Path,default=DOC/'PUBLIC_REPORT_PROTOCOL.json')
    p.add_argument('--output',type=Path,default=PRIVATE/'public-fresh-v15');p.add_argument('--receipts',type=Path,default=DOC)
    p.add_argument('--late-report',type=Path,default=DOC/'PUBLIC_REVIEW_KO.md');args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
