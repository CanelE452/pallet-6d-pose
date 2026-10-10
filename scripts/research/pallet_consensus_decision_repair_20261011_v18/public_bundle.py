"""Freeze a public byte closure and materialize an initially absent V17/V18 bundle.

Only public scripts/_docs bindings and the two complete experiment trees are
followed. External declarations are retained as metadata without opening assets.
Large gzip originals are restored from manifest-ordered public parts, never
copied from an ignored source original. This creates a bundle, not a Git clone.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
OLD_CODE=CODE.parent/'pallet_hybrid_standard_controls_20261011_v17'
OLD_DOC=REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'
PRIVATE=Path('/tmp/pallet-consensus-decision-repair-private-20261011-v18')
FILES=('PUBLIC_BUNDLE_PROTOCOL.json','PUBLIC_BUNDLE_STARTED.json','PUBLIC_BUNDLE_CHECKS.json','PUBLIC_BUNDLE_MANIFEST.json')
LATE_DOC_NAMES=('PUBLIC_REVIEW.md','RESULT_KO.md','README.md','BUILD_LEDGER.md',
    'REPORT_BUILD_PROTOCOL.json','REPORT_BUILD_STARTED.json','REPORT_BUILD_CHECKS.json','PUBLIC_READER_CHECKS.json')
READER_PROOF_NAMES={
    'PUBLIC_FRESH_V17_STATISTICS':('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json'),
    'PUBLIC_FRESH_V18_STATISTICS':('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json'),
    'PUBLIC_FRESH_V17_RUNTIME':('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json'),
    'PUBLIC_FRESH_V18_RUNTIME':('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')}
LIMIT=50*1024*1024
PUBLIC_ORIGINS=('public_repository','experiment','immutable_baseline_dependency')
LIMITS=dict(private_RGB_GT_weights_model_asset_reads=0,new_metric_timing_pose_J_SVD_arithmetic_calls=0,
    dependency_classification_uses_file_existence=False,external_metadata_is_not_current_asset_verification=True,
    large_ignored_originals_directly_copied=False,Git_clone_claim=False,automatic_retry=False,
    full_V16_scalar_FAIL_not_converted_to_PASS=True)

def require(value,message):
    if not value:raise ValueError(message)
def safe(path):
    p=Path(path).absolute();require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink '+str(p));return p
def read(path):
    with safe(path).open(encoding='utf-8') as f:return json.load(f)
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def bind(path):
    p=safe(path);require(p.is_file(),'missing public file '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),origin='public_repository' if p.is_relative_to(REPO) else 'bundle_output')
def write(path,value):
    with safe(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def relative(name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'safe full public relative path required '+str(name));return p
def declarations(value,location='$'):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value):yield location,value
        for k,v in value.items():yield from declarations(v,location+'.'+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from declarations(v,location+'.'+str(i))
def classify(v,legacy_external):
    require(isinstance(v['path'],str) and type(v['bytes']) is int and v['bytes']>=0 and isinstance(v['sha256'],str) and re.fullmatch('[0-9a-f]{64}',v['sha256']),'typed byte declaration')
    if v.get('origin')=='external_readonly_dependency':return None,'explicit_external_readonly_dependency'
    p=Path(v['path']);full=not p.is_absolute() and '..' not in p.parts and bool(p.parts) and p.parts[0] in ('scripts','_docs')
    origin=v.get('origin')
    key=(v['path'],v['bytes'],v['sha256'])
    if key in legacy_external:
        require(origin in ('experiment','immutable_baseline_dependency'),'public/legacy binding conflicts with exact explicit external authority')
        return None,'legacy_origin_joined_to_explicit_external_runtime_alias_path_SHA_bytes'
    if origin=='public_repository':require(full,'explicit public origin has unsafe path');return relative(v['path']),'explicit_public_repository'
    if origin in PUBLIC_ORIGINS or origin is None:
        if full:return relative(v['path']),'legacy_full_public_relative_path_SHA_bytes'
        return None,'legacy_metadata_without_full_public_relative_path'
    raise ValueError('undeclared dependency origin '+str(origin))

def gate(folder,version):
    stats=('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')
    timing=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
    proofs={}
    for kind,names in (('statistics',stats),('runtime',timing)):
        own,start,check=[read(folder/n) for n in names];bindings=[bind(folder/n) for n in names]
        if kind=='statistics':
            prefix='independent_native_hybrid_standard_statistics' if version==17 else 'independent_point_consensus_decision_statistics'
            expected=dict(saved_scored_rows_streamed=980 if version==17 else 1470,moment_scalars_checked=648 if version==17 else 972,
                distributions_checked=108 if version==17 else 162,pair_groups_checked=27 if version==17 else 45,metric_CI_slots_checked=81 if version==17 else 135)
            require(own['schema']==prefix+'_protocol_v'+str(version) and check['schema']==prefix+'_checks_v'+str(version)
                and check['error'] is None and all(check['actual_counts'][k]==v for k,v in expected.items()),'actual independent statistics topology PASS')
            require(same(check['protocol'],bindings[0]) and same(start['protocol'],bindings[0]),'statistics protocol/STARTED authority')
            source=OLD_CODE/'verify.py' if version==17 else CODE/'verify.py'
        else:
            prefix='independent_hybrid_standard_runtime' if version==17 else 'independent_restored_point_consensus_runtime'
            protocol_schema=prefix+'_check_protocol_v17' if version==17 else prefix+'_protocol_v18'
            require(own['schema']==protocol_schema and check['schema']==prefix+'_checks_v'+str(version)
                and check['exception'] is None and check['rows']==150 and check['measured']==130 and check['summary_slots']==56,
                'actual independent full fresh150 timing PASS')
            require(same(check['own_protocol'],bindings[0]) and same(start['own_protocol'],bindings[0]),'runtime protocol/STARTED authority')
            source=OLD_CODE/'runtime_check.py' if version==17 else CODE/'runtime_checks.py'
            require(same(check['inputs']['runtime_protocol'],bind(folder/'RUNTIME_PROTOCOL.json')),'actual runtime producer protocol authority')
        require(check['complete'] is True and check['passed'] is True and check['failure_count']==0
            and check['inputs']==own['inputs']==start['inputs'] and same(check['checker'],bind(source)),'actual independent PASS/current code/full input closure')
        proofs[kind]=dict(protocol=bindings[0],started=bindings[1],checks=bindings[2],checker=bind(source))
    return proofs

def legacy_external_index():
    index={}
    for folder in (OLD_DOC,DOC):
        path=folder/'RUNTIME_PROTOCOL.json';producer=read(path);protocol_binding=bind(path)
        for alias,value in producer['inputs'].items():
            prefix=next((p for p in ('fresh_v7:source:','fresh_v7:baseline:') if alias.startswith(p)),None)
            if prefix is None or value.get('origin')!='external_readonly_dependency':continue
            name=alias[len(prefix):];p=Path(name)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in ('scripts','_docs'):continue
            for name in (name,value['path']):
                q=Path(name);require(not q.is_absolute() and '..' not in q.parts,'safe exact external alias metadata')
                index.setdefault((name,value['bytes'],value['sha256']),[]).append(dict(alias=alias,declaration=value,runtime_protocol=protocol_binding))
    return index

def snapshot(args):
    gates=dict(V17=gate(OLD_DOC,17),V18=gate(DOC,18));legacy_external=legacy_external_index();excluded={safe(args.receipts/n) for n in FILES}
    excluded.update(safe(DOC/n) for n in LATE_DOC_NAMES);excluded.add(safe(OLD_DOC/'PUBLIC_REVIEW_KO.md'))
    excluded.update(safe(DOC/folder/name) for folder,names in READER_PROOF_NAMES.items() for name in names)
    files={};queue=deque();scanned=set();skipped={};provenance={};edges=0
    def add(path,expected=None,reason='tree'):
        p=safe(path);name=str(p.relative_to(REPO));relative(name);require(p not in excluded,'excluded file is a required dependency '+name)
        if p.suffix=='.gz' and (expected['bytes'] if expected is not None else p.stat().st_size)>LIMIT:
            manifest=safe(p.parent/'ARCHIVE_MANIFEST.json');archive=read(manifest)
            matches=[v for v in archive['files'] if v['original']['path']==p.name]
            require(archive['original_files_preserved'] is True and len(matches)==1,'large gzip needs unique preserved archive manifest '+name)
            entry=matches[0];original=entry['original'];parts=entry['parts'];declared=entry.get('storage')
            require(original['bytes']>LIMIT and bool(parts) and declared in (None,'ordered_parts'),'large gzip must derive from ordered public parts '+name)
            if expected is not None:require(same(original,expected),'large declared original differs from public archive authority')
            part_specs=[];part_names=[]
            for part in parts:
                part_path=Path(part['path']);require(not part_path.is_absolute() and '..' not in part_path.parts and part_path.parts and part_path.parts[0]=='archives','public ordered archive-part path')
                q=safe(p.parent/part_path);require(q!=p and same(bind(q),part),'ordered public part differs');part_names.append(str(q.relative_to(REPO)));part_specs.append(dict(path=str(q.relative_to(REPO)),bytes=part['bytes'],sha256=part['sha256']))
            require(len(set(part_names))==len(part_names) and sum(v['bytes'] for v in parts)==original['bytes'],'complete unique ordered original bytes')
            value=dict(path=name,bytes=original['bytes'],sha256=original['sha256'],origin='public_repository')
            spec=dict(binding=value,source_kind='ordered_archive_parts',ordered_parts=part_specs)
            if name in files:require(files[name]==spec,'conflicting public original declaration '+name);return
            files[name]=spec;provenance[name]=dict(manifest=bind(manifest),manifest_format='legacy_parts_without_storage' if declared is None else 'explicit_storage',effective_storage='ordered_parts',original=value,parts=part_specs,source_original_opened=False)
            add(manifest,reason='archive_authority')
            for part in part_specs:add(REPO/part['path'],part,'ordered_part')
        else:
            value=bind(p)
            if expected is not None:require(same(value,expected),'current public dependency differs '+name)
            spec=dict(binding=value,source_kind='direct_public_file')
            if name in files:require(files[name]==spec,'conflicting public byte declaration '+name);return
            files[name]=spec
            if p.suffix=='.json':queue.append(p)
    for tree in (OLD_CODE,OLD_DOC,CODE,DOC):
        require(tree.is_dir(),'public experiment tree missing')
        for p in sorted(tree.rglob('*')):
            safe(p)
            if '__pycache__' in p.parts or p.suffix=='.pyc' or p in excluded:continue
            if p.is_file():
                allowed=p.suffix=='.py' if tree in (OLD_CODE,CODE) else p.suffix in ('.json','.md','.gz','.csv','.png','.txt') or p.name=='.gitignore' or bool(re.fullmatch(r'.+\.part\d{3}',p.name))
                require(allowed,'undeclared asset in experiment tree '+str(p));add(p)
    # Source-only scalar closures also require these exact public helper modules.
    for name in ('pallet_finite_edge_consensus_repair_20261011_v13/verify.py','pallet_three_head_observation_20261010_v7/statistics.py',
        'pallet_endpoint_observability_repair_20261011_v11/runtime_check.py','pallet_factor_subset_consensus_20261011_v15/runtime_check.py'):
        add(REPO/'scripts/research'/name,reason='standalone_public_scalar_source')
    while queue:
        p=queue.popleft();name=str(p.relative_to(REPO))
        if name in scanned:continue
        scanned.add(name)
        for location,value in declarations(read(p)):
            q,classification=classify(value,legacy_external);edges+=1
            if q is None:
                key=json.dumps(value,sort_keys=True,separators=(',',':'));skipped.setdefault(key,dict(declaration=value,classification=classification,referenced_by=[]))['referenced_by'].append(dict(file=name,location=location))
            else:add(REPO/q,value,'JSON_binding')
    return dict(files=dict(sorted(files.items())),archive_provenance=provenance,excluded_paths=sorted(str(p.relative_to(REPO)) for p in excluded),
        gate_proofs=gates,late_output_exclusions_are_exact_files_not_directories=True,protected_existing_V17_three_reports_remain_included=True,
        late_outputs_not_claimed_as_part_of_frozen_byte_bundle=True,
        legacy_external_alias_authority=[dict(path=k[0],bytes=k[1],sha256=k[2],sources=v) for k,v in sorted(legacy_external.items())],
        JSON_files_scanned=sorted(scanned),binding_edges_visited=edges,
        external_or_unresolved_legacy_metadata=list(skipped.values()),limits=LIMITS)

def guard(args,stage):
    target=safe(args.output);receipts=safe(args.receipts)
    require(target==PRIVATE/'public-fresh-v17-v18' and not target.exists() and target.parent.is_dir(),'fixed initially absent fresh bundle')
    require(receipts==DOC and receipts.is_dir(),'own existing V18 receipt folder')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(receipts/name).exists(),'preserve '+name)
    return target,receipts
def freeze(args):
    target,out=guard(args,'freeze');snap=snapshot(args)
    write(out/FILES[0],dict(schema='paired_standard_consensus_public_bundle_protocol_v18',snapshot=snap,target=str(target),
        code=bind(Path(__file__)),target_initially_absent=True,frozen_before_copy=True,limits=LIMITS))
    print('PUBLIC_V17_V18_BUNDLE_FROZEN',len(snap['files']),flush=True)
def run(args):
    target,out=guard(args,'run');own=read(out/FILES[0]);ob=bind(out/FILES[0]);snap=snapshot(args)
    require(own['schema']=='paired_standard_consensus_public_bundle_protocol_v18' and own['snapshot']==snap and own['limits']==LIMITS
        and own['target']==str(target) and own['target_initially_absent'] is True and same(own['code'],bind(Path(__file__))),'own frozen public closure differs')
    write(out/FILES[1],dict(own_protocol=ob,target=str(target),target_initially_absent=True,limits=LIMITS))
    began=time.perf_counter();copied=[];restored=[];error=None;manifest_binding=None
    try:
        target.mkdir(exist_ok=False)
        for name,spec in snap['files'].items():
            if spec['source_kind']=='ordered_archive_parts':continue
            q=safe(target/name);q.parent.mkdir(parents=True,exist_ok=True)
            with safe(REPO/name).open('rb') as source,q.open('xb') as dest:shutil.copyfileobj(source,dest,8*1024*1024);dest.flush();os.fsync(dest.fileno())
            require(same(bind(q),spec['binding']),'copied public bytes differ '+name);copied.append(name)
        for name,spec in snap['files'].items():
            if spec['source_kind']!='ordered_archive_parts':continue
            q=safe(target/name);q.parent.mkdir(parents=True,exist_ok=True)
            with q.open('xb') as dest:
                for part in spec['ordered_parts']:
                    p=safe(target/part['path']);require(same(bind(p),part),'copied ordered part differs')
                    with p.open('rb') as source:shutil.copyfileobj(source,dest,8*1024*1024)
                dest.flush();os.fsync(dest.fileno())
            require(same(bind(q),spec['binding']),'public-part restored original bytes differ '+name);restored.append(name)
        require(snapshot(args)==snap,'source public closure changed during copy')
        manifest=dict(schema='paired_standard_consensus_public_bundle_manifest_v18',target=str(target),files={k:v['binding'] for k,v in snap['files'].items()},
            original_public_bytes_preserved=True,restored_originals_from_copied_public_parts=restored,archive_provenance=snap['archive_provenance'],
            external_or_unresolved_legacy_metadata=snap['external_or_unresolved_legacy_metadata'],excluded_paths=snap['excluded_paths'],limits=LIMITS)
        write(out/FILES[3],manifest)
        manifest_binding=bind(out/FILES[3])
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    complete=error is None and len(copied)+len(restored)==len(snap['files'])
    write(out/FILES[2],dict(schema='paired_standard_consensus_public_bundle_checks_v18',complete=complete,passed=complete,
        own_protocol=ob,code=own['code'],target=str(target),target_initially_absent=True,gate_proofs=snap['gate_proofs'],copied_files=copied,
        restored_originals_from_copied_public_parts=restored,total_files=len(copied)+len(restored),total_bytes=sum(v['binding']['bytes'] for v in snap['files'].values()),
        manifest=manifest_binding,error=error,limits=LIMITS,elapsed_seconds=time.perf_counter()-began,
        source_files_mutated=False,partial_target_preserved_on_failure=True,Git_clone_claim=False))
    print('PUBLIC_V17_V18_BUNDLE',complete,len(copied),len(restored),flush=True)
    require(complete,'public bundle incomplete; preserve first failure and partial target')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--output',type=Path,default=PRIVATE/'public-fresh-v17-v18');p.add_argument('--receipts',type=Path,default=DOC)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
