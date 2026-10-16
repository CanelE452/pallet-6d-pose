"""Copy the complete V17/V18 trees and four actual public reader input closures.

Historical JSON receipts are preserved as bytes. Their nested provenance and
protection inventories are not reinterpreted as current asset dependencies.
"""
import ast
import copy
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
SOURCE=CODE/'public_bundle.py';SOURCE_SHA='e0fce1a64f5662e5f1e3f9269b2d22f5ae7a4b2feeee9b3ab2a3ff54d4dbca9f'
READY=CODE/'public_bundle_ready.py';READY_SHA='4a468a6529a6fefb18453535723d11af30a58a7d63f16618adad2c2cb992321c'
V2=REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
FIRST_SHA='49c28dce160be2a7e3963f9adacf99c6666c17e24bfdd5e07ede0db36c1888f4'
FINAL_SHA='d13e65d4de20627903cba528badadf4c4a9a5375fbf18cc43986340f2025e99f'
SPECS={
    'code':dict(relative_path='scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',old_bytes=30395,old_sha256='22ad2227458b437e360ea3b7bf26d1a524beff4b1b2a68db17eef53af8145434',current_bytes=31343,current_sha256='262953a907a8805326c8537a81723933862ffc0e58b499d54296de1669b0b095'),
    'checks_code':dict(relative_path='scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline_checks.py',old_bytes=8605,old_sha256='17fba2f3f5458cbcef632b9dc26c2d82eb66d795503e7a922f54fbb0db669adc',current_bytes=8735,current_sha256='506ab9da6e56c95ad23ce18af526f376036eaaa809535353bbdb0cfb93f9ce7b')}

def require(value,message):
    if not value:raise ValueError(message)
def canonical(node):return ast.dump(node,include_attributes=False)
def namespace():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable original public bundle source')
    original=ast.parse(SOURCE.read_text());before=canonical(original)
    snapshot=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
    loops=[n for n in snapshot.body if isinstance(n,ast.While) and canonical(n.test)==canonical(ast.Name(id='queue',ctx=ast.Load()))]
    require(len(loops)==1,'exact original recursive JSON walk')
    walk=loops[0];replacement=ast.parse('edges = attach_required_inputs(legacy_external, add)').body[0]
    revised=copy.deepcopy(original);new_snapshot=next(n for n in revised.body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
    index=next(i for i,n in enumerate(new_snapshot.body) if canonical(n)==canonical(walk));new_snapshot.body[index]=copy.deepcopy(replacement)
    restored=copy.deepcopy(revised);restore_snapshot=next(n for n in restored.body if isinstance(n,ast.FunctionDef) and n.name=='snapshot');restore_snapshot.body[index]=copy.deepcopy(walk)
    require(canonical(restored)==before,'whole original canonical source restored after one snapshot walk replacement')
    old_add=next(n for n in snapshot.body if isinstance(n,ast.FunctionDef) and n.name=='add')
    new_add=next(n for n in new_snapshot.body if isinstance(n,ast.FunctionDef) and n.name=='add')
    require(canonical(old_add)==canonical(new_add),'original archive/direct-file add body unchanged')
    unchanged=[n.name for n in original.body if isinstance(n,ast.FunctionDef) and n.name!='snapshot']
    for name in unchanged:
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in revised.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require(canonical(a)==canonical(b),'original non-snapshot function unchanged '+name)
    h=dict(__file__=str(Path(__file__).resolve()),__name__='isolated_actual_reader_public_bundle')
    # Load the complete immutable original without changing its AST. Replace only
    # snapshot with the proved one-block adaptation in the same namespace.
    exec(compile(original,str(SOURCE),'exec'),h)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new_snapshot],type_ignores=[])),str(SOURCE)+' [reader closure snapshot]','exec'),h)
    return h,dict(original_source_sha256=SOURCE_SHA,original_full_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        snapshot_recursive_walk_AST_replacements=1,whole_original_canonical_restore_proven=True,
        original_add_function_AST_sha256=hashlib.sha256(canonical(old_add).encode()).hexdigest(),
        unchanged_original_non_snapshot_functions=unchanged,original_source_changed=False,
        copy_archive_gate_freeze_run_numeric_or_origin_classification_policy_changes=0)

def history(h,witness):
    read,bind,same=h['read'],h['bind'],h['same'];first=V2/'PIPELINE_CHECKS.json';final=V2/'PIPELINE_CHECKS_FINAL.json'
    fb,lb=bind(first),bind(final);require(fb['bytes']==lb['bytes']==2155 and fb['sha256']==FIRST_SHA and lb['sha256']==FINAL_SHA,'exact original/final CPU receipt bytes')
    old,new=read(first),read(final)
    require(old['schema']==new['schema']=='validated_boundary_pipeline_cpu_contract_checks_v2' and old['passed'] is True and new['passed'] is True and old['check_count']==new['check_count']==11,'preserved first/final synthetic CPU11 receipts')
    current={};declarations=[]
    for field,spec in SPECS.items():
        actual=bind(REPO/spec['relative_path'])
        require(old[field]==dict(path=spec['relative_path'],origin='public_repository',bytes=spec['old_bytes'],sha256=spec['old_sha256']),'exact historical declaration '+field)
        require(new[field]==dict(path=spec['relative_path'],origin='public_repository',bytes=spec['current_bytes'],sha256=spec['current_sha256']) and same(actual,new[field]),'exact current/final source '+field)
        current[field]=actual;declarations.append(dict(JSON_location='$.'+field,historical_declared_relative_path=spec['relative_path'],historical_declared_size_bytes=spec['old_bytes'],historical_declared_sha256=spec['old_sha256'],current_source_binding=actual))
    failures={}
    for name,schema,source,sha in (('FIRST_PUBLIC_BUNDLE_FREEZE_FAILURE.json','root_first_public_bundle_freeze_failure_v18',SOURCE,SOURCE_SHA),('FIRST_READY_PUBLIC_BUNDLE_FREEZE_FAILURE.json','root_first_ready_public_bundle_freeze_failure_v18',READY,READY_SHA)):
        p=DOC/name;value=read(p);source_binding=bind(source)
        require(source_binding['sha256']==sha and same(value['source'],source_binding) and value['schema']==schema and value['stage']=='freeze' and value['exit_code']==1
            and value['own_protocol_created'] is False and value['copy_target_created'] is False and value['copy_calls']==value['new_numeric_model_GT_fit_calls']==0 and value['failure_preserved'] is True,'exact preserved pre-protocol/copy failure '+name)
        failures[name]=bind(p)
    return dict(source_adapter=witness,original_bundle_source=bind(SOURCE),failed_ready_source=bind(READY),current_source=bind(Path(__file__).resolve()),
        preserved_bundle_freeze_failures=failures,historical_first_CPU_receipt=fb,final_CPU_receipt=lb,current_sources=current,historical_declarations=declarations,
        historical_source_replay_performed=False,historical_CPU11_is_not_proof_for_current_source=True,
        checked_reachable_Git_path_history_commit='1c9a7168aff9ceed28b0269cd0000c42a22c10b6',
        old_source_unavailable_in_checked_reachable_Git_path_history=True,Git_history_lookup_not_reexecuted_by_bundle_helper=True)

def reader_inputs(h):
    external=h['legacy_external_index']();checked={};public={};readers={};skipped=[];edges=0
    for family,folder in (('V17',h['OLD_DOC']),('V18',h['DOC'])):
        for stage,name in (('statistics','STATISTICS_VERIFICATION_PROTOCOL.json'),('runtime','RUNTIME_CHECK_PROTOCOL.json')):
            p=folder/name;own=h['read'](p);inputs={};omitted=[]
            for alias,value in own['inputs'].items():
                q,classification=h['classify'](value,external);edges+=1
                if q is None:
                    require(value.get('origin')=='external_readonly_dependency' or (value['path'],value['bytes'],value['sha256']) in external,'actual reader input requires explicit public or exact external authority')
                    item=dict(reader=family+'.'+stage,alias=alias,declaration=value,classification=classification,asset_opened=False);omitted.append(item);skipped.append(item);continue
                key=str(q)
                if key not in checked:checked[key]=h['bind'](REPO/q)
                require(h['same'](checked[key],value),'current actual reader input differs '+family+'.'+stage+'.'+alias)
                public[key]=value;inputs[alias]=checked[key]
            readers[family+'.'+stage]=dict(protocol=h['bind'](p),current_public_inputs=inputs,explicit_external_metadata=omitted,
                all_current_public_inputs_byte_checked_before_snapshot=True,historical_source_exceptions_used_for_live_reader_inputs=False)
    require(len(readers)==4,'exact four actual current public readers')
    return external,public,readers,skipped,edges

def main():
    h,witness=namespace();proof=history(h,witness);original_snapshot=h['snapshot'];context={}
    def attach_required_inputs(external,add):
        require(external==context['external'],'same original exact external alias authority')
        for name,value in sorted(context['public'].items()):add(REPO/name,value,'actual_current_reader_input')
        for p in (V2/'PIPELINE_CHECKS.json',V2/'PIPELINE_CHECKS_FINAL.json',SOURCE,READY,*(REPO/v['relative_path'] for v in SPECS.values()),*(DOC/name for name in proof['preserved_bundle_freeze_failures'])):
            add(p,reason='explicit_historical_CPU_and_preserved_failure_bytes')
        return context['edges']
    h['attach_required_inputs']=attach_required_inputs
    def snapshot(args):
        require(history(h,witness)==proof,'historical/source/failure bytes changed')
        external,public,readers,skipped,edges=reader_inputs(h)
        context.clear();context.update(external=external,public=public,edges=edges)
        value=original_snapshot(args)
        require(value['JSON_files_scanned']==[],'historical JSON declarations must not be recursively parsed')
        value.update(actual_four_reader_current_input_closure=readers,historical_CPU_source_scope=proof,
            external_or_unresolved_legacy_metadata=skipped,
            dependency_scope='complete_V17_V18_code_doc_trees_plus_four_actual_reader_public_inputs_and_explicit_CPU_history',
            historical_JSON_receipts_copied_byte_exact=True,historical_nested_bindings_not_reinterpreted_as_live_dependencies=True,
            protection_inventories_preserved_as_metadata_not_current_asset_verification=True,
            current_required_public_original_bytes_opened_for_hash_verification=True,
            archive_provenance_source_original_opened_field_scope='archive_add_and_copy_only_not_required_input_hash_verification',
            malformed_historical_JSON_is_preserved_as_bytes_not_parsed=True)
        value['limits']=dict(value['limits'],recursive_historical_JSON_dependency_walks=0,current_reader_input_mismatch_exceptions=0,
            historical_first_CPU_source_replayed=False,historical_CPU11_is_not_current_code_proof=True,
            preserved_nested_provenance_assets_claimed_current=False,current_reader_public_input_closure_count=4)
        return value
    h['snapshot']=snapshot;h['main']()

if __name__=='__main__':main()
