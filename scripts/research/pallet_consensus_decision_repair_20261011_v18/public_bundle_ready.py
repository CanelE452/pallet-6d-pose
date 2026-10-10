"""Keep two exact historical CPU source declarations separate from live inputs."""
import ast
import copy
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
SOURCE=CODE/'public_bundle.py';SOURCE_SHA='e0fce1a64f5662e5f1e3f9269b2d22f5ae7a4b2feeee9b3ab2a3ff54d4dbca9f'
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
V2=REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
FIRST_SHA='49c28dce160be2a7e3963f9adacf99c6666c17e24bfdd5e07ede0db36c1888f4'
FINAL_SHA='d13e65d4de20627903cba528badadf4c4a9a5375fbf18cc43986340f2025e99f'
SPECS={
    '$.code':dict(relative_path='scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',old_bytes=30395,old_sha256='22ad2227458b437e360ea3b7bf26d1a524beff4b1b2a68db17eef53af8145434',current_bytes=31343,current_sha256='262953a907a8805326c8537a81723933862ffc0e58b499d54296de1669b0b095'),
    '$.checks_code':dict(relative_path='scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline_checks.py',old_bytes=8605,old_sha256='17fba2f3f5458cbcef632b9dc26c2d82eb66d795503e7a922f54fbb0db669adc',current_bytes=8735,current_sha256='506ab9da6e56c95ad23ce18af526f376036eaaa809535353bbdb0cfb93f9ce7b')}
CLASSIFICATION='historical_first_CPU_fixture_source_not_current_dependency_unavailable_in_checked_Git_history'

def require(v,s):
    if not v:raise ValueError(s)
def namespace():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable original public bundle source')
    original=ast.parse(SOURCE.read_text());canonical=ast.dump(original,include_attributes=False)
    before=ast.parse('classify(value, legacy_external)',mode='eval').body
    after=ast.parse('classify_historical(value, legacy_external, p, location)',mode='eval').body;hits=0
    class Replace(ast.NodeTransformer):
        def visit_Call(self,n):
            nonlocal hits
            if ast.dump(n,include_attributes=False)==ast.dump(before,include_attributes=False):hits+=1;return copy.deepcopy(after)
            return self.generic_visit(n)
    tree=Replace().visit(original);require(hits==1,'exact one declaration classification call')
    class Undo(ast.NodeTransformer):
        def visit_Call(self,n):
            if ast.dump(n,include_attributes=False)==ast.dump(after,include_attributes=False):return copy.deepcopy(before)
            return self.generic_visit(n)
    require(ast.dump(Undo().visit(ast.parse(ast.unparse(tree))),include_attributes=False)==canonical,'whole original canonical source restored')
    witness=dict(original_source_sha256=SOURCE_SHA,original_full_AST_sha256=hashlib.sha256(canonical.encode()).hexdigest(),AST_call_changes=1,
        full_original_canonical_restore_proven=True,original_source_changed=False,copy_origin_gate_archive_or_numeric_policy_changes=0)
    h=dict(__file__=str(Path(__file__).resolve()),__name__='isolated_historical_public_bundle')
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE)+' [exact historical declaration scope]','exec'),h)
    return h,witness

def prepare(h,witness):
    read,bind,same=h['read'],h['bind'],h['same'];first=V2/'PIPELINE_CHECKS.json';final=V2/'PIPELINE_CHECKS_FINAL.json'
    fb,lb=bind(first),bind(final);require(fb['bytes']==lb['bytes']==2155 and fb['sha256']==FIRST_SHA and lb['sha256']==FINAL_SHA,'exact original and final CPU receipt bytes')
    old,new=read(first),read(final);require(old['schema']==new['schema']=='validated_boundary_pipeline_cpu_contract_checks_v2' and old['passed'] is True and new['passed'] is True and old['check_count']==new['check_count']==11,'first/final synthetic CPU11 history')
    current={}
    for location,spec in SPECS.items():
        field=location[2:];value=old[field];value_new=new[field];actual=bind(REPO/spec['relative_path'])
        require(value==dict(path=spec['relative_path'],origin='public_repository',sha256=spec['old_sha256'],bytes=spec['old_bytes']),'exact historical source declaration '+location)
        require(value_new==dict(path=spec['relative_path'],origin='public_repository',sha256=spec['current_sha256'],bytes=spec['current_bytes']) and same(value_new,actual),'exact final/current source '+location);current[location]=actual
    failure_path=DOC/'FIRST_PUBLIC_BUNDLE_FREEZE_FAILURE.json';failure=read(failure_path)
    require(failure['schema']=='root_first_public_bundle_freeze_failure_v18' and failure['stage']=='freeze' and failure['exit_code']==1
        and failure['own_protocol_created'] is False and failure['copy_target_created'] is False and failure['copy_calls']==failure['new_numeric_model_GT_fit_calls']==0
        and failure['source']['sha256']==SOURCE_SHA and same(failure['source'],bind(SOURCE)) and failure['failure_preserved'] is True,'preserved original pre-protocol/copy failure')
    return dict(source_adapter=witness,original_bundle_source=bind(SOURCE),ready_source=bind(Path(__file__).resolve()),first_freeze_failure=bind(failure_path),
        historical_first_CPU_receipt=fb,final_CPU_receipt=lb,current_sources=current,
        historical_source_replay_performed=False,historical_CPU11_is_not_proof_for_current_source=True,
        checked_reachable_Git_path_history_commit='1c9a7168aff9ceed28b0269cd0000c42a22c10b6',
        old_source_unavailable_in_checked_reachable_Git_path_history=True,Git_history_lookup_not_reexecuted_by_bundle_helper=True)

def main():
    h,witness=namespace();proof=prepare(h,witness);original_classify=h['classify'];original_snapshot=h['snapshot'];seen={}
    def classify_historical(value,legacy_external,path,location):
        if path==V2/'PIPELINE_CHECKS.json' and location in SPECS:
            spec=SPECS[location];require(value==dict(path=spec['relative_path'],origin='public_repository',sha256=spec['old_sha256'],bytes=spec['old_bytes']),'historical exception exact value')
            seen[location]=dict(referring_receipt=proof['historical_first_CPU_receipt'],JSON_location=location,
                historical_declared_relative_path=spec['relative_path'],historical_declared_size_bytes=spec['old_bytes'],historical_declared_sha256=spec['old_sha256'],
                current_source_binding=proof['current_sources'][location],classification=CLASSIFICATION,original_historical_receipt_preserved=True,
                old_CPU_source_not_replayed=True,current_reader_source_exempted=False)
            return None,CLASSIFICATION
        return original_classify(value,legacy_external)
    h['classify_historical']=classify_historical
    def snapshot(args):
        require(prepare(h,witness)==proof,'historical proof/source bytes changed');seen.clear();external=h['legacy_external_index']();checked={};readers={}
        for family,folder in (('V17',h['OLD_DOC']),('V18',h['DOC'])):
            for stage,name in (('statistics','STATISTICS_VERIFICATION_PROTOCOL.json'),('runtime','RUNTIME_CHECK_PROTOCOL.json')):
                proto_path=folder/name;own=h['read'](proto_path);inputs={};skipped=[]
                for alias,value in own['inputs'].items():
                    q,classification=original_classify(value,external)
                    if q is None:
                        require(value.get('origin')=='external_readonly_dependency' or (value['path'],value['bytes'],value['sha256']) in external,'reader input must be explicit public or external metadata');skipped.append(dict(alias=alias,declaration=value,classification=classification));continue
                    key=str(q)
                    if key not in checked:checked[key]=h['bind'](REPO/q)
                    require(h['same'](checked[key],value),'current actual reader input differs '+family+'.'+stage+'.'+alias);inputs[alias]=checked[key]
                readers[family+'.'+stage]=dict(protocol=h['bind'](proto_path),current_public_inputs=inputs,explicit_external_metadata=skipped,
                    all_current_public_inputs_byte_checked_before_snapshot=True,historical_source_exceptions_used_for_live_reader_inputs=False)
        value=original_snapshot(args);require(set(seen)==set(SPECS),'exact two preserved historical classifications required')
        # Parent recursive skipped declarations are also converted away from the
        # generic binding shape; these historical fields are metadata, not assets.
        for item in value['external_or_unresolved_legacy_metadata']:
            if item['classification']==CLASSIFICATION:
                raw=item.pop('declaration');item['historical_declared_relative_path']=raw['path'];item['historical_declared_size_bytes']=raw['bytes'];item['historical_declared_sha256']=raw['sha256']
        value.update(historical_CPU_source_scope=proof,historical_source_exceptions=[seen[k] for k in sorted(seen)],actual_four_reader_current_input_closure=readers,
            historical_bindings_not_rewritten_or_claimed_as_current=True);value['limits']=dict(value['limits'],historical_first_CPU_source_replayed=False,current_reader_input_mismatch_exceptions=0,historical_source_exception_count=2)
        return value
    h['snapshot']=snapshot;h['main']()

if __name__=='__main__':main()
