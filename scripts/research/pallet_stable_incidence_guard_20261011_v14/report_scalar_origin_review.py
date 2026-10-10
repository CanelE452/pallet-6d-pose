"""Classify legacy experiment metadata by its exact frozen external declaration.

Preserve both pre-protocol failures and all scalar/table functions. A second
public-path guard joins only declared runtime source aliases by path, bytes
and SHA. Missing-file existence never determines public versus external.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

CODE=Path(__file__).resolve().parent
SOURCE=CODE/'report_scalar_review.py'
SOURCE_SHA='6a98b3f129ca8bf9f279cfd844992c62a0540c68213f0d10ff1dba936c05e113'
DOC=CODE.parents[2]/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'

def require(value,message):
    if not value:raise ValueError(message)

def main():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'preserve first origin review')
    inherited=dict(__file__=str(SOURCE),__name__='unchanged_first_origin_review')
    exec(compile(ast.parse(SOURCE.read_text()),str(SOURCE),'exec'),inherited)
    env=inherited['kernel']();bind=env['binding'];failure_path=DOC/'REPORT_ORIGIN_REVIEW_FIRST_FAILURE.json'
    failure=env['read'](failure_path)
    require(failure['schema']=='preserved_stable_incidence_origin_review_metadata_failure_v14'
        and failure['CLI_stage']=='report_scalar_review.freeze' and failure['error']['type']=='ValueError'
        and failure['failed_before_own_protocol_and_STARTED'] is True and failure['protocol_exists'] is False
        and failure['new_statistics_or_private_GT_calls']==failure['new_model_or_numerical_experiment_calls']==0
        and failure['source_modified'] is False and failure['missing_external_file_opened'] is False
        and env['same'](failure['review_checker'],bind(SOURCE)),'actual second zero-arithmetic failure')
    protocol_path=DOC/'RUNTIME_PROTOCOL.json';receipt_path=DOC/'RUNTIME_RECEIPT.json'
    protocol,receipt=env['read'](protocol_path),env['read'](receipt_path)
    prefix='fresh_v7:source:';declared={}
    for alias,value in protocol['inputs'].items():
        if alias.startswith(prefix) and value.get('origin')=='external_readonly_dependency':
            name=alias[len(prefix):];p=Path(name)
            if not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'):
                declared[(name,value['bytes'],value['sha256'])]=dict(alias=alias,declaration=value)
    def legacy_external(value):
        return value.get('origin')=='experiment' and (value['path'],value['bytes'],value['sha256']) in declared
    matches=[value for value in receipt['model_bindings'] if legacy_external(value)]
    require(receipt['complete'] is True and matches,'explicit legacy runtime declaration witness required')
    tree=ast.parse(env['CODE'].joinpath('report_scalar_check.py').read_text());original=ast.dump(tree,include_attributes=False)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='public_path')
    guards=ast.parse("if value.get('origin') == 'external_readonly_dependency':\n return None\nif review_legacy_external(value):\n return None").body
    function.body[:0]=copy.deepcopy(guards);revised=ast.dump(tree,include_attributes=False)
    restored=copy.deepcopy(tree);target=next(n for n in restored.body if isinstance(n,ast.FunctionDef) and n.name=='public_path')
    require(all(
        ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False) for a,b in zip(target.body[:2],guards)),'two exact origin guards')
    del target.body[:2];require(ast.dump(restored,include_attributes=False)==original,'whole scalar inspector AST restoration')
    env['review_legacy_external']=legacy_external
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'[two declared-origin guards]','exec'),env)
    marker=dict(review_checker=bind(Path(__file__)),first_origin_review=bind(SOURCE),first_origin_review_failure=bind(failure_path),
        runtime_protocol=bind(protocol_path),runtime_receipt=bind(receipt_path),public_path_AST_sites=2,
        full_canonical_original_AST_restoration_proven=True,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_full_AST_sha256=hashlib.sha256(revised.encode()).hexdigest(),scalar_arithmetic_AST_changes=0,
        legacy_classification_requires_origin_path_SHA_and_bytes=True,missing_file_heuristic_added=False,
        original_two_failed_attempts_preserved=True,external_declared_metadata_is_not_current_asset_verification=True,
        original_tables_saved_rows_or_physical_policy_changed=False,new_private_asset_or_experiment_calls=0,
        matched_legacy_runtime_bindings=matches,external_alias_declarations=[dict(path=k[0],**v) for k,v in declared.items()])
    oldclosure,oldwrite=env['closure'],env['write']
    def closure(args):
        files,b,scope=oldclosure(args)
        for key,path in (('legacy_origin_review_source',Path(__file__)),('first_origin_review_failure',failure_path)):
            files[key]=path;b[key]=bind(path)
        return files,b,scope
    def write(path,value):value['legacy_external_origin_review']=marker;oldwrite(path,value)
    env.update(closure=closure,write=write);env['main']()

if __name__=='__main__':main()
