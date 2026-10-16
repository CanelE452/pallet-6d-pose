"""Preserve the original freeze failure; honor declared external metadata origin.

Execute the full original stdlib report inspector with one public-path AST
guard added. Tables, raw scalar joins and arithmetic are unchanged. The
separate review source/failure/AST authority is bound into all own receipts.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

CODE=Path(__file__).resolve().parent
SOURCE=CODE/'report_scalar_check.py'
SOURCE_SHA='5650b1401e956ba29b16509d293b03e8c5ccc4bf5d95af77bcd34a18e259c5a6'
DOC=CODE.parents[2]/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
FAILURE=DOC/'REPORT_PUBLIC_METADATA_FIRST_FAILURE.json'

def require(value,message):
    if not value:raise ValueError(message)

def binding(path):
    p=Path(path).absolute();require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'review evidence missing/symlink')
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def kernel():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'original report checker must remain unchanged')
    failure=json.loads(FAILURE.read_text())
    require(failure['schema']=='preserved_stable_incidence_report_public_metadata_first_failure_v14'
        and failure['complete'] is False and failure['CLI_stage']=='report_scalar_check.freeze'
        and failure['original_checker']['sha256']==SOURCE_SHA and failure['original_checker']['bytes']==binding(SOURCE)['bytes']
        and failure['failed_before_own_protocol'] is True and failure['failed_before_STARTED'] is True
        and failure['new_statistics_CI_timing_GT_fit_model_calls']==0 and failure['error']['type']=='ValueError'
        and failure['cause']=='external_readonly_dependency origin ignored when classifying declared public paths'
        and failure['missing_external_dependency_opened'] is False and failure['original_checker_modified'] is False,
        'preserve the actual zero-arithmetic metadata-classification freeze failure')
    tree=ast.parse(SOURCE.read_text());original=ast.dump(tree,include_attributes=False)
    target=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='public_path']
    require(len(target)==1,'one original public_path function')
    guard=ast.parse("if value.get('origin') == 'external_readonly_dependency':\n    return None").body[0]
    target[0].body.insert(0,copy.deepcopy(guard));derived=ast.dump(tree,include_attributes=False)
    restored=copy.deepcopy(tree);function=next(n for n in restored.body if isinstance(n,ast.FunctionDef) and n.name=='public_path')
    require(ast.dump(function.body[0],include_attributes=False)==ast.dump(guard,include_attributes=False),'exact external-origin guard')
    function.body.pop(0);require(ast.dump(restored,include_attributes=False)==original,'full canonical original AST restoration')
    env=dict(__file__=str(SOURCE),__name__='isolated_external_origin_report_review')
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE)+' [declared external origin only]','exec'),env)
    marker=dict(original_checker=binding(SOURCE),original_failure=binding(FAILURE),review_checker=binding(Path(__file__)),
        original_checker_modified=False,original_freeze_attempt_passed=False,original_failure_before_protocol_and_STARTED=True,
        public_path_AST_sites=1,external_origin_classifier_only=True,missing_file_heuristic_added=False,
        full_canonical_original_AST_restoration_proven=True,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_full_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),scalar_arithmetic_AST_changes=0,
        original_tables_saved_rows_or_physical_policy_changed=False,new_private_asset_or_experiment_calls=0)
    oldclosure,oldwrite=env['closure'],env['write']
    def closure(args):
        files,b,scope=oldclosure(args)
        for key,path in (('origin_review_source',Path(__file__)),('original_failed_report_checker',SOURCE),('original_public_metadata_failure',FAILURE)):
            files[key]=path;b[key]=env['binding'](path)
        return files,b,scope
    def write(path,value):
        value['origin_classification_review']=marker;oldwrite(path,value)
    env.update(closure=closure,write=write)
    return env

def main():kernel()['main']()

if __name__=='__main__':main()
