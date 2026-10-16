"""Rerun unchanged seven CPU groups with their declared source environment.

The first four PASS/three missing-PALLET_SOURCE_ROOT errors stay immutable.
No guard, test, selected observation or numerical policy is changed.
"""
import ast
import hashlib
import json
import os
from pathlib import Path

CODE=Path(__file__).resolve().parent
REPO=CODE.parents[2]
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
SOURCE=CODE/'cpu_checks.py'
FILES=('REVIEW_CPU_PROTOCOL.json','REVIEW_CPU_STARTED.json','REVIEW_CPU_CHECKS.json')
CONFIG=dict(PALLET_SOURCE_ROOT='/home/minjae/Documents/github/pallet-pose',
    PALLET_BASELINE_ROOT='/home/minjae/Documents/github/pallet-pose-handoff-20261006')
REVIEW=dict(original_failed_CPU_attempt_preserved=True,original_tests_and_guard_changed=False,
    original_tests_run=7,original_errors=3,environment_only_correction=True,
    same_seven_groups_reexecuted=True,configured_environment=CONFIG,
    controlled_fixtures_are_not_natural_optimizer_results=True)

def require(v,s):
    if not v:raise ValueError(s)

def main():
    old=json.loads((DOC/'CPU_CHECKS.json').read_text())
    first=json.loads((DOC/'CPU_PROTOCOL.json').read_text())
    require(old['schema']=='stable_incidence_CPU_checks_v14' and old['complete'] is True
        and old['passed'] is False and old['tests_run']==7 and old['failures']==[]
        and len(old['errors'])==3 and all("KeyError: 'PALLET_SOURCE_ROOT'\n" in e['traceback'] for e in old['errors'])
        and old['error'] is None and old['source_unchanged'] is True and old['opencv_threads_restored'] is True
        and old['cleanup_errors']==[] and old['inputs']==first['inputs'],
        'preserve original four-PASS/three-source-environment-errors CPU result')
    tree=ast.parse(SOURCE.read_text());witness=dict(
        full_original_CPU_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        original_CPU_AST_edits=0)
    env=dict(__file__=str(SOURCE),__name__='unchanged_incidence_CPU_environment_review')
    exec(compile(tree,str(SOURCE)+' [unchanged CPU AST; declared environment]','exec'),env)
    require(env['binding'](DOC/'CPU_PROTOCOL.json')==old['own_protocol'] and
        json.loads((DOC/'CPU_STARTED.json').read_text())['protocol']==old['own_protocol'],
        'original CPU protocol/STARTED differs')
    require(env['inputs']()==old['inputs'],'original CPU dependencies changed')
    original_inputs,original_write=env['inputs'],env['write']
    def inputs():
        out=original_inputs()
        for p in (Path(__file__).resolve(),DOC/'CPU_PROTOCOL.json',DOC/'CPU_STARTED.json',DOC/'CPU_CHECKS.json'):
            out[str(p.relative_to(REPO))]=env['binding'](p)
        return out
    def write(path,value):
        value['environment_review']=dict(REVIEW,**witness)
        original_write(path,value)
    env.update(inputs=inputs,write=write,FILES=FILES)
    previous={k:os.environ.get(k) for k in CONFIG}
    try:
        for key,value in CONFIG.items():
            require(Path(value).is_dir(),'declared source environment directory missing')
            os.environ[key]=value
        env['main']()
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        require(previous=={k:os.environ.get(k) for k in CONFIG},'CPU environment restoration failed')

if __name__=='__main__':main()
