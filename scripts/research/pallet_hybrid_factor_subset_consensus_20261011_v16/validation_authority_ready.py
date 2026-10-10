"""Explicit authority for the separate V16 scalar environment correction.

No original failed attempt is changed into a PASS. This stdlib-only helper
joins the preserved zero-arithmetic failure to an own-frozen successful run.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
ORIGINAL=CODE/'validation_checks.py'
ORIGINAL_SHA='1567db06b27e0394e95a539c943eac19eb57b5d257ebccf8d24a65866bbf697c'
FILES=('VALIDATION_READY_PROTOCOL.json','VALIDATION_READY_STARTED.json','VALIDATION_READY_CHECKS.json')
ALIASES=dict(zip(('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'),FILES))

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=d.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))

def source_proof():
    require(binding(ORIGINAL)['sha256']==ORIGINAL_SHA,'immutable original checker')
    tree=ast.parse(ORIGINAL.read_text());original=ast.dump(tree,include_attributes=False)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='helpers')
    calls=[n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='h' for t in n.targets)]
    require(len(calls)==1 and isinstance(calls[0],ast.Call) and isinstance(calls[0].func,ast.Name) and calls[0].func.id=='dict' and not calls[0].args and not any(k.arg=='ast' for k in calls[0].keywords),'one isolated helpers namespace missing ast')
    target=calls[0];target.keywords.append(ast.keyword(arg='ast',value=ast.Name(id='ast',ctx=ast.Load())))
    derived=ast.dump(tree,include_attributes=False);target.keywords.pop()
    require(ast.dump(tree,include_attributes=False)==original,'whole canonical checker restoration')
    target.keywords.append(ast.keyword(arg='ast',value=ast.Name(id='ast',ctx=ast.Load())))
    marker=dict(original_checker_changed=False,scalar_formula_or_threshold_changes=0,
        AST_environment_keyword_insertions=1,inserted_keyword='ast=ast',
        whole_original_checker_canonical_restore_proven=True,original_Audit_or_raw_arithmetic_calls=0,
        original_source_sha256=ORIGINAL_SHA,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        ready_full_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),geometry_or_solver_rerun=False)
    return ast.fix_missing_locations(tree),marker

def failure_proof(folder):
    folder=Path(folder);p,s,e=[read(folder/n) for n in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','FIRST_SCALAR_ENVIRONMENT_FAILURE.json')]
    require(not (folder/'VALIDATION_CHECKS.json').exists(),'preserved original has no CHECKS')
    require(e['schema']=='preserved_hybrid_scalar_environment_failure_v16' and e['checker_sha256']==ORIGINAL_SHA and e['error_type']=='NameError' and e['error_message']=="name 'ast' is not defined" and e['own_checks_written'] is False and e['independent_checks_completed']==e['new_PnP_optimizer_J_SVD_model_GT_scoring_calls']==0 and e['original_checker_protocol_started_geometry_preserved'] is True and e['performance_threshold_changed'] is False and e['geometry_rerun'] is False,'exact original pre-arithmetic environment failure')
    require(p['schema']=='hybrid_factor_subset_scalar_protocol_v16' and p['inputs']==s['inputs'] and same(s['own_protocol'],binding(folder/'VALIDATION_PROTOCOL.json')) and s['new_production_calls']==0 and same(p['checker'],binding(ORIGINAL)),'preserved original freeze and STARTED')
    require(e['completed_geometry_frames']==245 and e['completed_geometry_rows']==490 and e['geometry_sha256']==binding(folder/'GEOMETRY_SEALED.jsonl.gz')['sha256'] and e['ledger_sha256']==binding(folder/'CONTROL_LEDGERS.jsonl.gz')['sha256'],'same completed geometry, no pose rerun')
    return p

def geometry_authority(folder=DOC):
    folder=Path(folder);original=failure_proof(folder);_,marker=source_proof()
    own,start,check=[read(folder/n) for n in FILES]
    b={k:binding(p) for k,p in dict(original_checker=ORIGINAL,original_protocol=folder/'VALIDATION_PROTOCOL.json',
        original_started=folder/'VALIDATION_STARTED.json',original_failure=folder/'FIRST_SCALAR_ENVIRONMENT_FAILURE.json',
        ready_checker=CODE/'validation_environment_ready.py',ready_protocol=folder/FILES[0],
        ready_started=folder/FILES[1],ready_checks=folder/FILES[2],authority_code=Path(__file__)).items()}
    require(own['schema']=='hybrid_factor_subset_scalar_protocol_v16' and check['schema']=='hybrid_factor_subset_scalar_checks_v16' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['new_production_calls']==0,'actual separate environment-ready geometry PASS')
    require(own['inputs']==start['inputs']==check['inputs'] and own['environment_ready']==start['environment_ready']==check['environment_ready']==marker and all(same(v['ready_checker'],b['ready_checker']) and same(v['authority_code'],b['authority_code']) for v in (own,start,check)),'current explicit environment-ready source and proof')
    require(same(check['own_protocol'],b['ready_protocol']) and same(start['own_protocol'],b['ready_protocol']) and same(own['checker'],b['original_checker']) and same(check['checker'],b['original_checker']),'original scalar source versus actual ready protocol distinguished')
    extras=dict(ready_checker='ready_checker',authority_code='authority_code',original_validation_protocol='original_protocol',original_validation_started='original_started',original_failure='original_failure')
    require(set(own['inputs'])==set(original['inputs'])|set(extras) and all(own['inputs'][k]==v for k,v in original['inputs'].items()) and all(same(own['inputs'][k],b[v]) for k,v in extras.items()),'all original inputs unchanged plus exact ready/failure closure')
    for k,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):require(same(own['inputs'][k],binding(folder/name)),'actual checked current '+k)
    return dict(bindings=b,marker=marker,checks=check)
