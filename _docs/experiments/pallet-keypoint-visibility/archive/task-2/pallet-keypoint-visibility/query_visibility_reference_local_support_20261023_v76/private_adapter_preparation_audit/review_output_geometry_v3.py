"""Minimal generated export-only addendum; no data/model/runtime I/O."""
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
NEW=ROOT/'private_evaluation/prepare_and_infer_after_freeze.py'
OLD=ROOT/'private_evaluation/provenance/PRE_INVALID_GEOMETRY_ROUTE_FIX_V2/prepare_and_infer_after_freeze.py'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
old=ast.parse(OLD.read_text());new=ast.parse(NEW.read_text());normalized=copy.deepcopy(new)
removed=0
for n in ast.walk(normalized):
    if isinstance(n,ast.Call):
        keep=[]
        for k in n.keywords:
            if k.arg=='geometry_self_hidden_if_valid':removed+=1
            else:keep.append(k)
        n.keywords=keep
assert removed==1 and ast.dump(normalized,include_attributes=False)==ast.dump(old,include_attributes=False)
main=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='main')
loop=next(n for n in main.body if isinstance(n,ast.For)and isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Name)and n.iter.func.id=='enumerate'and any(isinstance(x,ast.Name)and x.id=='priorrows'for x in n.iter.args))
cases=list(itertools.product([False,True],[False,True],[False,True,None],[-2.,0.,2.]))
prior=[{'id':'generated','native_corner':i,'valid':v,'geometry_self_hidden':g,'query_x':i+.3,'query_y':i+.7}for i,(v,a,g,z)in enumerate(cases)]
env={'np':np,'priorrows':prior,'z':np.asarray([x[3]for x in cases],np.float32),'av':np.asarray([x[1]for x in cases],bool),'rows':[]}
exec(compile(ast.Module(body=[loop],type_ignores=[]),str(NEW),'exec'),env)
for (v,a,g,z),raw,out in zip(cases,prior,env['rows']):
    valid=v and a
    assert out['valid']==valid and out['geometry_self_hidden']==g
    assert out['geometry_self_hidden_if_valid']==(g if valid else None)
    assert out['status']==('OK'if valid else'UNKNOWN')
    if not valid:
        assert out['p_visible']is None and out['logit']is None and out['geometry_self_hidden_if_valid']is None
    for k in ['id','native_corner','query_x','query_y']:assert out[k]==raw[k]
report={'PASS':True,'scope':'Code AST and36generatedrowexports only; actualcurrent/sourceinputs/RGB/GT/scores/models/weights/forward0',
    'bindings':{'V3':b(NEW),'V2':b(OLD),'audit_code':b(Path(__file__)),'prior_V2_audit':b(HERE/'CODE_ONLY_GENERATED_REVIEW_V2.json')},
    'only_change':'Exactly one CSV r.update keyword geometry_self_hidden_if_valid; normalized fullprogram AST identical toV2',
    'generated_cases':len(cases),'invalid_oldflag_preserved_but_newroute_is_None':True,'valid_true_false_unknown_geometry_preserved':True,
    'q_logits_probabilities_and_validity_unchanged':True,'blockers':0,
    'consumer_requirement':'Use geometry_self_hidden_if_valid only for newV76 geometry route. Keep historicalV72 geometry_self_hidden route unchanged.'}
out=HERE/'OUTPUT_GEOMETRY_V3_ADDENDUM.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'PASS':True,'code':b(NEW),'report':b(out)}))
