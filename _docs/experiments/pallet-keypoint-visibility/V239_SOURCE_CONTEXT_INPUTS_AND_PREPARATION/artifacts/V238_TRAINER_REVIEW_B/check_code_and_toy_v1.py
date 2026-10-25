"""Bounded V238 code/random-CPU review. No actual SOURCE/model/weight loader."""
from pathlib import Path
import ast, hashlib, json, math
import numpy as np
import torch
from torch import nn
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V238_TRAINER_REVIEW_B'
PINS={
 'trainer':(R/'V238_CONTEXT_RESIDUAL_TRAINER_A/train_context_residual.py','74fa6517c5058a8a1f5fc0dd4ad298d24b5ece72c9056b569ca99a198ab24e24'),
 'module':(R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py','dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9'),
}
texts={}
for name,(p,sha) in PINS.items():
 assert not any(s in str(p).lower() for s in ('v112','__pycache__','.pyc'))
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;texts[name]=b.decode()
parsed=ast.parse(texts['trainer'])
selected=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in ('relative_xy','node_args','alignment','source_gate')]
ns={'np':np,'torch':torch}
exec(compile(ast.Module(body=selected,type_ignores=[]),'<pure trainer functions>','exec'),ns)
module={'__name__':'pure_random_relational'}
exec(compile(texts['module'],'<pure module>','exec'),module)
checks=[]
# Same original seed17 base constructed before any residual parameters.
torch.manual_seed(17);base=module['CropFamilyHead']()
torch.manual_seed(17);head=module['RelationalQueryHead']()
assert sum(p.numel() for p in head.parameters())==86054
assert len(list(head.parameters()))==10
assert all(torch.equal(v,head.base.state_dict()[k]) for k,v in base.state_dict().items())
x=torch.randn(3,1186);nodes=torch.randn(3,8,384);ctx=torch.randn(3,384)
slots=torch.tensor([0,3,7]);orig=torch.ones(3,8,dtype=torch.bool);supp=orig.clone();supp[:,2]=False
rel=torch.zeros(3,8,2);context_ok=torch.tensor([True,False,True])
y,details=head(x,nodes,ctx,slots,orig,supp,context_ok,rel)
assert torch.equal(y,base(x)) and torch.isfinite(y).all()
assert (details['attention_other7'][:,2]==0).all()
assert (details['attention_other7'][torch.arange(3),slots]==0).all()
checks.append('seed17 unchanged base,86054/10,zero residual exact output,input-only other-node mask')
# Independent float64 geometry result, unavailable NaN retained as zero, no original mutation.
q=np.array([[[1.,2.],[11.,-8.],[1000.,1000.],[np.nan,np.nan],[1.,2.],[1.,2.],[1.,2.],[1.,2.]]],np.float64)
av=np.array([[1,1,1,0,1,1,1,1]],bool);L=np.full((1,8),10.,np.float64)
g={'q8':q,'L8':L,'original_available8':av,'node_available':av.copy(),'context_fraction':np.ones(1,np.float32)}
r=ns['relative_xy'](g,np.array([0]),np.array([0]))
assert r.dtype==np.float32 and np.array_equal(r[0,1],[1.,-1.]) and np.array_equal(r[0,2],[8.,8.])
assert not r[0,0].any() and not r[0,3].any() and np.isnan(q[0,3]).all()
args=ns['node_args'](np.array([0]),np.array([0]),np.array([0]),r,g,np.zeros((1,8,384),np.float32),np.zeros((1,384),np.float32),'cpu')
assert torch.equal(args[3],torch.from_numpy(av)) and torch.equal(args[4],torch.from_numpy(g['node_available']))
checks.append('F64 input-only relative geometry,clip8/FP32,self-Uzero,separate exact support mask')
# Caller rejects silently checking a supported subset rather than all active rows.
class Partial:
 @staticmethod
 def verify_active_node_alignment(*a,**kw):return {'checked':2,'max_abs':0.}
try:ns['alignment'](Partial,x,(nodes,ctx,slots,orig,supp,context_ok,rel))
except ValueError:pass
else:raise AssertionError('Partial active alignment accepted')
checks.append('alignment caller requires checked==batch length')
# Fixed gate arithmetic, not a trained result or a revised criterion.
a={f'family{f}_DEV':{'D':{'recall':.5},'E':{'recall':.4}} for f in (0,1)}
b={f'family{f}_DEV':{'D':{'recall':.6},'E':{'recall':.5}} for f in (0,1)}
z=ns['source_gate'](a,b)
assert z['PASS'] and abs(z['mean_family_DEV_balanced_recall_gain_pp']-10)<1e-9
b['family1_DEV']['E']['recall']=.49
assert not ns['source_gate'](a,b)['PASS']
checks.append('unchanged prospective3pp/2pp/YCB-E50% gate')
run=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='run')
head_lines=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='head']
assert head_lines==[384,424]
assert 424<429<433<434
result={'completed':True,'PASS':True,'scope':'CODE and invented CPU tensors only','checks':checks,
 'chronology_finding':{'V1_preexec_repair_required':True,'head_output_line':424,'all_DEV_count_guard_line':429,'whole_DEV_receipt_line':433,'whole_CONTROL_receipt_line':434},
 'actual_SOURCE_or_model_weights_or_pixels_or_feature_or_fit':0}
p=H/'INDEPENDENT_CODE_AND_TOY_V1.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'PASS':True,'groups':len(checks),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
