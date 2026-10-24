from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,hashlib,sys
import numpy as np
import torch
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;V=O.parent;C=V/'source_fit_preparation/train_source_once.py'
s=importlib.util.spec_from_file_location('independent_c12_math',C);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);_,c10,_,d=m.source_modules()
N=m.N;typed=np.full(N,'U');domain=np.full(N,'OLD',dtype='<U40');split=np.full(N,'TRAIN',dtype='<U10');z=np.zeros(N,np.float32);a=np.ones(N,bool)
ix=np.arange(m.BASE,m.BASE+9);typed[ix]=list('DDDDEEESU');domain[ix]=m.MATERIAL;split[ix]='VALIDATION';z[ix]=[-10,-2,2,5,-4,-1,3,90,-90];a[ix[[3,4]]]=False
known=np.isin(typed,['D','E']);target=np.where(typed=='D',0,np.where(typed=='E',1,-1)).astype(np.int8);meta=dict(typed_state=typed,domain=domain,split=split,known=known,external_target=target,old={})
parts={str(i):dict(mean_BCE=float((i+1)/10),count=(i+1)*11,unavailable=0)for i in range(8)}
fake=SimpleNamespace(selector=lambda *_:(0,dict(parts)))
value,actual=m.selector10(z,a,meta,fake)
means=[]
for st in'DE':
 ii=ix[typed[ix]==st];each=[]
 for j in ii:
  zz=float(z[j])if a[j]else 0.;y=0 if st=='D'else 1;each.append(float(np.logaddexp(0.,zz)-y*zz))
 means.append(sum(each)/len(each));assert actual['material_VALIDATION_'+st]['count']==len(ii)
 assert actual['material_VALIDATION_'+st]['unavailable']==1
assert abs(value-(sum((i+1)/10 for i in range(8))+sum(means))/10)<1e-15
report,gates=m.material_report(d,z,a,meta,m.read(m.DESIGN));cm=report['VALIDATION']['matrix'];assert cm==[[2,1,1],[1,1,1]]
assert gates['material_VAL_D_recall_min']['value']==2/4 and gates['material_VAL_E_recall_min']['value']==1/3 and gates['material_VAL_known_U_rate_max']['value']==2/7 and all(not x['PASS']for x in gates.values())
raw=torch.tensor([3.,-2.,4.,1.],requires_grad=True);labels=torch.tensor([-1.,0.,1.,1.]);av=torch.tensor([True,True,False,True]);loss,bce,hinge=c10.supervised_loss(raw,labels,av,torch.empty((0,3),dtype=torch.long));expected=(np.logaddexp(0,-2)+np.logaddexp(0,1)-1)/2
assert abs(float(loss.detach())-expected)<2e-7;loss.backward();assert raw.grad[0]==raw.grad[2]==0 and raw.grad[1]!=0 and raw.grad[3]!=0
try:m.material_pools(meta,a)
except AssertionError:empty_rejected=True
else:raise AssertionError('Empty TRAIN material pool accepted')
r=dict(completed=True,PASS=True,scope='Invented metadata/logits only; no actual source arrays/checkpoint/modeldata/private/V112.',selector_equal_ten_cohorts_independent_scalar_loop=True,unavailable_known_safe0_ln2_included=True,material_CM_with_U=cm,material_recall_and_U_gates_all_known_denominators=True,S_U_excluded_not_negative=True,unknown_unavailable_loss_and_gradient_zero=True,empty_TRAIN_material_no_fallback=True,code={'path':str(C),'sha256':hashlib.sha256(C.read_bytes()).hexdigest()})
with(O/'OWN_GENERATED_MATH.json').open('x')as f:json.dump(r,f,indent=2)
print(json.dumps(r))
