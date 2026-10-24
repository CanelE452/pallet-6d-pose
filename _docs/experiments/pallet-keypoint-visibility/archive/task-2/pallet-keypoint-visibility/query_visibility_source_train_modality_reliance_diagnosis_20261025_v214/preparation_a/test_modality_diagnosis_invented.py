"""Small authored I/O, branch-zero, metric and replay fixtures; source bodies0."""
from pathlib import Path
import importlib.util,io,json,hashlib
import numpy as np
import torch
H=Path(__file__).parent
sp=importlib.util.spec_from_file_location('v214',H/'diagnose_source_train_modalities.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
checks=[]
def reject(fn):
 try:fn()
 except(AssertionError,ValueError):return
 raise AssertionError('Expected rejection')
# Exercise production NPY row reader, not np.load(fullarray).
a=np.arange(48,dtype=np.int64).reshape(8,2,3)
for version in[(1,0),(2,0)]:
 f=io.BytesIO();np.lib.format.write_array(f,a,version=version,allow_pickle=False);f.seek(0);z=m.npy_selected(f,[6,1,4],a.shape,a.dtype);m.need(np.array_equal(z,a[[6,1,4]]),'Selected-only ordered rows')
checks.append('NPY v1/v2 header plus selected rowbytes in caller order; unselected bytes opaque')
f=io.BytesIO();np.lib.format.write_array(f,a,allow_pickle=False);f.seek(0);reject(lambda:m.npy_selected(f,[1,1],a.shape,a.dtype));checks.append('Duplicate row selection rejects')
u=np.array(['alpha','beta','gamma']);f=io.BytesIO();np.lib.format.write_array(f,u,allow_pickle=False);f.seek(0);m.need(m.npy_selected(f,[2],u.shape,None).tolist()==['gamma'],'Unicode selected META row');checks.append('Selected Unicode META header')
m.need(m.meta_equal(np.array([np.nan,np.nan]),None,'query_xy')and m.meta_equal(np.array(np.nan),None,'L')and not m.meta_equal(np.array(np.nan),None,'query_xy'),'Null q pair/Lscalar strict');checks.append('Original U/null META exact shapes')
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.spatial_residual=torch.nn.Sequential(torch.nn.Linear(96,1,bias=False));self.spatial_residual[0].weight.data.fill_(1.);self.eval().requires_grad_(False)
 def forward(self,x):return self.spatial_residual(x)
model=Toy();x=torch.arange(192,dtype=torch.float32).reshape(2,96);before=m.tensor_state_hash(model);factual=model(x).detach()
for mode,sl in[(m.MODES[2],(80,96)),(m.MODES[3],(72,96)),(m.MODES[4],(64,72)),(m.MODES[5],(64,96))]:
 expected=x.clone();expected[:,sl[0]:sl[1]]=0
 with m.condition_intervention(model,mode):m.need(torch.equal(model(x),expected.sum(1,keepdim=True)),'Exact fixed branch slice')
 m.need(torch.equal(model(x),factual)and m.tensor_state_hash(model)==before,'Hook removed, no state modification')
checks.append('Production scoped hooks zero only fixed condition slices; features64/weights unchanged and hook removed')
raw=torch.rand((2,8,96,96),dtype=torch.float32);const=m.input_intervention(raw,m.MODES[1]);m.need(torch.all(const[:,:3]==np.float32(127/255))and torch.equal(const[:,3:],raw[:,3:])and not torch.equal(raw[:,:3],const[:,:3]),'Constant127 retains valid/XY/SAM');checks.append('Production fixedRGB127 no fitted statistic or nonRGB change')
m.verify_replay(np.array([-1.,1.],np.float32),np.array([-1.+1e-5,1.-1e-5],np.float32));reject(lambda:m.verify_replay(np.array([-1e-6],np.float32),np.array([1e-6],np.float32)));reject(lambda:m.verify_replay(np.array([1.],np.float32),np.array([1.001],np.float32)));checks.append('Production replay tolerance accepts tiny same-sign; rejects sign drift/over1e-4')
t=np.array([0,0,1],np.int8);z=np.array([-1.,1.,np.nan],np.float32);cm=m.confusion(t,m.decoded(z));m.need(cm['CM_D_E_by_D_E_U']==[[1,1,0],[0,0,1]]and cm['U']==1,'D/E/U retains unresolved');checks.append('Full CM P/R/U denominators')
d=m.deltas(np.array([-1.,1.,-1.],np.float32),np.array([1.,-1.,-1.],np.float32),t);m.need(d['sign_flips']==2 and d['transition_D_E_U']==[[1,1,0],[1,0,0],[0,0,0]],'Paired transition exact');checks.append('Same-row logit and sign transitions')
reject(lambda:m.safe(str(m.P/'blocked_v112/never_open.npz')));reject(lambda:m.payload_file({'path':'/never/open.npz','sha256':'no'}));checks.append('Lexical and finite payload whitelist reject before access')
result=dict(completed=True,PASS=True,checks=checks,code=m.bind(H/'diagnose_source_train_modalities.py'),actual_SOURCE_NPZ_PNG_weights_scores_or_model_reads=0,only_authored_CPU_toy_model=True)
with(H/'INVENTED_CHECKS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
