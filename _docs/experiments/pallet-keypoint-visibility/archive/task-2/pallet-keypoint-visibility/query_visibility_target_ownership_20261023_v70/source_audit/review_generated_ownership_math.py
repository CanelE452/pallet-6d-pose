"""Generated-array-only V70 prototype, BCE/ranking and sampler review."""
from pathlib import Path
import hashlib, importlib, inspect, json, math, sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
paths=[ROOT/'model'/f for f in ['input_prototype.py','ownership_head.py','fit_ownership_head.py']]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={str(p):sha(p) for p in paths}
proto=importlib.import_module('input_prototype')
trainer=importlib.import_module('fit_ownership_head')
assert list(inspect.signature(proto.input_prototypes).parameters)==['x','sam','box_grid','context','target_index']
assert set(trainer.INPUT_KEYS)=={'x','rgb_native','sam','box_grid','prototype','prototype_info'}
assert 'context' not in inspect.signature(trainer.TargetOwnershipHead.forward).parameters

checks={}
a=np.arange(1,64,dtype=np.uint8);v=np.ones(63,bool)
k,y=trainer.known_ve(a,v)
assert a[k&(y==1)].tolist()==[1] and a[k&(y==0)].tolist()==[2,4,6]
assert not trainer.known_ve(np.array([1,2,4,6,63]),np.zeros(5,bool))[0].any()
checks['all63_V_E_masks_O_UNKNOWN_exclusion']=True

z=torch.tensor([-.7,.3,1.4,-2.],dtype=torch.float64,requires_grad=True)
labels=torch.tensor([1.,0.,1.,0.],dtype=torch.float64);src=torch.tensor([0,1,1,1])
loss=trainer.source_macro_bce(z,labels,src)
scalar=[math.log1p(math.exp(float(q)))-float(t)*float(q) for q,t in zip(z,labels)]
expected=(scalar[0]+sum(scalar[1:])/3)/2
assert abs(float(loss)-expected)<1e-14
grad=torch.autograd.grad(loss,z)[0].numpy()
sig=1/(1+np.exp(-z.detach().numpy()));expected_grad=(sig-labels.numpy())*np.array([.5,1/6,1/6,1/6])
assert np.max(np.abs(grad-expected_grad))<1e-14
assert abs(expected-np.mean(scalar))>1e-4
checks['BCE_per_source_mean_and_analytic_gradient']=True

z=torch.tensor([1.,-1.,-.2,.4],dtype=torch.float64,requires_grad=True)
op=torch.tensor([0,2]);cp=torch.tensor([1,3])
rank=trainer.pair_ranking_loss(z,op,cp)
g=torch.autograd.grad(rank,z)[0].numpy();diff=z.detach().numpy()[op]-z.detach().numpy()[cp]
expected=float(np.logaddexp(0,-diff).mean())
expected_g=np.zeros(4);expected_g[op]=-1/(1+np.exp(diff))/2;expected_g[cp]=1/(1+np.exp(diff))/2
assert abs(float(rank)-expected)<1e-14 and np.max(np.abs(g-expected_g))<1e-14
checks['pair_ranking_sign_normalization_and_gradient']=True

pools={'SYNTH':np.arange(100),'ORIGINAL_YCB':np.arange(100,300),'CONTROLLED_YCB':np.arange(300,500),'PALLOX':np.arange(500,600)}
pairs=np.array([[100+i,300+i]for i in range(32)],np.int64)
ra,rb=np.random.default_rng(20261023),np.random.default_rng(20261023)
for _ in range(128):
    ix,op,cp=trainer.build_batch(ra,pools,pairs);ix2,op2,cp2=trainer.build_batch(rb,pools,pairs)
    assert np.array_equal(ix,ix2) and np.array_equal(op,op2) and np.array_equal(cp,cp2)
    assert len(ix)==256 and len(op)==len(cp)==32
    assert np.array_equal(ix[cp]-ix[op],np.full(32,200))
    assert all(np.isin(ix,pool).sum()==64 for pool in pools.values())
checks['generated_128_batches_source64_pair32_permutation_identity_seed']=True

rng=np.random.default_rng(915)
n=6;x=rng.normal(size=(n,65,9,9)).astype(np.float32);x[:,64]=(rng.random((n,9,9))>.2)
sam=rng.normal(0,.2,size=(n,2,9,9)).astype(np.float32)
box=rng.uniform(-1.3,1.3,(n,4,9,9)).astype(np.float32)
context=rng.normal(size=(n,64)).astype(np.float32);group=np.array([3,3,3,8,8,12],np.int64)
p,info,audit=proto.input_prototypes(x,sam,box,context,group)
independent=[]
for i in range(n):
    numerator=np.zeros(64);den=0.
    for j in range(n):
        if i==j or group[i]!=group[j]:continue
        for yy in range(9):
            for xx in range(9):
                weight=np.clip(float(x[j,64,yy,xx]),0,1)
                weight*=int(abs(float(box[j,0,yy,xx]))<=1 and abs(float(box[j,1,yy,xx]))<=1)
                weight*=1/(1+math.exp(-np.clip(8*float(sam[j,0,yy,xx]),-32,32)))
                numerator+=x[j,:64,yy,xx].astype(np.float64)*weight;den+=weight
    independent.append(context[i] if den<=1e-8 else numerator/den)
assert np.max(np.abs(p-np.asarray(independent,dtype=np.float32)))<2e-7
xchange=x.copy();xchange[0,:64]+=11
pchange,_,_=proto.input_prototypes(xchange,sam,box,context,group)
assert np.array_equal(pchange[0],p[0])
assert np.max(np.abs(pchange[1]-p[1]))>1
assert np.array_equal(p[-1],context[-1]) and info[-1,2]==1
assert audit['pooling_uses_GT_or_VE_subset'] is False
checks['prototype_brute_native_sites_same_current_target_LOO_and_fallback']=True

rejected=0
try:proto.input_prototypes(np.zeros((9,65,9,9)),np.zeros((9,2,9,9)),np.zeros((9,4,9,9)),np.zeros((9,64)),np.zeros(9,np.int64))
except ValueError:rejected+=1
bad=x.copy();bad[0,0,0,0]=np.nan
try:proto.input_prototypes(bad,sam,box,context,group)
except ValueError:rejected+=1
assert rejected==2
checks['prototype_nonfinite_and_more_than8_ROIs_rejected']=True
assert before=={str(p):sha(p) for p in paths},'Code changed during generated review'

out=ROOT/'source_audit/INDEPENDENT_OWNERSHIP_MATH_REVIEW.json'
assert not out.exists()
report=dict(schema='V70_GENERATED_OWNERSHIP_MATH_REVIEW_V1',completed=True,status='PASS',checks=checks,
    code_bindings=[dict(path=p,sha256=s)for p,s in before.items()],review_script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
    code_review=['GT-free prototype runs on all included ROIs before knownVE filtering. Same current target/image is asserted by cache grouping; original/composite indices are distinct.',
    'Source-balanced BCE and ranking loss do not use O/U supervision. Validation selector is knownVE-only4-source macro; all-known VEO quality remains separate.',
    'Prototype fallback uses supplied same-target input context, which may include current ROI; the explicit fallback flag limits the pure LOO claim.',
    'At most8 other initial ROI context is a new deployment input assumption; overlapping ROIs are not certified clean target pixels.',
    'All inputs/code/protocol are hashed before actual source array loading. The root must bind exact immutable TRAIN_CHANGED_PAIR_INDEX.csv and cache hashes before authorizing a fit.'],
    blockers=[],fit_authorized_by_this_review=False,actual_source_RGB_GT_cache_feature_weight_current_reads=0,
    model_forward_calls=0,optimizer_updates=0,GPU_calls=0,PnP_calls=0,
    generated_prototype_calls=4,generated_loss_autograd_only=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASS',path=str(out),sha256=sha(out))))
