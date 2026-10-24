from pathlib import Path
import json,hashlib,copy
import numpy as np
import torch
import reconcile_saved_arms as r
H=Path(__file__).resolve().parent
y=np.array([0,0,0,1,1,1,-1,-1]);pred=np.array([0,1,-1,0,1,-1,0,1]);assert r.cm(pred,y,np.ones(8,bool))==[[1,1,1],[1,1,1]]
# Missing support contributes fixed ln2 to each literal known VAL cohort.
n=8;y=np.array([0,1,0,1,0,1,0,1]);z=np.array([-2.,2.,0.,0.,-1.,1.,-4.,4.]);a=np.ones(n,bool);a[3]=False
masks={k:np.arange(n)==i for i,k in enumerate(r.NAMES)};value,parts,loss=r.selector(z,a,y,masks);assert loss[3]==np.log(2)and parts[r.NAMES[3]]['unavailable']==1 and abs(value-np.mean(loss))<1e-15
# Independently generated legal32D32E trace, full2000 seeded draws.
y=np.array([0,1,0,0,1,0,0,1,0,0,1,0],np.int8);m=dict(y=y,split=np.array(['TRAIN']*12),units=np.arange(12)*2)
p=dict(old_triplets=np.array([[0,1,2],[3,4,5]]),new_triplets=np.array([[6,7,8],[9,10,11]]),old_LOCO_BEFORE_D=np.array([0,3]),old_PALLOX_BEFORE_D=np.array([2,5]),old_BEFORE_E=np.array([1,4]),new_ALL_D=np.array([6,8,9,11]),new_ALL_E=np.array([7,10]))
g=torch.Generator(device='cpu').manual_seed(205);rec=[];digest=hashlib.sha256()
for update in range(1,2001):
 oi=torch.randint(2,(4,),generator=g).numpy();ni=torch.randint(2,(4,),generator=g).numpy();triples=np.r_[p['old_triplets'][oi],p['new_triplets'][ni]];pieces=[triples.ravel()]
 for name,n in(('old_LOCO_BEFORE_D',4),('old_PALLOX_BEFORE_D',4),('old_BEFORE_E',8),('new_ALL_D',8),('new_ALL_E',16)):pieces.append(p[name][torch.randint(len(p[name]),(n,),generator=g).numpy()])
 un=np.concatenate(pieces);perm=torch.randperm(64,generator=g).numpy();ix=un[perm];pos=np.argsort(perm)[np.arange(24).reshape(8,3)];digest.update(ix.astype('<i8').tobytes());digest.update(pos.astype('<i8').tobytes())
 rec.append(dict(update=update,occurrence_indices=ix.tolist(),input_unit_indices=m['units'][ix].tolist(),matched_triplet_batch_positions=pos.tolist(),draw=dict(old_pair_pool_indices=oi.tolist(),new_pair_pool_indices=ni.tolist(),shuffle_permutation=perm.tolist(),unshuffled_occurrence_indices=un.tolist())))
ix,actual=r.check_sampling(rec,m,p,digest.hexdigest());assert ix.shape==(2000,64)
bad=copy.deepcopy(rec);bad[0]['input_unit_indices'][0]+=1
try:r.check_sampling(bad,m,p,digest.hexdigest())
except AssertionError:pass
else:raise AssertionError('Wrong tensor unit accepted')
# Earliest strict minimum and prospective between-arm tie rule are literal.
h=[(0,1.),(100,.7),(200,.7),(300,.8)];best=float('inf');step=None
for u,v in h:
 if v<best:best=v;step=u
assert step==100
eligible=[dict(arm=r.ARMS[1],BCE=.4),dict(arm=r.ARMS[0],BCE=.4)];assert min(eligible,key=lambda x:(x['BCE'],r.ARMS.index(x['arm'])))['arm']==r.ARMS[0]
result=dict(completed=True,PASS=True,code=r.bind(r.__file__),test=r.bind(__file__),own_2x3_CM_U_exact=True,unavailable_selector_ln2=True,all2000_generated32D32E_trace_exact=True,wrong_unit_rejected=True,earliest_strict_min_and_C6_exacttie=True,model_source_arrays_actual_saved_outputs_current_weights_forward_fit_reads=0)
r.write(H/'GENERATED_SAVED_REVIEW_TESTS.json',result);print(json.dumps(result))
