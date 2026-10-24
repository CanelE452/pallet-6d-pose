"""Streaming geometry ownership resolution; no colors or semantic classes."""
import numpy as np

def update_nearest_two(best,second,owner,depth,part):
 d=np.asarray(depth,np.float64)
 better=d<best
 second[better]=best[better];best[better]=d[better];owner[better]=part
 notbetter=~better
 second[notbetter]=np.minimum(second[notbetter],d[notbetter])

def final_ownership(best,second,owner):
 tol=64*np.finfo(np.float64).eps*np.maximum(1.,np.where(np.isfinite(best),np.abs(best),1.))
 with np.errstate(invalid='ignore'):amb=np.isfinite(best)&np.isfinite(second)&(second-best<=tol)
 result=owner.copy();result[amb]=-2
 return result,amb

def generated():
 rng=np.random.default_rng(20261023);d=rng.uniform(1,9,(9,7,13));d[rng.random(d.shape)<.3]=np.inf;d[1,0,0]=d[2,0,0]=.5
 best=np.full(d.shape[1:],np.inf);second=best.copy();owner=np.full(best.shape,-1,np.int16)
 for p in range(len(d)):update_nearest_two(best,second,owner,d[p],p)
 oracle=np.sort(d,axis=0)
 assert np.array_equal(best,oracle[0]) and np.array_equal(second,oracle[1])
 out,amb=final_ownership(best,second,owner);assert amb[0,0]and out[0,0]==-2
 assert np.all((out==-2)|(out==-1)|((out>=0)&(out<9)))
 return dict(PASS=True,generated_only=True,random_layers=9,pixels=91,tie_U_preserved=True,semantic_mapping='UNKNOWN')
if __name__=='__main__':
 import json;print(json.dumps(generated()))
