"""Own prospective source-metric/selector arithmetic. Invented arrays only."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def metrics(z,y,available):
 z=np.asarray(z,np.float64);y=np.asarray(y,np.int8);available=np.asarray(available,bool)
 known=(y==0)|(y==1);assert z.shape==y.shape==available.shape
 safe=np.where(available,z,0.)
 assert np.isfinite(safe[known]).all()
 prediction=np.where(available,(safe>=0).astype(np.int8),-1)
 cm=np.array([[int(np.sum((y==t)&(prediction==p)))for p in [0,1,-1]]for t in [0,1]])
 losses=np.logaddexp(0,safe[known])-y[known]*safe[known]
 macro=.5*np.mean(losses[y[known]==0])+.5*np.mean(losses[y[known]==1])
 return dict(CM=cm.tolist(),macro=float(macro),D_recall=float(cm[0,0]/cm[0].sum()),E_recall=float(cm[1,1]/cm[1].sum()),known_U=float(cm[:,2].sum()/cm.sum()))

def earliest(history):
 best=float('inf');update=None
 for u,loss in history:
  assert u>=0 and u%100==0 and np.isfinite(loss)
  if loss<best:best=loss;update=u
 return update,best

def main():
 z=[-1000.,-1.,1.,np.nan,-1e-12,0.,np.nan]
 y=[0,0,0,0,1,1,-1]
 a=[True,True,True,False,True,True,True]
 # An unknown NaN must never enter known arithmetic. Avoid computing its sign.
 z[-1]=0.
 r=metrics(z,y,a);assert r['CM']==[[2,1,1],[1,1,0]]
 assert r['D_recall']==.5 and r['E_recall']==.5 and r['known_U']==1/6
 assert earliest([(0,.69),(100,.72),(200,.6),(300,.6),(400,.61)])==(200,.6)
 assert earliest([(0,.5),(100,.7),(200,.7)])==(0,.5)
 # The fixed known denominator includes unsupported rows at safe0 BCE=ln2.
 two=metrics([10.,np.nan],[0,1],[True,False]);assert two['CM']==[[0,1,0],[0,0,1]]
 assert abs(two['macro']-(np.logaddexp(0,10)+np.log(2))/2)<1e-15
 # Sign/tiny-zero edge cases: rounded sigmoid is not the decoder.
 assert metrics([-1e-12,0.],[0,1],[True,True])['CM']==[[1,0,0],[0,1,0]]
 out=HERE/'GENERATED_METRIC_ORACLE_BEFORE_STABLE_CODE.json';assert not out.exists()
 result=dict(completed=True,PASS=True,scope='Prospective independent arithmetic only; no source data/model/fit execution.',
     fixed_generated_CM=r['CM'],known_U_denominator_exact=True,safe0_unavailable_BCE_ln2=True,
     unknown_excluded=True,update0_included=True,earliest_strict_tie=True,rawsign_tiny_zero=True)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
