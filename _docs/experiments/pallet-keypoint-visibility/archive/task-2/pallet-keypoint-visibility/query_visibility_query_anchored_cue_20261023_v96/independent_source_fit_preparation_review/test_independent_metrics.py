"""Small independent analytic fixtures; no actual source/model data."""
from pathlib import Path
import json,hashlib,importlib.util,math
import numpy as np
H=Path(__file__).resolve().parent;F=H.parent/'source_fit_preparation/train_source_once.py'
s=importlib.util.spec_from_file_location('independent_c3_metrics',F);t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
def main():
 y=np.array([0,0,0,1,1,1,-1,-1],np.int8);z=np.array([-2,2,100,-2,2,-100,-2,2.],np.float32);available=np.array([1,1,0,1,1,0,1,1],bool)
 p=np.array([-1 if not a else 1 if v>=0 else 0 for v,a in zip(z,available)],np.int8)
 cm=np.zeros((2,3),int)
 for yy,pp in zip(y,p):
  if yy>=0:cm[yy,{0:0,1:1,-1:2}[int(pp)]]+=1
 meta=dict(labels=y,known=y>=0)
 result=t.cm_metrics(z,available,meta,np.ones(8,bool));assert result['matrix']==cm.tolist()==[[1,1,1],[1,1,1]]
 assert result['known_N']==6 and result['unknown_GT_N']==2 and result['known_U_rate']==2/6
 assert result['D_recall']==result['E_recall']==1/3 and result['D_precision']==result['E_precision']==.5
 assert result['unknown_GT_predictions']=={'0':1,'1':1,'-1':0}
 # Four centre-U/edge and strict-sign endpoints remain separate from labels.
 assert t.decode(np.array([-0.,0.,-1e-7,1e-7]),np.array([1,1,1,0],bool)).tolist()==[1,1,0,-1]
 labels=np.array([0,0,0,0,0,0,1,1,1,1],np.int8)
 source=np.array(['LOCO','LOCO']+['PALLOX']*8);variant=np.array(['BEFORE']*3+['CUE_PRESERVING_SHAM']*3+['BEFORE']*2+['OPAQUE_COVER']*2)
 zz=np.array([-1,-2,3,0,-2,1,2,-3,5,2],np.float32);a=np.ones(10,bool);a[8]=False
 m=dict(labels=labels,source=source,variant=variant,split=np.array(['VALIDATION']*10))
 actual,strata=t.selector(zz,a,m)
 ls=[math.log1p(math.exp(float(v if av else 0)))-int(yy)*float(v if av else 0) for v,av,yy in zip(zz,a,labels)]
 groups=[[0,1],[2],[3,4,5],[6,7],[8,9]];expected=sum(sum(ls[i] for i in ix)/len(ix) for ix in groups)/5
 assert abs(actual-expected)<1e-14 and strata['COVER_E']['unavailable']==1
 report=dict(completed=True,PASS=True,trainer={'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),independent_2x3=cm.tolist(),known_U_denominator=6,unknown_not_negative=2,five_cohort_macro_BCE=actual,independent_five_cohort_macro_BCE=expected,unavailable_logit_BCE=math.log(2),raw_sign_zero_is_E=True,actual_source_arrays_GT_weights_model_forward_fit_current_PnP=0)
 with(H/'INDEPENDENT_ANALYTIC_GENERATED_RESULTS.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
