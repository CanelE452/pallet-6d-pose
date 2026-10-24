from pathlib import Path
import json,numpy as np,torch
import run_frozen_v61_source_once as b
C,astsha=b.proxy_class();m=C();assert sum(x.numel()for x in m.parameters())==28033
z=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(0),np.float32(-1)),2.,-2.,0.],np.float32);a=np.array([1,1,1,1,1,1,0],bool)
assert b.decode(z,a).tolist()==[0,0,0,1,0,1,-1]
p=np.array([0,1,-1,0,1,-1,0,1]);y=np.array([0,0,0,1,1,1,-1,-1]);r=b.metric(p,y,np.ones(8,bool));assert r['CM']==[[1,1,1],[1,1,1]]and r['D_recall']==1/3 and r['hidden_recall']==1/3 and r['known_prediction_U_rate']==1/3 and r['teacher_U_predictions']=={'0':1,'1':1,'-1':0}
pp=np.array([0,1,0,0,0,0,0,-1,0]);pairs=np.arange(9).reshape(3,3);split=np.array(['TRAIN']*6+['VALIDATION']*3)
assert b.pair_metric(pp,pairs,'TRAIN',split)['all3_correct']==1 and b.pair_metric(pp,pairs,'VALIDATION',split)['all3_correct']==0
# Generated head signature is P3 only; source loader/module is never imported.
with torch.inference_mode():assert m(torch.zeros(2,65,9,9)).shape==(2,)
r=dict(completed=True,PASS=True,code=b.bind(b.__file__),test=b.bind(__file__),ProxyHead_AST_SHA=astsha,parameters=28033,raw_visible_zero_signedzero_positive_subnormal_D=True,negative_subnormal_hidden=True,factual_invalid_U=True,own_CM_2x3_denominators_exact=True,pair_formula_exact=True,source_data_weights_backbone_current_fit=0)
b.write(Path(__file__).parent/'GENERATED_BASELINE_TESTS.json',r);print(json.dumps(r))
