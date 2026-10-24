"""Eight fixed cases; saved-map arithmetic only, no model or input changes."""
from pathlib import Path
import json, hashlib, math
import numpy as np
R=Path(__file__).resolve().parent
V=R/'current_actual_crop_diagnosis_v1'
RAW=R/'private_evaluation/NATIVE_CENTER_ONE_SHOT/RAW_NATIVE_CENTRE_OUTPUTS.npz'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
policy=V/'REFERENCE_READOUT_POLICY_AFTER_FIGURE_REVIEW.json'
assert json.loads(policy.read_text())['raw_outputs']==bind(RAW)
cases=json.loads((V/'ACTUAL_CROPS_AND_FIGURES.json').read_text())['cases']
with np.load(RAW,allow_pickle=False)as z:
    maps=z['physical_external_logit_map'];valids=z['map_valid']
    lookup={k:i for i,k in enumerate(zip(z['id'].tolist(),z['native_corner'].astype(int).tolist()))}
result=[]
for c in cases:
    i=lookup[(c['id'],int(c['native_corner']))]
    q=np.array(c['query_xy']);ref=np.array(c['reference_xy']);L=c['roi_side']
    norm=2*(ref-q)/L;xy=4+9*(ref-q)/L;inside=bool(np.all(norm>=-1)&np.all(norm<1))
    zq=float(maps[i,4,4])
    out=dict(case_index=c['index'],id=c['id'],native_corner=c['native_corner'],initial_error_px=c['initial_reference_error_px'],
        original_q_logit=zq,original_q_decision='D'if zq<0 else'E',reference_inside_original_crop=inside,GT_reference_saved_map_diagnostic_only=True)
    if inside:
        x,y=map(float,np.clip(xy,0,8));x0,y0=math.floor(x),math.floor(y);x1,y1=min(8,x0+1),min(8,y0+1)
        a=x-x0;b=y-y0
        entries=[(y0,x0,(1-a)*(1-b)),(y0,x1,a*(1-b)),(y1,x0,(1-a)*b),(y1,x1,a*b)]
        den=sum(w*float(valids[i,yy,xx])for yy,xx,w in entries)
        num=sum(w*float(valids[i,yy,xx])*float(maps[i,yy,xx])for yy,xx,w in entries)
        out.update(reference_sampling_index_xy=xy.tolist(),reference_factual_available=den>0,reference_logit=num/den if den else None,
            reference_diagnostic_decision=('D'if num/den<0 else'E')if den else'U')
        out['point_readout_decision_changes']=bool(den and(out['reference_diagnostic_decision']!=out['original_q_decision']))
    result.append(out)
output=dict(completed=True,policy=bind(policy),code=bind(Path(__file__)),cases=result,
    actual_original_current_metrics_unchanged=True,new_forward_fit_PnP=0)
serialized=json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n'
path=V/'FIXED8_SAVED_REFERENCE_READOUT_DIAGNOSTIC.json'
with path.open('x')as f:f.write(serialized)
print(json.dumps(result,ensure_ascii=False,indent=2))
