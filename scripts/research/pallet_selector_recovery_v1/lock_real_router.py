"""Prediction-only router decisions. No reference or severity input."""
import numpy as np
import torch
from . import common as C
from . import models as M
from . import router_features as RF

def main():
    C.setup();lock=C.read(C.sdoc(4)/'ROUTER_SELECTION_LOCK.json');C.verify(lock['checkpoint']);ck=torch.load(C.ROOT/lock['checkpoint']['path'],map_location='cpu',weights_only=False)
    fl=C.read(C.DOC/'REAL_FEATURE_LOCK.json');C.verify(fl['features']);C.verify(fl['predictions']);z=dict(np.load(C.ROOT/fl['features']['path']));pp=C.read(C.ROOT/fl['predictions']['path'])
    meta={r['id']:r for r in C.read(C.ROOT/'data/pallet/results/pallet_visible_refine_hidden_pnp_v1/INFERENCE_METADATA.json')}
    base=C.read(C.sdoc(4)/'BASE_SELECTOR_FOR_ROUTER.json')['base'];sc=None
    if base=='SYNTH_SCORER':
        sl=C.read(C.sdoc(2)/'SCORER_SELECTION_LOCK.json');C.verify(sl['checkpoint']);sc=torch.load(C.ROOT/sl['checkpoint']['path'],map_location='cpu',weights_only=False)
    ch=RF.choices(z,pp,base,sc);xx=[];names=[]
    for i,fid in enumerate(z['ids']):
        rows=[pp[a][fid] for a in C.ARMS];n=[C.HYP[int(ch[a]['selected'][i])] if ch[a]['selected'][i]>=0 else None for a in C.ARMS];names.append(n)
        xx.append(RF.make(rows[0]['prediction'],rows[1]['prediction'],rows[0]['geometry'],rows[1]['geometry'],z['S1_ctx'][i],*n,ch['S0']['margin'][i],ch['S1']['margin'][i],meta[fid]['xyz']))
    score=M.scores(ck,np.array(xx));p=1/(1+np.exp(np.clip(-score,-80,80)));dec={str(fid):dict(chosen=C.ARMS[int(v>.5)],probability_S1=float(v),margin=abs(float(v)-.5),selected_hypotheses=dict(zip(C.ARMS,n))) for fid,v,n in zip(z['ids'],p,names)}
    C.verify(lock['checkpoint']);C.freeze(C.sraw(4)/'REAL_ROUTER_DECISIONS.json',dec)
    C.freeze(C.sdoc(4)/'REAL_ROUTER_DECISION_LOCK.json',dict(created_at=C.now(),decisions=C.bind(C.sraw(4)/'REAL_ROUTER_DECISIONS.json'),GT_input=False,
        severity_input=False,session_input=False,router=lock['checkpoint'],weights_unchanged=True,base_selector=base,frames=len(dec),both_experts_same_selector=True))
    print('REAL_ROUTER_LOCKED',len(dec),flush=True)

if __name__=='__main__':main()
