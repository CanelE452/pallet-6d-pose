"""Apply the synthetic-selected scorer without opening any real references."""
import numpy as np
import torch
from . import common as C
from . import models as M

def main():
    lock=C.read(C.sdoc(2)/'SCORER_SELECTION_LOCK.json');C.verify(lock['checkpoint']);ck=torch.load(C.ROOT/lock['checkpoint']['path'],map_location='cpu',weights_only=False)
    fl=C.read(C.DOC/'REAL_FEATURE_LOCK.json');C.verify(fl['features']);z=dict(np.load(C.ROOT/fl['features']['path']));dec={}
    for arm in C.ARMS:
        score=M.scores(ck,M.pack_inputs(z,arm,lock['winner']));idx=M.selection(score,C.HYP);swap=M.selection(score[:,::-1],C.HYP[::-1]);assert np.array_equal(idx,1-swap)
        idx=np.where(z[arm+'_valid'],idx,z[arm+'_current'])
        dec[arm]={str(fid):dict(selected=C.HYP[int(i)] if i>=0 else None,scores=s.tolist(),margin=abs(float(s[0]-s[1])),fallback=not bool(v)) for fid,i,s,v in zip(z['ids'],idx,score,z[arm+'_valid'])}
    C.verify(lock['checkpoint'])
    C.freeze(C.sraw(3)/'REAL_SCORER_DECISIONS.json',dec)
    C.freeze(C.sdoc(3)/'REAL_SCORER_DECISION_LOCK.json',dict(created_at=C.now(),GT_input=False,
        decisions=C.bind(C.sraw(3)/'REAL_SCORER_DECISIONS.json'),scorer=lock['checkpoint'],selection_lock=C.bind(C.sdoc(2)/'SCORER_SELECTION_LOCK.json'),
        features=fl['features'],weights_unchanged=True,candidate_order_swap_invariance=True,frames=128,arms=list(C.ARMS)))
    print('REAL_SCORER_LOCKED',flush=True)

if __name__=='__main__':main()
