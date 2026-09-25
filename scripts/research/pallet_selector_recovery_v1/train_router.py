"""One synthetic-only expert router. No real modules imported or read."""
from . import common as C
READS=C.synth_guard()
import numpy as np
import torch
from . import models as M
from challenge.evaluation_v2.pose_metrics import pose_auc

def stats(err):
    return dict(mean_ADDnorm=float(np.mean(err)),median_ADDnorm=float(np.median(err)),ADDsym_AUC=pose_auc(err),available=int(np.isfinite(err).sum()),frames=len(err))

def main():
    C.setup();gpu=C.gpu();lock=C.read(C.sdoc(4)/'ROUTER_DATASET_LOCK.json');C.verify(lock['dataset']);z=dict(np.load(C.ROOT/lock['dataset']['path']))
    train=z['split']=='TRAIN';val=z['split']=='VAL';x,mean,std=M.normalize(z['X'],train)
    state,fit=M.fit(x,z['y'],train,val,linear=False,pairwise=False)
    path=C.sraw(4)/'ROUTER.pt';assert not path.exists();torch.save(dict(state=state,mean=mean,std=std,d=x.shape[1],variant='ROUTER'),path)
    C.freeze(C.sdoc(4)/'ROUTER_VAL_RESULTS.json',fit)
    C.freeze(C.sdoc(4)/'ROUTER_SELECTION_LOCK.json',dict(created_at=C.now(),checkpoint=C.bind(path),criterion='VAL expert-selection accuracy only; earliest best epoch',real_used=False,
        dataset=lock['dataset'],feature_contract=C.bind(C.sdoc(4)/'ROUTER_FEATURE_CONTRACT.json'),base_selector=C.bind(C.sdoc(4)/'BASE_SELECTOR_FOR_ROUTER.json')))
    C.freeze(C.sdoc(4)/'ROUTER_TRAINING_AUDIT.json',dict(real_GT_reads=0,guard_active=True,read_paths=sorted(set(READS)),gpu=gpu,base_optimizer_steps=0,seed=42,variants=1,TEST_used_in_fit=False))
    C.freeze(C.sdoc(4)/'ROUTER_TEST_ONCE_LOCK.json',dict(created_at=C.now(),selection=C.bind(C.sdoc(4)/'ROUTER_SELECTION_LOCK.json'),test_reads=1))
    ck=torch.load(path,map_location='cpu',weights_only=False);test=z['split']=='TEST';p=1/(1+np.exp(np.clip(-M.scores(ck,z['X'][test]),-80,80)));sel=(p>.5).astype(int)
    errors=z['errors'][test];labels=z['y'][test];conditions=z['condition'][test];applied=z['applied'][test];out={}
    masks={'ALL':np.ones(len(p),bool),'CLEAN_SYNTH':conditions=='CLEAN_SYNTH','OCCLUDED_SYNTH':conditions=='OCCLUDED_SYNTH','OCCLUDED_APPLIED_ONLY':applied}
    for name,mask in masks.items():
        e=errors[mask];s=sel[mask];n=int(mask.sum())
        out[name]=dict(n=n,accuracy=float(np.mean(s==labels[mask])),route_S0=int((s==0).sum()),route_S1=int((s==1).sum()),
            S0=stats(e[:,0]),S1=stats(e[:,1]),ROUTED=stats(e[np.arange(n),s]),ORACLE=stats(e.min(1)),brier=float(np.mean((p[mask]-labels[mask])**2)))
    C.freeze(C.sdoc(4)/'ROUTER_SYNTH_RESULTS.json',dict(groups=out,TEST_once=True,GT_oracle_posthoc_only=True,GT_reference='exact synthetic pose, C2 symmetry',no_test_tuning=True))
    print('ROUTER_TEST',out['ALL'],flush=True)

if __name__=='__main__':main()
