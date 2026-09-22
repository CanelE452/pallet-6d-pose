"""Frozen forward only. Evaluation reference is deliberately not loaded here."""
import argparse
import copy
import gc
import numpy as np
import torch
import cv2
from . import common as C
from . import data as D
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1.train import load_fit as historical_FULL
B=C.B

def load_model(weight):
    if weight=='FULL':return historical_FULL('FULL')
    if weight=='PRIOR1':return B.L.load_base().cuda().eval().requires_grad_(False)
    assert weight=='C';fit=C.read(C.DOC/'FIT_C.json');C.verify(fit['checkpoint'])
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    m=B.model('FULL');m.load_state_dict(ck['model_state_dict'],strict=True);assert C.state_hash(m.state_dict())==fit['final_state_sha']
    return m.cuda().eval().requires_grad_(False)

@torch.no_grad()
def infer(arm):
    p=C.protocol();assert arm in ('B','C','D');assert C.read(C.DOC/'PRETRAIN_TESTS.json')['PASS']
    for b in C.read(C.DOC/'CODE_LOCK.json')['files']:C.verify(b)
    B.L.setup('cuda');cv2.setNumThreads(1);weight,expansion=C.ARMS[arm];m=load_model(weight);before=C.state_hash(m.state_dict())
    frozen=C.read(B.E.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0'];lock=C.read(C.DOC/'INPUT_LOCK.json');pred={};heat={}
    dest=C.RAW/'heatmaps'/arm;dest.mkdir(parents=True,exist_ok=False)
    for i,r in enumerate(lock['eval_records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));raw=frozen[r['id']];inp=D.prepare_input(im,raw,expansion)
        old=D.CORE.prepare_input(im,raw);check=D.prepare_input(im,raw,C.BASE_EXPANSION)
        assert inp is not None and old is not None
        for key in ('rgb','points','valid','matrix','box'):np.testing.assert_array_equal(check[key],old[key])
        result=copy.deepcopy(raw);args=[torch.as_tensor(inp[k],device='cuda')[None] for k in ('rgb','points','valid')]
        z=m(*args);q=expectation(z)[0].cpu().numpy();xy=D.transform_points(q,np.linalg.inv(inp['matrix']))
        xy[~inp['valid']]=inp['original_points'][~inp['valid']];xy[8]=inp['original_points'][8]
        D.CORE.selected(result)['keypoints_xy']=xy.tolist();B.E.assert_preserved(raw,result);pred[r['id']]=result
        path=dest/f'{i:04d}.npz'
        with path.open('xb') as f:np.savez_compressed(f,logits=z[0].cpu().numpy(),matrix=inp['matrix'],expectation=q,valid=inp['valid'])
        heat[r['id']]=C.bind(path)
        if (i+1)%70==0:print('INFER',arm,i+1,B.L.gpu_guard(),flush=True)
    assert C.state_hash(m.state_dict())==before
    path=C.RAW/f'PREDICTIONS_{arm}.json';C.save(path,dict(predictions=pred,GT_input=False,weight=weight,expansion=expansion,model_state_sha=before))
    C.save(C.DOC/f'PREDICTIONS_{arm}.json',dict(predictions=C.bind(path),heatmaps=heat,state_sha=before,GT_access=False,frozen=True))
    del m;gc.collect();torch.cuda.empty_cache();print('FROZEN',arm,flush=True)

def freeze_all():
    lock=C.read(C.DOC/'INPUT_LOCK.json');predA=C.bind(B.RAW/'PREDICTIONS_FULL.json');C.verify(predA)
    heatA={k:v['cache'] for k,v in C.read(C.H.DOC/'HEATMAP_FREEZE.json')['models']['FULL'].items()}
    # A predictions and logits already frozen and reproduced before this experiment.
    arms={'A':dict(predictions=predA,heatmaps=heatA,state_sha=lock['FULL_fit']['final_state_sha'],frozen=True)}
    for arm in ('B','C','D'):arms[arm]=C.read(C.DOC/f'PREDICTIONS_{arm}.json')
    assert arms['A']['state_sha']==arms['B']['state_sha'];assert arms['C']['state_sha']==arms['D']['state_sha']
    for a,r in arms.items():
        C.verify(r['predictions']);assert len(r['heatmaps'])==len(lock['eval_records'])
        for b in r['heatmaps'].values():C.verify(b)
    C.save(C.DOC/'PREDICTION_LOCK.json',dict(arms=arms,all_frozen_before_scoring=True,primary_predeclared='C'))
    print('ALL_PREDICTIONS_FROZEN',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['B','C','D','freeze']);a=p.parse_args()
    freeze_all() if a.stage=='freeze' else infer(a.stage)
