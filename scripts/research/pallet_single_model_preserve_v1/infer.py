"""Freeze raw outputs and fixed selector decisions without opening real GT."""
import time
import cv2
import numpy as np
import torch
from . import common as C
from .adapter import Model
from scripts.research.pallet_selector_recovery_v1 import features as F,models as M

def pose(p,meta,scorer):
    g=F.extract(p,meta['K'],meta['dims'],meta['hw']);name=g['selection'];scores=None
    if g['valid']:
        scores=M.scores(scorer,np.array(g['features'],np.float32)[None])[0];name=C.P.HYP[int(M.selection(scores[None],C.P.HYP)[0])]
    hh=[dict(name=h['name'],pose=F.production_pose(h,meta['dims']),score=h['score']) for h in g['hypotheses']]
    selected=next((h['pose'] for h in hh if h['name']==name),dict(available=False));d9=next((h['pose'] for h in hh if h['name']==g['selection']),dict(available=False))
    return dict(current=selected,hypotheses=hh,D9=d9,scorer_scores=scores,selected=name)

def main():
    C.setup();gpu=C.gpu();m=Model(load=True);ck=torch.load(C.ROOT/m.bindings['scorer']['path'],map_location='cpu',weights_only=False);rows=C.read(C.RAW/'INFERENCE_INPUTS.json');old=C.read(C.PREV_RAW/'PREDICTIONS.json')['S1'];pred={pop:{a:{} for a in ('BASE','PRES1')} for pop in rows};times={a:[] for a in ('BASE','PRES1')};hw={};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for pop,rr in rows.items():
        for i,r in enumerate(rr):
            im=cv2.imread(str(C.ROOT/r['image']['path']));hw[r['id']]=im.shape[:2]
            for arm,enabled in [('BASE',False),('PRES1',True)]:
                torch.cuda.synchronize();t=time.perf_counter();p=m.predict(im,100 if pop=='real' else 0,enabled);torch.cuda.synchronize()
                if pop=='real' and i>=10:times[arm].append((time.perf_counter()-t)*1000)
                if pop=='real' and arm=='BASE':
                    a,b=C.selected(p),C.selected(old[r['id']]);assert (a is None)==(b is None)
                    if a:assert np.allclose(a['keypoints_xy'],b['keypoints_xy'],atol=1e-4,rtol=1e-6)
                pred[pop][arm][r['id']]=p
            a,b=pred[pop]['BASE'][r['id']],pred[pop]['PRES1'][r['id']];assert len(a['candidates'])==len(b['candidates']) and a['selected_index']==b['selected_index']
            for x,y in zip(a['candidates'],b['candidates']):assert x['box_xyxy']==y['box_xyxy'] and x['score']==y['score'] and x['keypoints_conf']==y['keypoints_conf']
            if i%64==0:print('PRES1_INFER',pop,i,'/',len(rr),flush=True)
    m.integrity();C.freeze(C.RAW/'RAW_PREDICTIONS.json',pred);C.freeze(C.DOC/'RAW_PREDICTIONS_LOCK.json',dict(created_at=C.now(),predictions=C.bind(C.RAW/'RAW_PREDICTIONS.json'),GT_input=False,
        checkpoint=C.read(C.DOC/'FIT.json')['checkpoint'],S1=m.bindings['checkpoint'],scorer=m.bindings['scorer'],box_class_conf_unchanged=True,base_reproduction=True))
    poses={};pose_start=time.perf_counter()
    for arm in ('BASE','PRES1'):
        poses[arm]={r['id']:pose(pred['real'][arm][r['id']],dict(r,hw=hw[r['id']]),ck) for r in rows['real']}
    C.freeze(C.RAW/'POSE_DECISIONS.json',poses);C.verify(m.bindings['scorer']);C.freeze(C.DOC/'POSE_DECISIONS_LOCK.json',dict(created_at=C.now(),poses=C.bind(C.RAW/'POSE_DECISIONS.json'),raw_lock=C.bind(C.DOC/'RAW_PREDICTIONS_LOCK.json'),selector=m.bindings['scorer'],same_selector=True,GT_input=False,oracle_used=False))
    C.freeze(C.DOC/'COMPUTE_COST.json',dict(created_at=C.now(),model_params=sum(p.numel() for p in m.net.model.parameters()),additional_params=sum(p.numel() for p in m.adapter.parameters()),
        timings_ms={a:dict(n=len(v),mean=np.mean(v),median=np.median(v),P90=np.quantile(v,.9)) for a,v in times.items()},GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,wall_seconds=time.perf_counter()-start,pose_seconds=time.perf_counter()-pose_start,gpu_start=gpu,gpu_end=C.gpu(),
        timing='one GPU S1 with hook disabled/enabled sequential, batch1, excludes first10 pairs, image decode excluded; PnP/scorer time separate; RTX3080 not Jetson',training_seconds=C.read(C.DOC/'FIT.json')['seconds']))
    print('RAW_AND_POSE_FROZEN',flush=True)
if __name__=='__main__':main()
