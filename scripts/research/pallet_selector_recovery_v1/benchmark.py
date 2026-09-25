"""Single-GPU sequential dual-expert pilot cost. No GT or severity input."""
import time
import cv2
import numpy as np
import torch
from . import common as C
from . import features as F
from . import models as M
from . import router_features as RF
from .extract_features import Expert

def main():
    C.setup();gpu=C.gpu();wall=time.perf_counter();rows=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['heldout'][:30]
    meta={r['id']:r for r in C.read(C.ROOT/'data/pallet/results/pallet_visible_refine_hidden_pnp_v1/INFERENCE_METADATA.json')}
    b=C.read(C.sdoc(4)/'ROUTER_SELECTION_LOCK.json')['checkpoint'];C.verify(b);ck=torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=False)
    base=C.read(C.sdoc(4)/'BASE_SELECTOR_FOR_ROUTER.json')['base'];sc=None
    if base=='SYNTH_SCORER':
        sb=C.read(C.sdoc(2)/'SCORER_SELECTION_LOCK.json')['checkpoint'];C.verify(sb);sc=torch.load(C.ROOT/sb['path'],map_location='cpu',weights_only=False)
    models={a:Expert(a) for a in C.ARMS};images=[cv2.imread(str(C.ROOT/r['image']['path'])) for r in rows]
    padded=[cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101) for im in images]
    for _ in range(5):
        for a in C.ARMS:models[a].run([padded[0]])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for r,im,pad in zip(rows,images,padded):
        torch.cuda.synchronize();start=time.perf_counter();parts={};pp={};z={'ids':np.array([r['id']])};md=meta[r['id']]
        for a in C.ARMS:
            st=time.perf_counter();p,ctx=models[a].run([pad]);torch.cuda.synchronize();parts[a+'_raw_ms']=(time.perf_counter()-st)*1000;p=p[0]
            for c in p['candidates']:
                c['keypoints_xy']=(np.array(c['keypoints_xy'])-100).tolist();c['box_xyxy']=(np.array(c['box_xyxy'])-100).tolist()
            g=F.extract(p,md['K'],md['xyz'],im.shape[:2]);parts[a+'_with_geometry_ms']=(time.perf_counter()-st)*1000;pp[a]={r['id']:dict(prediction=p,geometry=g)}
            z[a+'_geo']=np.array([g['features'] if g['valid'] else np.zeros((2,len(F.names())))]);z[a+'_ctx']=ctx;z[a+'_valid']=np.array([g['valid']]);z[a+'_current']=np.array([C.HYP.index(g['selection']) if g['selection'] in C.HYP else -1])
        route_start=time.perf_counter();ch=RF.choices(z,pp,base,sc);ps=[pp[a][r['id']] for a in C.ARMS];names=[C.HYP[int(ch[a]['selected'][0])] if ch[a]['selected'][0]>=0 else None for a in C.ARMS]
        x=RF.make(ps[0]['prediction'],ps[1]['prediction'],ps[0]['geometry'],ps[1]['geometry'],z['S1_ctx'][0],*names,ch['S0']['margin'][0],ch['S1']['margin'][0],md['xyz'])
        M.scores(ck,x[None]);parts['router_CPU_ms']=(time.perf_counter()-route_start)*1000;parts['dual_total_ms']=(time.perf_counter()-start)*1000;times.append(parts)
    for m in models.values():m.integrity()
    stats={k:dict(mean=float(np.mean([r[k] for r in times])),median=float(np.median([r[k] for r in times])),P90=float(np.quantile([r[k] for r in times],.9))) for k in times[0]}
    C.freeze(C.sdoc(4)/'COMPUTE_COST.json',dict(created_at=C.now(),frames=30,batch=1,warmup_iterations_per_expert=5,selector=base,timings_ms=stats,
        GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,wall_seconds=time.perf_counter()-wall,
        raw_timings=times,gpu_start=gpu,gpu_end=C.gpu(),includes='Sequential dual RGB inference, GAP hook CPU transfer, production PnP/geometric features, router CPU evaluation; checkpoint loading and image decode excluded from timed samples',
        caveat='RTX3080 pilot, both experts resident. Python router wrapper reconstructs tiny CPU MLP per sample (conservative cost); not a Jetson benchmark or optimized deployment.'))
    print('DUAL_COST',stats['dual_total_ms'],flush=True)

if __name__=='__main__':main()
