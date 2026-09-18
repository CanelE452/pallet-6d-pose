"""Same balanced hardware timing and exact fresh-R0/cached prediction parity."""
import time
import numpy as np
import torch,cv2
import cv_env as E
from code_adapter import alter,mode_for,context
from evaluate import head_for
from inference import predict_captured,registry_input
from pose import infer

@torch.no_grad()
def main():
    device=E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1);pe=E.D.old('paper_evaluation')
    keys=E.read(E.D.DOC/'RUNTIME_PROTOCOL.json')['keys'];arms=E.ARMS['A'];rule=E.protocol()['rule'];norm=E.read(E.D.DOC/'DIM_NORMALIZATION_LOCK.json')
    E.freeze(E.DOC/'RUNTIME_PROTOCOL.json',dict(keys=keys,arms=arms,seed=1,T=1.,warmup=20,repeats=5,threads=4,opencv=1,balanced=True,
      scope='RGB in RAM through R0, conditioning/refinement/serialization and fixed prediction-only PnP; both heads resident',B_same_architecture=True))
    images={k:cv2.imread(str(E.ROOT/k)) for k in keys};frames={pe.canonical_key(r['image']):r for r in E.read(E.D.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']};meta={}
    for k in keys:
        r=frames[k];a=E.read(E.ROOT/r['annotation'])['camera_data']['intrinsics'];K=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float);d,g=registry_input(r['object_type']);meta[k]=(K,d,g)
    heads={a:head_for(a,1)[0] for a in arms};extractor=E.D.old('features').FrozenYoloFeatures(E.D.R0);records=[]
    def run(arm,k):
        torch.cuda.synchronize();begin=time.perf_counter();cap=extractor.predict(images[k]);torch.cuda.synchronize();captured=time.perf_counter()
        K,d,g=meta[k];z=alter(torch.from_numpy(context(d[None],[g],norm,True)),mode_for(arm)).numpy()
        result,_=predict_captured(heads[arm],'N4_META_SYM',cap,d,g,1.,rule,images[k].shape[:2],norm,z)
        torch.cuda.synchronize();refined=time.perf_counter();idx=result['selected_index'];p=infer(None if idx is None else result['candidates'][idx]['keypoints_xy'],K,d[[0,2,1]])
        torch.cuda.synchronize();end=time.perf_counter()
        return result,dict(full_ms=(end-begin)*1000,R0_ms=(captured-begin)*1000,P_only_ms=(refined-captured)*1000,PnP_ms=(end-refined)*1000,pose_available=p['available'])
    for arm in arms:
        saved={r['key']:r for r in E.read(E.RAW/f'predictions/DEV/{arm}_seed1_PRIMARY.json')['records']}
        for k in keys:
            result,_=run(arm,k);reference=saved[k];assert len(result['candidates'])==len(reference['candidates'])
            for a,b in zip(result['candidates'],reference['candidates']):
                for field in ['score','box_xyxy','keypoints_xy']:assert np.array_equal(np.array(a[field]),np.array(b[field])),(arm,k,field)
        for i in range(20):run(arm,keys[i%len(keys)])
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(5):
        for arm in (arms if repeat%2==0 else arms[::-1]):
            for k in keys:
                _,r=run(arm,k);records.append(dict(arm=arm,key=k,repeat=repeat,**r))
    s={a:{k:dict(median=float(np.median([r[k] for r in records if r['arm']==a])),P90=float(np.quantile([r[k] for r in records if r['arm']==a],.9))) for k in ['full_ms','R0_ms','P_only_ms','PnP_ms']} for a in arms}
    E.write(E.DOC/'RUNTIME_AND_PARAMS.json',dict(complete=True,device=device,summary=s,records=records,parity_exact=True,params=E.read(E.DOC/'CAPACITY_PARITY.json')['params'],
      peak_allocated_bytes=torch.cuda.max_memory_allocated(),memory_scope='Shared R0 + both heads, process scoped not isolated',samples_per_arm=5*len(keys),B_same_cost_by_architecture=True))
    extractor.close();print('CODE_RUNTIME_COMPLETE',s,flush=True)
if __name__=='__main__':main()
