"""Balanced N0/N4 RGB-to-PnP timing on the same historical closeout images."""
import time
import numpy as np
import torch,cv2
import dcp_env as E
from inference import load_head,predict_captured,registry_input
from pose import infer

@torch.no_grad()
def main():
    assert E.read(E.DOC/'METADATA_SENSITIVITY.json')['complete'];device=E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1)
    pe=E.old('paper_evaluation');keys=E.read(E.ROOT/'_docs/experiments/pallet_sensors_refinement_closeout_v1/RUNTIME_PROTOCOL.json')['keys'];arms=['N0_BASE_REPLAY','N4_META_SYM']
    E.freeze(E.DOC/'RUNTIME_PROTOCOL.json',dict(keys=keys,arms=arms,seed=1,warmup=20,repeats=5,threads=4,opencv=1,balanced=True,full='image in RAM -> reflect/letterbox/R0 -> P -> fixed prediction-only PnP',
      P_only='feature capture to refined candidate serialization, including conditioning and preservation checks',isolation='both small heads resident; shared R0; peak allocated is process-scoped, not per-arm isolated'))
    images={k:cv2.imread(str(E.ROOT/k)) for k in keys};frames={pe.canonical_key(r['image']):r for r in E.read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']};meta={}
    for k in keys:
        r=frames[k];a=E.read(E.ROOT/r['annotation'])['camera_data']['intrinsics'];K=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float);d,g=registry_input(r['object_type']);meta[k]=(K,d,g)
    heads={a:load_head(a,1)[0] for a in arms};extractor=E.old('features').FrozenYoloFeatures(E.R0);norm=E.read(E.DOC/'DIM_NORMALIZATION_LOCK.json');selection=E.read(E.DOC/'CALIBRATION_AND_SELECTION.json');records=[]
    def run(arm,k):
        torch.cuda.synchronize();begin=time.perf_counter();cap=extractor.predict(images[k]);torch.cuda.synchronize();captured=time.perf_counter()
        K,d,g=meta[k];T=selection['temperatures'][arm+'_seed1']['temperature'];result,_=predict_captured(heads[arm],arm,cap,d,g,T,selection['rule'],images[k].shape[:2],norm)
        torch.cuda.synchronize();refined=time.perf_counter();idx=result['selected_index'];pose=infer(None if idx is None else result['candidates'][idx]['keypoints_xy'],K,d[[0,2,1]])
        torch.cuda.synchronize();end=time.perf_counter();return result,dict(full_ms=(end-begin)*1000,R0_ms=(captured-begin)*1000,P_only_ms=(refined-captured)*1000,PnP_ms=(end-refined)*1000,pose_available=pose['available'])
    for arm in arms:
        saved={r['key']:r for r in E.read(E.RAW/f'predictions/REAL_DEV/{arm}_seed1.json')['records']}
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
    E.write(E.DOC/'RUNTIME_AND_MEMORY.json',dict(complete=True,device=device,summary=s,records=records,parity_exact=True,
      params=E.read(E.DOC/'MODEL_AND_PARAMETER_AUDIT.json')['parameters'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
      memory_scope='shared R0 and both small heads resident, after warmup; not isolated per-arm memory',runtime_samples_per_arm=5*len(keys)))
    extractor.close();print('RUNTIME_COMPLETE',s,flush=True)
if __name__=='__main__':main()
