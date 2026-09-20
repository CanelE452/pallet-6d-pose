"""Same-session R0/N0/N2 seed1 timing, using only historical DEV images."""
import time
import cv2
import numpy as np
import torch
from scripts.evaluation import final_dimension_release as R


@torch.no_grad()
def main():
    lock=R.checked_lock(); E=R.setup()
    from inference import load_head,predict_captured,registry_input,serial
    from pose import infer
    device=E.gpu();torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    pe=E.old('paper_evaluation')
    keys=R.read(R.ROOT/'_docs/experiments/pallet_sensors_refinement_closeout_v1/RUNTIME_PROTOCOL.json')['keys']
    arms=['R0',*R.ARMS]
    R.freeze(R.DOC/'RUNTIME_PROTOCOL.json',dict(keys=keys,arms=arms,seed=1,warmup=20,repeats=5,
        threads=4,interop_threads=1,opencv_threads=1,all_valid_samples_retained=True,
        order='rotate starting arm by repeat, reverse odd repeats',
        scope='BGR in RAM -> reflect/letterbox/R0 -> optional P -> prediction-only PnP',
        excludes='image decoding, camera acquisition, GUI',
        memory='shared R0 and both small heads, process-scoped',model_lock=R.binding(R.DOC/'MODEL_LOCK.json')))
    images={k:cv2.imread(str(R.ROOT/k)) for k in keys}
    frames={pe.canonical_key(r['image']):r for r in R.read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']}
    meta={}
    for k in keys:
        r=frames[k];a=R.read(R.ROOT/r['annotation'])['camera_data']['intrinsics']
        K=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float)
        d,g=registry_input(r['object_type']);meta[k]=(K,d,g)
    heads={a:load_head(a,1)[0] for a in R.ARMS}
    extractor=E.old('features').FrozenYoloFeatures(E.R0)
    norm=R.read(R.DCP/'DIM_NORMALIZATION_LOCK.json');records=[]
    def run(arm,key):
        torch.cuda.synchronize();begin=time.perf_counter();cap=extractor.predict(images[key])
        torch.cuda.synchronize();captured=time.perf_counter();K,d,g=meta[key]
        if arm=='R0': result=dict(candidates=serial(cap['candidates']),selected_index=cap['selected_index'])
        else:
            result,_=predict_captured(heads[arm],arm,cap,d,g,lock['temperatures'][arm+'_seed1'],
                lock['decode_rule'],images[key].shape[:2],norm)
        torch.cuda.synchronize();refined=time.perf_counter();idx=result['selected_index']
        pose=infer(None if idx is None else result['candidates'][idx]['keypoints_xy'],K,d[[0,2,1]])
        torch.cuda.synchronize();end=time.perf_counter()
        return result,dict(full_ms=1000*(end-begin),RGB_to_2D_ms=1000*(refined-begin),
            R0_ms=1000*(captured-begin),refinement_ms=1000*(refined-captured),
            PnP_ms=1000*(end-refined),pose_available=pose['available'])
    try:
        for arm in R.ARMS:
            saved={r['key']:r for r in R.read(E.RAW/f'predictions/REAL_DEV/{arm}_seed1.json')['records']}
            for k in keys:
                result,_=run(arm,k)
                assert result['selected_index']==saved[k]['selected_index']
                assert len(result['candidates'])==len(saved[k]['candidates'])
                for a,b in zip(result['candidates'],saved[k]['candidates']):
                    for field in ('score','box_xyxy','keypoints_xy'):
                        assert np.array_equal(a[field],b[field]),(arm,k,field)
        for arm in arms:
            for i in range(20): run(arm,keys[i%len(keys)])
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(5):
            order=arms[repeat%3:]+arms[:repeat%3]
            if repeat%2: order=order[::-1]
            for arm in order:
                E.gpu()
                for key in keys:
                    _,measurement=run(arm,key)
                    records.append(dict(arm=arm,key=key,repeat=repeat,**measurement))
        fields=('full_ms','RGB_to_2D_ms','R0_ms','refinement_ms','PnP_ms')
        summary={a:{k:dict(median=float(np.median([r[k] for r in records if r['arm']==a])),
                          P90=float(np.quantile([r[k] for r in records if r['arm']==a],.9))) for k in fields} for a in arms}
        R.freeze(R.DOC/'RUNTIME_RESULTS.json',dict(complete=True,summary=summary,records=records,
            parity_exact=True,device=device,samples_per_arm=5*len(keys),
            threads=dict(torch=torch.get_num_threads(),interop=torch.get_num_interop_threads(),opencv=cv2.getNumThreads()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),memory_scope='shared process, not isolated per arm',
            protocol=R.binding(R.DOC/'RUNTIME_PROTOCOL.json'),code=R.binding(__file__)))
        print(summary,flush=True)
    finally: extractor.close()


if __name__=='__main__':main()
