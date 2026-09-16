"""Balanced same-device complete pipelines, with separate stage instrumentation."""
import time,importlib
import numpy as np
import torch,cv2
import env as E
from b_inference import ThreeLineInference

def pose(candidates,camera,dimensions):
    if not candidates:return None
    points=np.asarray(max(candidates,key=lambda c:c['score'])['keypoints_xy'],float)
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    r=select_pnp_hypotheses(points,camera,dimensions,None)
    h=next((h for h in r.hypotheses if h.name==r.selected_hypothesis and h.success),None)
    if h is None:return None
    module=importlib.import_module('run_pose_evaluation');d=h.camera_facing_dimensions.as_dict()
    model=module.cuboid(float(d['width']),float(d['height']),float(d['depth']));valid=np.isfinite(points[:8]).all(-1)
    if valid.sum()<6:return None
    return module.solve(model,points[:8],camera,valid)

def main():
    device=E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1)
    pe=E.C.old('paper_evaluation')
    reference=E.read(E.ROOT/'_docs/experiments/pallet_sensors_refinement_closeout_v1/RUNTIME_PROTOCOL.json')
    keys=reference['keys'];arms=['B0_P','B1_ONE_AMBIG','B2_THREE_EQUAL','B3_THREE_AMBIG']
    plan=dict(keys=keys,seed=1,arms=arms,warmup_per_arm=20,repeats=5,thread_intraop=4,opencv=1,
      full_pipeline='RGB+reflect/letterbox+R0+P+optional ROI/DHT/all-mode score+native decode+same PnP',
      B4='diagnostic donor control, not a deployment runtime',no_fastest_selection=True)
    E.freeze(E.DOC/'B/RUNTIME_LOCK.json',plan)
    images={k:cv2.imread(str(E.ROOT/k)) for k in keys}
    manifest=E.read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list'];lookup={pe.canonical_key(r['image']):r for r in manifest}
    import pose_evaluation_paths as paths
    contract=paths.load_pose_object_contract(str(E.C.POSE/'POSE_EVAL_OBJECT_CONTRACT.json'));metadata={}
    for k in keys:
        frame=lookup[k];a=E.read(E.ROOT/frame['annotation'])['camera_data']['intrinsics'];spec=paths.object_spec(contract,frame['object_type'])
        metadata[k]=(np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float),dict(x=spec['long_m'],y=spec['height_m'],z=spec['short_m']))
    m=ThreeLineInference();records=[];stage=[];parity_max=0.
    for arm in arms:
        saved=E.read(E.RAW/f'B/DEV_predictions/{arm}_seed1.json')['frames']
        for k in keys:
            r=m.predict(images[k],arm)
            assert len(r['candidates'])==len(saved[k])
            for a,b in zip(r['candidates'],saved[k]):
                assert a['score']==b['score'] and np.array_equal(a['box_xyxy'],b['box_xyxy'])
                d=float(np.max(abs(np.array(a['keypoints_xy'])-np.array(b['keypoints_xy']))));parity_max=max(parity_max,d)
                gain=min(640/(images[k].shape[0]+200),640/(images[k].shape[1]+200))
                assert d*gain<.0003,(arm,k,d)
                assert np.array_equal(np.array(a['keypoints_xy'])[8],np.array(b['keypoints_xy'])[8])
    for arm in arms:
        for i in range(20):
            k=keys[i%len(keys)];r=m.predict(images[k],arm);pose(r['candidates'],*metadata[k])
    settings=dict(torch_intraop=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),opencv=cv2.getNumThreads(),
      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
    for repeat in range(5):
        order=arms[repeat%4:]+arms[:repeat%4]
        if repeat%2:order=order[::-1]
        for arm in order:
            for k in keys:
                torch.cuda.synchronize();t=time.perf_counter();r=m.predict(images[k],arm);torch.cuda.synchronize();t2=time.perf_counter()
                q=pose(r['candidates'],*metadata[k]);torch.cuda.synchronize();end=time.perf_counter()
                records.append(dict(arm=arm,repeat=repeat,key=k,image_to_2D_ms=(t2-t)*1000,PnP_ms=(end-t2)*1000,full_ms=(end-t)*1000,pose_available=q is not None))
        E.write(E.RAW/'B/RUNTIME_PROGRESS.json',dict(completed_blocks=repeat+1,records=records))
        print('B runtime block',repeat+1,flush=True)
    for arm in arms:
        for k in keys:
            torch.cuda.synchronize();r=m.predict(images[k],arm,stages=True);stage.append(dict(arm=arm,key=k,stages_ms=r['stages']))
    summary={}
    for arm in arms:
        rows=[r for r in records if r['arm']==arm];summary[arm]={}
        for key in ['image_to_2D_ms','PnP_ms','full_ms']:
            v=[r[key] for r in rows];summary[arm][key]=dict(median=float(np.median(v)),P90=float(np.quantile(v,.9)),n=len(v))
    E.write(E.DOC/'B/runtime.json',dict(complete=True,device=device,settings=settings,summary=summary,records=records,
      separate_instrumented_stages=stage,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
      memory_scope='One shared wrapper with all three small P heads and DHT resident; not isolated deployment memory.',
      B0_omits_Hough_work=True,seed=1,B4_deployment_runtime=None,accuracy_parity_max_raw_px=parity_max,
      accuracy_comparison='same fixed rule; 104 untimed image/method checks against serialized DEV predictions before timing'))
    m.close();print('B RUNTIME COMPLETE',summary,flush=True)
if __name__=='__main__':main()
