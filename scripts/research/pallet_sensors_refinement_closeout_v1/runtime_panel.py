"""Balanced actual-image pipelines; no training/scoring concurrent with timing."""
import time,gc,importlib
import numpy as np
import torch
from env import *
from point_inference import PointInference
from direct_inference import DirectInference
def main_pose(candidates,camera,dimensions):
    if not candidates:return None
    points=np.asarray(max(candidates,key=lambda c:c['score'])['keypoints_xy'],float)
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    result=select_pnp_hypotheses(points,camera,dimensions,None)
    hypothesis=next((h for h in result.hypotheses if h.name==result.selected_hypothesis and h.success),None)
    if hypothesis is None:return None
    dims=hypothesis.camera_facing_dimensions.as_dict()
    module=importlib.import_module('run_pose_evaluation')
    model=module.cuboid(float(dims['width']),float(dims['height']),float(dims['depth']))
    usable=np.isfinite(points[:8]).all(-1)
    if usable.sum()<6:return None
    return module.solve(model,points[:8],camera,usable)
def run():
    verify();assert read(DOC/'UNIFIED_DEV_RESULTS.json')['complete']
    if (DOC/'RUNTIME_PANEL.json').exists():assert read(DOC/'RUNTIME_PANEL.json')['complete'];return
    torch.set_num_threads(4);status=gpu();assert not status['foreign_compute'],status
    # Also check no experiment training or canonical scoring process is active on CPU.
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    for line in processes.splitlines():
        if any(f'pallet_sensors_refinement_closeout_v1/{f}' in line for f in ('train_direct.py','analysis_panel.py','evaluate_direct.py')):
            assert 'python' not in line or line.strip().split()[0]==str(os.getpid()),line
    plan=read(DOC/'RUNTIME_PROTOCOL.json');er=old('evaluate_real');pe=old('paper_evaluation');_,cache,_=er.baseline_inputs(LINE)
    images={k:er.load_bgr(k,cache['frame_metadata'][k]) for k in plan['keys']}
    manifest=read(C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list'];lookup={pe.canonical_key(r['image']):r for r in manifest}
    import pose_evaluation_paths as paths
    contract=paths.load_pose_object_contract(str(C.POSE/'POSE_EVAL_OBJECT_CONTRACT.json'));metadata={}
    for k in plan['keys']:
        frame=lookup[k];ann=read(ROOT/frame['annotation']);raw=ann['camera_data']['intrinsics'];spec=paths.object_spec(contract,frame['object_type'])
        metadata[k]=(np.array([[raw['fx'],0,raw['cx']],[0,raw['fy'],raw['cy']],[0,0,1]],float),dict(x=spec['long_m'],y=spec['height_m'],z=spec['short_m']))
    models={'R0':er.PlainBaseline(R0)};records=[];params={};prior={};memory=[]
    gt=read(C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames'];pose_reference={'R0':{r['frame_id']:r for r in read(C.POSE/'POSE_PER_FRAME_BY_ARM.json')['per_frame']['R0']}}
    try:
        for a in ('P','L','D'):
            for s in (1,2,3):
                name=f'{a}{s}'
                models[name]=PointInference(s) if a=='P' else DirectInference(s) if a=='D' else old('inference').PalletLinePoseInference(LINE/f'runs/image_line_only_seed{s}/last.pt',LINE/'SELECTION.json')
                path=BRAW/f'evaluation/{name}' if a=='P' else RAW/f'evaluation/{name}' if a=='D' else LINE/f'evaluation/image_line_only_seed{s}'
                prior[name]={r['image_key']:r['prediction'] for r in read(path/'IMAGE_PREDICTIONS.json')['records']}
                pose_label=f'image_line_only_seed{s}' if a=='L' else name
                pose_reference[name]={r['frame_id']:r for r in read(path/'POSE_PER_FRAME_BY_ARM.json')['per_frame'][pose_label]}
        for name,model in models.items():
            for i in range(plan['warmup']):
                k=plan['keys'][i%len(plan['keys'])];pred=model.predict(images[k]);main_pose(pred if name=='R0' else pred['candidates'],*metadata[k])
            if name=='R0':params[name]=dict(total=sum(p.numel() for p in model.model.model.parameters()),refiner_trainable=0)
            else:
                head=getattr(model,'head',None)
                if head is None:head=getattr(model,'model',None)
                params[name]=dict(refiner_trainable=sum(p.numel() for p in head.parameters()) if head is not None else (19810 if name.startswith('L') else None))
        for repeat in range(plan['repeats']):
            names=plan['models'];order=names[repeat:]+names[:repeat]
            if repeat%2:order=order[::-1]
            assert not gpu()['foreign_compute']
            for name in order:
                model=models[name];torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated()
                for k in plan['keys']:
                    torch.cuda.synchronize();begin=time.perf_counter();pred=model.predict(images[k]);torch.cuda.synchronize();t2=time.perf_counter()
                    candidates=pred if name=='R0' else pred['candidates'];pose=main_pose(candidates,*metadata[k]);torch.cuda.synchronize();end=time.perf_counter()
                    if name=='R0':
                        assert len(pred)==len(cache['frames'][k])
                        for c,r in zip(pred,cache['frames'][k]):
                            for field in ('score','box_xyxy','keypoints_xy'):er.exact(c[field],r[field],f'R0/{k}/{field}')
                    else:
                        er.check_prediction(pred,cache['frames'][k],frame_key=k)
                        for c,r in zip(pred['candidates'],prior[name][k]['candidates']):er.exact(c['keypoints_xy'],r['keypoints_xy'],f'{name}/{k}/accuracy')
                    fid=lookup[k]['frame_id'];expected=pose_reference[name].get(fid)
                    assert (pose is not None)==(expected is not None)
                    if pose is not None:
                        from symmetry_aware_pose_metrics import rotation_error_degrees,yaw_error_degrees
                        rotation,translation,_=pose;truth=gt[fid]
                        assert abs(np.linalg.norm(translation-np.asarray(truth['t_gt']))*100-expected['translation_error_cm'])<1e-8
                        assert abs(rotation_error_degrees(rotation,np.asarray(truth['R_gt_representative']))-expected['rotation_error_deg'])<1e-8
                        assert abs(yaw_error_degrees(rotation,np.asarray(truth['R_gt_representative']))-expected['yaw_error_deg'])<1e-8
                    records.append(dict(model=name,repeat=repeat,key=k,image_to_2d_ms=(t2-begin)*1000,pnp_ms=(end-t2)*1000,end_to_end_ms=(end-begin)*1000,pose_available=pose is not None))
                memory.append(dict(model=name,repeat=repeat,resident_allocated_bytes=initial,peak_allocated_bytes=torch.cuda.max_memory_allocated(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial))
            print('RUNTIME_BLOCK_COMPLETE',repeat+1,flush=True)
    finally:
        for name,m in models.items():
            if hasattr(m,'close'):m.close()
    # Separate allocated-memory panel: one model resident at a time. These
    # untimed calls never replace a latency sample or choose a faster repeat.
    models.clear();del model,m;gc.collect();torch.cuda.empty_cache();isolated_memory={}
    for name in plan['models']:
        if name=='R0':single=er.PlainBaseline(R0)
        elif name.startswith('P'):single=PointInference(int(name[1:]))
        elif name.startswith('D'):single=DirectInference(int(name[1:]))
        else:single=old('inference').PalletLinePoseInference(LINE/f'runs/image_line_only_seed{name[1:]}/last.pt',LINE/'SELECTION.json')
        single.predict(images[plan['keys'][0]]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for k in plan['keys']:single.predict(images[k])
        torch.cuda.synchronize();isolated_memory[name]=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='one model in this process; allocator bytes, display/external processes excluded; extra26 calls untimed')
        if hasattr(single,'close'):single.close()
        del single;gc.collect();torch.cuda.empty_cache()
    stats=lambda x:dict(n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),p90=float(np.quantile(x,.9)))
    base={(r['key'],r['repeat']):r for r in records if r['model']=='R0'};summary={}
    for name in plan['models']:
        rr=[r for r in records if r['model']==name]
        summary[name]={f:stats([r[f] for r in rr]) for f in ('image_to_2d_ms','pnp_ms','end_to_end_ms')}
        summary[name]['paired_added_2d_ms']=stats([r['image_to_2d_ms']-base[(r['key'],r['repeat'])]['image_to_2d_ms'] for r in rr])
        summary[name]['paired_added_e2e_ms']=stats([r['end_to_end_ms']-base[(r['key'],r['repeat'])]['end_to_end_ms'] for r in rr])
    for name in params:
        if name!='R0':params[name]['total']=params['R0']['total']+params[name]['refiner_trainable']
    write(DOC/'RUNTIME_PANEL.json',dict(complete=True,status='DESKTOP_ONLY_MEASURED',protocol_sha256=sha(DOC/'RUNTIME_PROTOCOL.json'),before=status,after=gpu(),summary=summary,records=records,memory=memory,isolated_memory=isolated_memory,params=params,
        accuracy_parity=True,canonical_MAIN_pose_parity=True,all_models_resident_for_balanced_order=True,memory_scope='allocated CUDA bytes; incremental workspace, not independent process total VRAM',
        pnp_scope='prediction-only canonical selector then SQPnP+RefineLM for selected camera-facing cuboid; no GT, IoU or error metric inside timer',
        limitations='Single desktop GPU, display/RustDesk active; 5 balanced cyclic blocks are not every possible model order; no Jetson measurement; no fastest-run selection'))
    print('RUNTIME_PANEL_COMPLETE',flush=True)
if __name__=='__main__':run()
