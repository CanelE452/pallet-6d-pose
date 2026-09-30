"""CPU fixed-checkpoint inference and controlled paired masks. No training."""
import argparse
import copy
import gc
import time
import resource
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from . import run as C
from scripts.research.pallet_posefix_large_error_v1 import core as CORE
from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import cover, overlap, fill

def load_refiner(arm):
    if arm=='PRIOR1':return C.L.L.load_base().eval().requires_grad_(False)
    fit=C.read(C.L.DOC/'FIT_FULL.json');C.verify(fit['checkpoint'])
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    model=C.L.model('FULL','cpu');model.load_state_dict(ck['model_state_dict'],strict=True)
    assert C.L.state_hash(model.state_dict())==fit['final_state_sha']
    return model.eval().requires_grad_(False)

@torch.no_grad()
def detector(model,image):
    # Existing native recipe, only device changes explicitly to CPU FP32.
    canvas=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101)
    p=model.predict(canvas,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cpu',verbose=False,save=False,stream=False)[0]
    out=[]
    if p.boxes is not None:
        for j in range(len(p.boxes)):
            out.append(dict(candidate_index=j,score=float(p.boxes.conf[j]),box_xyxy=(p.boxes.xyxy[j].cpu().numpy()-100).tolist(),
                keypoints_xy=(p.keypoints.xy[j].cpu().numpy()-100).tolist(),keypoints_conf=p.keypoints.conf[j].cpu().numpy().tolist()))
    return dict(candidates=out,selected_index=int(np.argmax([r['score'] for r in out])) if out else None)

def mask_plan(q,box,hw,seed):
    """Same existing mask size/aspect/fill family, fixed paired placement.

    Forced application replaces training's p=.5 scheduling. First valid of
    4096 seed-fixed positions is used independently for cover>=1/leave>=2 and
    avoid-all. Both must overlap the same predicted object box. No output or
    error is used for placement and no size retry is allowed.
    """
    rng=np.random.default_rng(seed);p=np.asarray(q);valid=np.isfinite(p).all(1)&~(p==-1).all(1);valid[8]=False
    ratio=float(rng.choice([.5,1.,2.]));fraction=float(rng.choice([.1,.2,.3]));area=float(np.prod(np.asarray(box)[2:]-box[:2]))
    w=max(1,round(np.sqrt(area*fraction*ratio)));h=max(1,round(np.sqrt(area*fraction/ratio)))
    result=dict(seed=int(seed),size=[w,h],aspect=ratio,area_fraction=fraction,fill_seed=int(rng.integers(0,2**31-1)),rectangles={},status='BLOCKED_PLACEMENT')
    if w>hw[1] or h>hw[0]:return result
    for _ in range(4096):
        rect=[int(rng.integers(0,hw[1]-w+1)),int(rng.integers(0,hw[0]-h+1)),w,h]
        n=int((cover(p,rect)&valid).sum())
        if overlap(rect,box)<=0:continue
        kind='cover' if n>=1 and valid.sum()-n>=2 else 'avoid' if n==0 else None
        if kind and kind not in result['rectangles']:
            result['rectangles'][kind]=dict(rectangle=rect,covered_qC=np.flatnonzero(cover(p,rect)&valid).tolist(),bbox_overlap=overlap(rect,box)/area)
        if len(result['rectangles'])==2:break
    if len(result['rectangles'])==2:result['status']='READY'
    return result

def masked(image,plan,kind):
    out=image.copy();l,t,w,h=plan['rectangles'][kind]['rectangle']
    out[t:t+h,l:l+w]=fill(plan).transpose(1,2,0)
    return out

def count_save(stage,start,cpu,count,**kw):
    C.save(C.RAW/f'COST_{stage}.json',dict(stage=stage,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        image_forwards=count,GPU_seconds=0,peak_GPU_memory_bytes=0,peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=0,optimizer_updates=0,device='CPU',**kw))

def prepare_e3():
    start,cpu=time.monotonic(),time.process_time();rows=[r for r in C.metadata() if r['severity']=='CLEAN']
    raw=C.read(C.RAW/'INPUT_PREDICTIONS.json')['identity'];rng=np.random.default_rng(42)
    plans={r['id']:mask_plan(CORE.selected(raw[r['id']])['keypoints_xy'],np.array(CORE.selected(raw[r['id']])['box_xyxy']),r['hw'],int(rng.integers(0,2**31-1))) for r in rows}
    C.save(C.RAW/'E3_MASK_PLANS.json',plans)
    C.save(C.DOC/'E3_PROTOCOL.json',dict(locked_before_new_forward=True,seed=42,mask_family='Existing Clean19 rectangle size/aspect/bilinear8x8-noise fill',
        placement='First valid of4096 fixed random geometric positions; pair same size/fill; cover>=1 and leave>=2 qC corner locations vs avoid all valid qC corner locations; both overlap box',
        changes_from_training_rule='Forced application and cover/avoid paired placement; no size resampling',
        source_code=C.bind(__file__),plan=C.bind(C.RAW/'E3_MASK_PLANS.json'),sample_size=29,independent_samples=29,
        point_visibility='Validity is not visibility; named keypoint-location occlusion',
        controls='Same clean selected bbox/score/confidence/candidate identity; only coordinates and RGB crossed. qO must native-match clean box IoU>=.5, else paired controlled comparison BLOCKED and native failures retained.',
        qC='Frozen clean R0 input, CPU smoke parity required',qO='Same R0 checkpoint CPU native masked prediction',
        CPU_parity_tolerance_original_pixels=.01,nativeOO=True,CC_reused_after_smoke=True,no_reference_in_model=True))
    count_save('E3_prepare',start,cpu,0)
    print('E3_PLAN', {i:p['status'] for i,p in plans.items() if p['status']!='READY'},flush=True)

def r0_e3():
    start,cpu=time.monotonic(),time.process_time();count=0
    rows=[r for r in C.metadata() if r['severity']=='CLEAN'];raw=C.read(C.RAW/'INPUT_PREDICTIONS.json')['identity'];plans=C.read(C.RAW/'E3_MASK_PLANS.json')
    b=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['R0_provenance']['baseline'];C.verify(b)
    model=YOLO(str(C.ROOT/b['path']),task='pose');output={};smoke={}
    r=rows[0];im=cv2.imread(str(C.ROOT/r['image']['path']));p=detector(model,im);count+=1
    a,z=CORE.selected(raw[r['id']]),CORE.selected(p)
    delta=float(np.max(np.abs(np.array(a['keypoints_xy'])-z['keypoints_xy'])))
    smoke=dict(id=r['id'],max_xy_difference_px=delta,passed=delta<=.01,CPU_prediction=p)
    C.save(C.RAW/'E3_R0_SMOKE.json',smoke)
    if not smoke['passed']:
        count_save('E3_R0',start,cpu,count,status='BLOCKED_PARITY');return
    for r in rows:
        i=r['id'];plan=plans[i];output[i]={}
        if plan['status']!='READY':continue
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']))
        for kind in ('cover','avoid'):
            occ=masked(im,plan,kind);output[i][kind]=detector(model,occ);count+=1
        print('R0_MASKED',i,flush=True)
    C.save(C.RAW/'E3_R0_PREDICTIONS.json',output)
    count_save('E3_R0',start,cpu,count,checkpoint=b,smoke_forwards=1,clean_qC_reused=29)

def r0_cpu_consistent():
    """Resolve GPU-cache mismatch by recomputing both sides on the same CPU.

    The original failed cache-reuse smoke is retained and reused as clean
    frame0. No tolerance is relaxed and no original file is overwritten.
    """
    start,cpu=time.monotonic(),time.process_time();count=0
    rows=[r for r in C.metadata() if r['severity']=='CLEAN'];plans=C.read(C.RAW/'E3_MASK_PLANS.json')
    b=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['R0_provenance']['baseline'];C.verify(b)
    smoke=C.read(C.RAW/'E3_R0_SMOKE.json');assert not smoke['passed']
    C.save(C.DOC/'E3_CPU_CONSISTENCY_ADDENDUM.json',dict(reason='CPU R0 vs historic GPU max coordinate delta exceeds locked .01px cache-reuse tolerance',
        smoke=C.bind(C.RAW/'E3_R0_SMOKE.json'),action='Recompute clean and masked R0 and clean refiner outputs on CPU; no mixed-device cache reuse for E3.',
        mask_plan_unchanged=True,forward_budget=725,no_new_model_or_parameters=True,first_clean_forward_reused=True,
        mask_locations='Frozen original clean R0 geometry; actual CPU qC/reference coverage also reported. No performance-based remasking.'))
    model=YOLO(str(C.ROOT/b['path']),task='pose');out={};clean={}
    for ix,r in enumerate(rows):
        i=r['id'];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']))
        clean[i]=smoke['CPU_prediction'] if ix==0 else detector(model,im);count+=int(ix!=0)
        out[i]={}
        if plans[i]['status']=='READY':
            for kind in ('cover','avoid'):out[i][kind]=detector(model,masked(im,plans[i],kind));count+=1
        print('CPU_CONSISTENT_R0',ix+1,flush=True)
    C.save(C.RAW/'E3_CPU_QC.json',clean);C.save(C.RAW/'E3_R0_PREDICTIONS.json',out)
    count_save('E3_R0_consistent',start,cpu,count,checkpoint=b,smoke_reused=1)

def refine_e3(arm):
    start,cpu=time.monotonic(),time.process_time();count=0
    rows=[r for r in C.metadata() if r['severity']=='CLEAN'];base=C.read(C.RAW/'INPUT_PREDICTIONS.json');raw=C.read(C.RAW/'E3_CPU_QC.json');plans=C.read(C.RAW/'E3_MASK_PLANS.json')
    occpred=C.read(C.RAW/'E3_R0_PREDICTIONS.json');model=load_refiner(arm);state=C.L.state_hash(model.state_dict())
    from scripts.research.pallet_posefix_large_error_v1.evaluate import iou
    r=rows[0];im=cv2.imread(str(C.ROOT/r['image']['path']));p=CORE.predict(model,im,raw[r['id']]);count+=1
    a,z=CORE.selected(base[arm][r['id']]),CORE.selected(p)
    delta=float(np.max(np.abs(np.array(a['keypoints_xy'])-z['keypoints_xy'])))
    C.L.E.assert_preserved(raw[r['id']],p)
    # Transform round trip is exact; old GPU cache is descriptive only after
    # the CPU consistency addendum. Reuse this smoke as the first CPU CC.
    inp=CORE.prepare_input(im,raw[r['id']]);q=np.array(CORE.selected(raw[r['id']])['keypoints_xy'])
    restored=CORE.transform_points(CORE.transform_points(q,inp['matrix']),np.linalg.inv(inp['matrix']))
    np.testing.assert_allclose(restored,q,atol=1e-9,rtol=0)
    C.save(C.RAW/f'E3_{arm}_SMOKE.json',dict(id=r['id'],max_xy_difference_to_GPU_cache_px=delta,transform_roundtrip_passed=True,CPU_prediction=p,cache_reused=False))
    first_cc=p
    for ix,r in enumerate(rows):
        i=r['id'];dest=C.RAW/'e3_frames'/arm/f'{ix:03d}.json'
        assert not dest.exists();output={}
        im=cv2.imread(str(C.ROOT/r['image']['path']));qC=raw[i]
        clean_output=first_cc if ix==0 else CORE.predict(model,im,qC);count+=int(ix!=0)
        for kind,native in occpred[i].items():
            occ=masked(im,plans[i],kind);nc=CORE.selected(native);cc=CORE.selected(qC)
            same=nc is not None and iou(cc['box_xyxy'],nc['box_xyxy'])>=.5
            results={'CC':clean_output}
            # OC is always defined with clean coordinates/box, regardless of qO.
            results['OC']=CORE.predict(model,occ,qC);count+=1
            if same:
                qO=copy.deepcopy(qC);CORE.selected(qO)['keypoints_xy']=copy.deepcopy(nc['keypoints_xy'])
                a1=CORE.prepare_input(im,qC);a2=CORE.prepare_input(occ,qO)
                np.testing.assert_array_equal(a1['matrix'],a2['matrix']);np.testing.assert_array_equal(a1['box'],a2['box'])
                for cond,img in [('CO',im),('OO',occ)]:results[cond]=CORE.predict(model,img,qO);count+=1
            results['nativeOO']=CORE.predict(model,occ,native);count+=int(CORE.prepare_input(occ,native) is not None)
            output[kind]=dict(predictions=results,same_object_pair=same,pair_box_iou=iou(cc['box_xyxy'],nc['box_xyxy']) if nc else None)
        C.save(dest,dict(id=i,arm=arm,conditions=output));print('REFINE',arm,ix+1,'forwards',count,flush=True)
    assert C.L.state_hash(model.state_dict())==state
    count_save('E3_'+arm,start,cpu,count,CC_cross_mask_reused_frames=29,smoke_forwards=1,smoke_reused_as_CC=True,checkpoint_state_sha=state,BN_unchanged=True)

def e6():
    start,cpu=time.monotonic(),time.process_time();report=C.read(C.ROOT/'data/pallet/results/paper_eval_v1/arms/REALFT_A.json')
    b=dict(path=report['weights']['resolved_path'],sha256=report['weights']['sha256']);C.verify(b)
    model=YOLO(str(C.ROOT/b['path']),task='pose');out={}
    for j,r in enumerate(C.metadata()):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));out[r['id']]=detector(model,im)
        if (j+1)%16==0:print('REALFT_A',j+1,flush=True)
    C.save(C.RAW/'E6_PREDICTIONS.json',out);count_save('E6_infer',start,cpu,len(out),checkpoint=b)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','r0','r0_cpu','PRIOR1','FULL125','e6']);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    {'prepare':prepare_e3,'r0':r0_e3,'r0_cpu':r0_cpu_consistent,'PRIOR1':lambda:refine_e3('PRIOR1'),'FULL125':lambda:refine_e3('FULL125'),'e6':e6}[a.stage]()

if __name__=='__main__':main()
