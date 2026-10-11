"""One predeclared 900-call deployment benchmark with immutable seed1 weights.

Decode and model loading occur before timing. Every timed interval includes the
actual detector, the selected correction, and the unchanged production F.
Accuracy/scoring and receipt writing happen after GPU-synchronized intervals.
"""
from collections import Counter
import copy
import importlib
import platform
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from . import common as C

ARMS=('BASE','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','POSEFIX','POSEFIX_SUBPIX')
WARMUP, REPEATS, FRAMES=20,5,26

def schedules():
    out=[]
    for phase, blocks in [('warmup',WARMUP),('measured',REPEATS)]:
        for block in range(blocks):
            order=list(ARMS[block%len(ARMS):]+ARMS[:block%len(ARMS)])
            if block%2:order.reverse()
            indices=[block%FRAMES] if phase=='warmup' else range(FRAMES)
            for index in indices:
                for pos,arm in enumerate(order):
                    out.append(dict(phase=phase,block=block,image_index=index,arm_position=pos,arm=arm))
    assert Counter(r['arm'] for r in out if r['phase']=='warmup')=={a:20 for a in ARMS}
    assert Counter(r['arm'] for r in out if r['phase']=='measured')=={a:130 for a in ARMS}
    assert len(out)==900
    return out

def run():
    import torch
    C.method_lock();C.verify_inputs()
    assert C.read(C.DOC/'EVALUATION_EXECUTION.json')['status']=='COMPLETE'
    assert C.read(C.DOC/'POSEFIX_319_PARITY.json')['status']=='PASS'
    assert not (C.DOC/'LATENCY.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True  # unchanged original YOLO/N3 runtime contract
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    assert torch.cuda.is_available()
    C.pose_api()
    from scripts.research.pallet_n3_subpix_20261008_v1 import runtime as R
    from scripts.research.pallet_training_free_compare_20261007_v1 import methods as M
    # These are existing inference-only components; no old runner is executed.
    R.C.ROOT=C.SOURCE
    selected,images,expected_base,panel_bindings=R._panel()
    expected={(r['method'],r['id']):r for r in C.rows(C.DOC/'RECT_ROWS.jsonl.gz')
              if r['backbone']=='YOLO' and r['seed']==1 and r['method'] in ARMS}
    assert len(expected)==319*6
    for row in selected:
        assert all((a,row['frame_id']) in expected for a in ARMS)
    source=C.SOURCE/'scripts/research/pallet_sensors_submission_v1'
    import sys
    sys.path.insert(0,str(source))
    prior_model=importlib.import_module('prior_model')
    crop_math=importlib.import_module('posefix_contract_math')
    checkpoint=C.SOURCE/'data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR1/last.pt'
    selection=C.read(C.SOURCE/'_docs/experiments/pallet_sensors_submission_v1/PRIOR_SELECTION.json')
    assert C.sha(checkpoint)==selection['checkpoints']['1']
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['complete'] and ck['step']==6000
    prior=prior_model.PoseFixPallet9().cuda().eval().requires_grad_(False)
    prior.load_state_dict(ck['model_state_dict']);del ck
    assert sum(p.numel() for p in prior.parameters())==68661641
    pipeline=R.Pipeline()
    schedule=schedules()
    bindings=[C.binding(Path(__file__)),C.binding(Path(R.__file__)),C.binding(checkpoint),
              C.binding(Path(prior_model.__file__)),C.binding(Path(crop_math.__file__))]
    # Historical inference sources, fixed panel, and checkpoints are hashed before
    # timing; benchmark coordinates are not used to select any method or setting.
    C.write('LATENCY_SOURCE_LOCK.json',dict(status='LOCKED_BEFORE_900_ACTUAL_CALLS',bindings=bindings,
        panel_bindings=panel_bindings,pipeline_bindings=pipeline.bindings,
        coordinate_rows_sha256=C.sha(C.DOC/'RECT_ROWS.jsonl.gz'),
        method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'),schedule=schedule,
        arms=ARMS,warmup_per_arm=20,measured_per_arm=130,total_calls=900,
        posefix_checkpoint='canonical PRIOR1 last6000',posefix_contract='one RAW pass, uncapped; +SubPix final Base-anchored cap once'))
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,temperature.gpu,memory.used','--format=csv,noheader'],text=True).strip()
    interference=R._interference()
    assert interference['quiet'] and interference['gpu_temperature_under_80'],('Fixed runtime interference guard',interference)
    calls=Counter();pnp=Counter();original={n:getattr(cv2,n) for n in ('solvePnP','solvePnPGeneric','solvePnPRefineLM')}
    def wrapped(name):
        def fn(*args,**kwargs):pnp[name]+=1;return original[name](*args,**kwargs)
        return fn
    for name in original:setattr(cv2,name,wrapped(name))
    receipts=[];max_points={a:0. for a in ARMS};max_pose={a:0. for a in ARMS}
    journal=C.DOC/'LATENCY_CALLS.jsonl'
    assert not journal.exists()
    try:
        with journal.open('x') as stream,torch.no_grad():
            for ordinal,item in enumerate(schedule):
                arm=item['arm'];row=selected[item['image_index']];image=images[item['image_index']]
                before=pnp.copy();d0=pipeline.detector_forwards;n0=pipeline.n3_forwards
                torch.cuda.synchronize();start=time.perf_counter_ns()
                captured=pipeline.extractor.predict(image)
                if arm in ('POSEFIX','POSEFIX_SUBPIX'):
                    raw=dict(candidates=pipeline.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
                    p0,support=R._raw(raw);final=copy.deepcopy(raw);diagnostic=None;usable=False;pf_raw=None;pf_atol=0.
                    if p0 is not None:
                        candidate=raw['candidates'][raw['selected_index']]
                        box=np.asarray(candidate['box_xyxy'],float)
                        usable=np.isfinite(box).all() and (box[2:]>box[:2]).all()
                        q=p0.copy()
                        if usable:
                            matrix=crop_math.axis_aligned_crop_matrix(box)
                            crop=cv2.warpAffine(image,matrix[:2],(288,384),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                            rgb=(crop[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)).transpose(2,0,1)
                            crop_points=crop_math.transform_points(np.where(support[:,None],p0,0),matrix).astype(np.float32)
                            args=[torch.as_tensor(rgb,device='cuda')[None],torch.as_tensor(crop_points,device='cuda')[None],torch.as_tensor(support,device='cuda')[None]]
                            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):logits=prior(*args)
                            heatmap_points=prior_model.expectation(logits)[0].cpu().numpy()
                            q=crop_math.transform_points(heatmap_points,np.linalg.inv(matrix));q[~support]=p0[~support];q[8]=p0[8]
                        pf_raw=q.copy()
                        # Inherited replay tolerance is in crop components. Its
                        # exact affine inverse converts that guard to raw pixels.
                        pf_atol=3e-4*float(np.abs(np.linalg.inv(matrix)[:2,:2]).sum(1).max()) if usable else 0.
                        if arm=='POSEFIX_SUBPIX':
                            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                            native,diagnostic=M.correct(gray,q,support,'SUBPIX')
                            q=M.cap_points(p0,native,image.shape[1],image.shape[0],support)
                        final['candidates'][final['selected_index']]['keypoints_xy']=q
                    calls['posefix_forward']+=int(p0 is not None and usable)
                else:
                    oldarm={'N3_DIM_SYM':'N3','N3_THEN_SUBPIX':'N3_SUBPIX'}.get(arm,arm)
                    final,diagnostic,_=pipeline.correct(oldarm,image,captured,row)
                q,support=R._raw(final)
                actual=pipeline.pose.infer(q,np.asarray(row['camera_intrinsics'],float),
                    np.asarray(row['dimensions_wdh_m'],float)[[0,2,1]],False)
                torch.cuda.synchronize();end=time.perf_counter_ns()
                # All numeric checks and output writes are outside the timed span.
                baseline=dict(candidates=pipeline.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
                R._check_raw(baseline,expected_base[item['image_index']])
                ref=expected[(arm,row['frame_id'])]
                pf_parity=None
                if arm.startswith('POSEFIX'):
                    pf_parity=R._point_parity(pf_raw,expected[('POSEFIX',row['frame_id'])]['qFinal'],pf_atol,(arm,'inherited crop tolerance'))
                tolerance=pf_atol if arm=='POSEFIX' else 1e-4 if arm=='N3_DIM_SYM' else 1e-7
                if arm=='POSEFIX_SUBPIX':
                    # cornerSubPix may amplify small inherited FP32 differences;
                    # publish those differences without changing the algorithm.
                    finite=np.isfinite(np.asarray(ref['qFinal'],float))
                    assert np.array_equal(np.isfinite(q),finite)
                    pp=dict(max_abs_px=float(np.max(np.abs(q[finite]-np.asarray(ref['qFinal'],float)[finite]))),
                        atol_px=None,rtol=0,finite_support_equal=True,status='RECORDED_RAW_PF_NUMERIC_VARIATION_AFTER_SUBPIX')
                else:pp=R._point_parity(q,ref['qFinal'],tolerance,(arm,row['frame_id']))
                pose_ref=ref['actual_pose']
                assert actual['available']==pose_ref['available'] and actual.get('selected_hypothesis')==pose_ref.get('selected_hypothesis')
                pose_delta=0.
                if actual['available']:
                    # PF FP32 repeat variation is checked in native pixels above;
                    # saved-vs-runtime pose geometry is recorded without retuning.
                    pose_delta=max(float(np.max(np.abs(np.asarray(actual[k],float)-np.asarray(pose_ref[k],float)))) for k in ('R_cf','centroid','cf_extents'))
                max_points[arm]=max(max_points[arm],pp['max_abs_px']);max_pose[arm]=max(max_pose[arm],pose_delta)
                detector_hooks=pipeline.detector_forwards-d0
                # Original Ultralytics owns one internal initialization forward
                # in the first preparation call. Explicit image calls remain1.
                assert detector_hooks==(2 if ordinal==0 else 1),('Unexpected detector hook count',ordinal,detector_hooks)
                record=dict(ordinal=ordinal,**item,id=row['frame_id'],session=row['session_id'],wall_ms=(end-start)/1e6,
                    detector_forwards=detector_hooks,explicit_detector_calls=1,
                    internal_initialization_forwards=detector_hooks-1,n3_forwards=pipeline.n3_forwards-n0,
                    actual_F_calls=1,PnP_calls={n:pnp[n]-before[n] for n in original},point_parity=pp,posefix_raw_parity=pf_parity,
                    pose_max_component_delta=pose_delta,pose_available=actual['available'],
                    subpix_corner_calls=(diagnostic or {}).get('algorithm_corner_calls',0))
                stream.write(__import__('json').dumps(C.finite(record),allow_nan=False)+'\n');stream.flush()
                receipts.append(record);calls[arm]+=1
                if (ordinal+1)%150==0:print('ACTUAL_FULL_PIPELINE_LATENCY',ordinal+1,900,flush=True)
    finally:
        for name,fn in original.items():setattr(cv2,name,fn)
        pipeline.close()
    assert len(receipts)==900 and pipeline.detector_forwards==901
    arms={}
    for arm in ARMS:
        v=np.asarray([r['wall_ms'] for r in receipts if r['phase']=='measured' and r['arm']==arm])
        assert len(v)==130
        arms[arm]=dict(n=130,mean_ms=float(v.mean()),sd_ms=float(v.std(ddof=1)),variance_ms2=float(v.var(ddof=1)),
            median_ms=float(np.median(v)),P90_ms=float(np.quantile(v,.9)),warmup_excluded=20,
            max_saved_coordinate_delta_px=max_points[arm],max_saved_pose_component_delta=max_pose[arm])
    for b in bindings:C.verify_binding(b)
    C.verify_inputs()
    C.write('LATENCY.json',dict(status='COMPLETE',arms=arms,hardware=dict(gpu_before=gpu_before,
        gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,temperature.gpu,memory.used','--format=csv,noheader'],text=True).strip(),
        python=platform.python_version(),python_executable=__import__('sys').executable,
        torch=torch.__version__,opencv=cv2.__version__,CUDA=torch.version.cuda,
        ultralytics=__import__('ultralytics').__version__,interference_before=interference,
        matmul_tf32=False,detector_N3_cudnn_tf32=True,posefix_cudnn_tf32=False),
        protocol=dict(batch=1,warmup_per_arm=20,frames=26,sessions=13,repeats=5,measured_calls_per_arm=130,
            actual_pipeline_calls=900,explicit_detector_calls=900,detector_forwards=901,
            detector_internal_initialization_forwards=1,actual_F_calls=900,PnP_calls=dict(pnp),
            includes='preprocessing + actual YOLO forward + correction + final F; GPU-synchronized wall time',
            excludes='decode, checkpoint loading, GT scoring, parity checking, receipt writing',
            arm_order='predetermined cyclic rotation; odd blocks reversed; fixed seed1'),
        actual_component_calls=dict(calls),raw_receipts_sha256=C.sha(journal),raw_receipts_path=journal.name,
        source_lock_sha256=C.sha(C.DOC/'LATENCY_SOURCE_LOCK.json'),new_training_updates=0,
        fixed_accuracy_predictions_unchanged=True,old_runtime_numbers_not_reused=True))
    print('SIX_ARM_FULL_PIPELINE_LATENCY_COMPLETE',arms,flush=True)

if __name__=='__main__':run()
