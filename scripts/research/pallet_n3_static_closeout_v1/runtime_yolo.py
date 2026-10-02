"""Fixed seed1 YOLO runtime, existing 26-frame protocol, no learning."""
import time,sys,platform
import cv2,numpy as np,torch
from .compute import C,DOC,RAW,SOURCE,write,bind
from scripts.research.pallet_n3_completion_v3 import runtime as RT,square_yolo as Y,metrics as M

def main():
    target=DOC/'RUNTIME_YOLO_SEED1.json'
    if target.exists():
        d=C.read(target);assert C.sha256(RAW/'RUNTIME_YOLO_RAW.json')==d['raw']['sha256'];print('REUSED');return
    started=time.time();gpu=C.gpu_snapshot(refuse_other_compute=True)
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(1);np.random.seed(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=C.read(C.RAW/'runtime/dope_seed1.json'); selected=old['selected']
    lock=C.read(RT.LEGACY_RUNTIME_PROTOCOL)
    assert C.sha256(RT.LEGACY_RUNTIME_PROTOCOL)==RT.LEGACY_RUNTIME_PROTOCOL_SHA256
    assert [r['image_key'] for r in selected]==lock['keys'] and len(selected)==26
    images=[]
    for r in selected:
        p=SOURCE/r['image_key'];assert C.sha256(p)==r['image']['sha256']
        images.append(cv2.imread(str(p)))
    env,inf=Y._runtime_modules(); spec=Y.selection_contract(verify_checkpoints=True)['methods']['N3_DIM_SYM_seed1']
    head,checkpoint=inf.load_head('N3_DIM_SYM',1);extractor=env.old('features').FrozenYoloFeatures(env.R0)
    pose=M._pose_contract();norm=C.read(Y.NORMALIZATION);dev=torch.device('cuda')
    hardware=RT._hardware(dev)
    assert hardware['uuid']==old['hardware']['uuid'],'Must use same physical GPU'
    def operation(path,i,cached=None):
        row=selected[i];cap=extractor.predict(images[i]) if cached is None else cached
        if path=='base_e2e':pred={'candidates':inf.serial(cap['candidates']),'selected_index':cap['selected_index']}
        else:
            dimensions,order=inf.registry_input(row['object_type']);assert dimensions.tolist()==row['dimensions_wdh_m']
            pred,_=inf.predict_captured(head,'N3_DIM_SYM',cap,dimensions,order,spec['temperature'],spec['rule'],images[i].shape[:2],norm)
        index=pred['selected_index'];points=None if index is None else np.asarray(pred['candidates'][index]['keypoints_xy'])
        result={'selected_index':index,'point_sha256':None if points is None else RT._point_digest(points)}
        if path!='n3_seed1_only':
            p=pose.infer(points,np.array(row['camera_intrinsics']),np.array(row['dimensions_wdh_m'])[[0,2,1]],source=False)
            result['pose_available']=bool(p['available'])
        return result
    def one(job):
        i=job['image_index'];path=job['path'];cached=extractor.predict(images[i]) if path=='n3_seed1_only' else None
        result,timing=RT._timed_call(lambda:operation(path,i,cached),dev)
        return {**job,'frame_id':selected[i]['frame_id'],'timing':timing,'output':result}
    warm=[one(j) for j in RT.warmup_schedule()]
    measured=[]
    for j in RT.schedule():
        measured.append(one(j))
        if len(measured)%78==0:print('Measured',len(measured),'/390',flush=True)
    summary={}
    for path in RT.PATHS:
        values=[r['timing'] for r in measured if r['path']==path]
        summary[path]={k:RT.describe_ms([r[k] for r in values]) for k in ['cuda_event_ms','wall_ms']}
        summary[path]['peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in values)
    import ultralytics
    raw={'complete':True,'config':old['config'],'hardware':hardware,'python':sys.version,'ultralytics':ultralytics.__version__,
        'selected':selected,'warmup':warm,'measurements':measured,'started_gpu':gpu,'finished_gpu':C.gpu_snapshot(refuse_other_compute=True),
        'parameters':{'base_total':sum(p.numel() for p in extractor.yolo.model.parameters()),'n3_total':sum(p.numel() for p in head.parameters())},
        'precision':{'base':'FP32','head':'FP32 with FP16 feature handoff','AMP':False,'TF32':False},
        'timing_scope':old['timing_scope'],'scope_note':'YOLO base uses frozen feature extractor with feature capture/cast/pad; no head in base. Different environment from DOPE/ResNet; separate panel.',
        'bindings':[bind(checkpoint),bind(Y.R0_WEIGHTS),bind(Y.DCP_SELECTION),bind(RT.LEGACY_RUNTIME_PROTOCOL)],'elapsed_seconds':time.time()-started,
        'training_runs':0,'optimizer_updates':0,'new_prediction_selection':False}
    write(RAW/'RUNTIME_YOLO_RAW.json',raw)
    write(target,{'complete':True,'raw':bind(RAW/'RUNTIME_YOLO_RAW.json'),'summary':summary,'hardware':hardware,'parameters':raw['parameters'],'elapsed_seconds':raw['elapsed_seconds'],'environment_panel':'pallet-yolo26; differs from pallet-pose'})
    extractor.close();print(summary,flush=True)

if __name__=='__main__':main()
