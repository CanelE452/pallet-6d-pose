"""All fixed public R2B frames; frozen proposal measurement only, no GT/PnP."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime/Ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib')
import numpy as np
import cv2,torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def write(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def state_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()
def serialize(z):
    return {'candidates':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in z['candidates']],
            'selected_index':z['selected_index'],'no_detection':z['selected_index'] is None,
            'input_shape':list(z['input_shape']),'canvas_shape':list(z['canvas_shape']),'added_border':z['added_border']}

def main():
    entry=json.loads((ROOT/'RUN_ENTRYLOCK.json').read_text());protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    for b in entry['bindings']:assert sha(b['path'])==b['sha256'],b
    assert not (ROOT/'FROZEN_PROPOSALS.jsonl').exists()
    rows=[json.loads(x) for x in (ROOT/'INPUTS_RGB_ONLY.jsonl').read_text().splitlines()]
    assert len(rows)==48
    sys.path.insert(0,'/tmp/pallet-bi-full-execution-20261019-v1')
    from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    weights=protocol['initializer_model'];assert sha(weights['path'])==weights['sha256']
    model=FrozenYoloFeatures(weights['path'],device='cuda:0')
    loaded_digest=state_digest(model.yolo.model)
    # Standard Ultralytics inference preparation, before the frozen-forward state snapshot.
    # AutoBackend performs this same initialization; no data or optimizer is used here.
    model.yolo.model.fuse(verbose=False)
    model.yolo.model.requires_grad_(False).eval()
    frozen_digest=state_digest(model.yolo.model)
    forward_calls=[0]
    handle=model.yolo.model.register_forward_pre_hook(lambda *_:forward_calls.__setitem__(0,forward_calls[0]+1))
    write(ROOT/'RUN_STARTED.json',{'entry':bind(ROOT/'RUN_ENTRYLOCK.json'),'loaded_before_standard_fusion_state_digest':loaded_digest,'frozen_before_forward_state_digest':frozen_digest,'inference_initialization_changed_state':loaded_digest!=frozen_digest,'initialization_is_not_learning':True,'optimizer_updates':0,'PnP':0})
    outputs=[];features=[];start=time.monotonic();first=None;firstp3=None
    for i,row in enumerate(rows):
        assert sha(row['image'])==row['image_sha256']
        bgr=cv2.imread(row['image']);assert bgr is not None and list(bgr.shape[:2])==row['raw_hw']
        assert hashlib.sha256(np.ascontiguousarray(bgr[:,:,::-1]).tobytes()).hexdigest()==row['RGB_pixels_sha256']
        before=forward_calls[0];z=model.predict(bgr,already_padded=False,cpu_features=True)
        out=dict(input_id=row['id'],image_sha256=row['image_sha256'],**serialize(z),forward_hooks=forward_calls[0]-before)
        p3=z['p3'].numpy();assert p3.shape==(64,80,80) and p3.dtype==np.float16 and np.isfinite(p3).all()
        if i==0:first=serialize(z);firstp3=p3.copy()
        features.append(p3.copy());outputs.append(out)
        if (i+1)%16==0:print(json.dumps({'source_frames':i+1,'total':48,'seconds':time.monotonic()-start}),flush=True)
    before=forward_calls[0];repeat=model.predict(cv2.imread(rows[0]['image']),already_padded=False,cpu_features=True)
    prediction_repeat_equal=serialize(repeat)==first
    p3_repeat_equal=np.array_equal(repeat['p3'].numpy(),firstp3)
    after_digest=state_digest(model.yolo.model)
    assert after_digest==frozen_digest
    assert all(not m.training for m in model.yolo.model.modules())
    assert all(not p.requires_grad for p in model.yolo.model.parameters())
    assert sha(weights['path'])==weights['sha256']
    assert prediction_repeat_equal and p3_repeat_equal,'No source replay tolerance relaxation'
    assert sum(a.nbytes for a in features)<=100000000
    np.savez_compressed(ROOT/'P3_INPUT_FEATURES_ONLY.npz',p3=np.stack(features))
    (ROOT/'FROZEN_PROPOSALS.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in outputs))
    import ultralytics
    receipt={'schema':'V72_ACTUAL_PUBLIC_FROZEN_YOLO_INITIALIZER_RESULT','completed':True,'source_only':True,'frames':48,'all_candidates_retained':True,'selection':'first argmax confidence among returned unchanged postprocessing candidates','no_detection_frames':sum(x['no_detection'] for x in outputs),'candidate_count':sum(len(x['candidates']) for x in outputs),'explicit_frame_predict_calls':49,'main_frame_predict_calls':48,'repeat_first_predict_calls':1,'total_module_forward_hooks':forward_calls[0],'repeat_forward_hooks':forward_calls[0]-before,'inference_initialization_state_digest':frozen_digest,'after_forward_state_digest':after_digest,'loaded_before_standard_inference_fusion_digest':loaded_digest,'standard_fusion_before_forward_snapshot':True,'trained_parameters_and_registered_buffers_unchanged_during_all_forward_calls':True,'checkpoint_unchanged':True,'model_eval_and_no_grad':True,'prediction_first_repeat_bit_exact':prediction_repeat_equal,'P3_first_repeat_bit_exact':p3_repeat_equal,'P3_uncompressed_bytes':sum(a.nbytes for a in features),'P3_shape':[48,64,80,80],'seconds':time.monotonic()-start,'runtime':{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'device':torch.cuda.get_device_name(0),'resolved_predictor_args':vars(model.yolo.predictor.args),'CUBLAS_WORKSPACE_CONFIG':os.getenv('CUBLAS_WORKSPACE_CONFIG')},'source_family':'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED','split':'UNASSIGNED_GROUP_HOLD','semantic_GT_from_prediction':False,'current223_access':0,'optimizer_updates':0,'PnP':0,'outputs':[bind(ROOT/'FROZEN_PROPOSALS.jsonl'),bind(ROOT/'P3_INPUT_FEATURES_ONLY.npz')],'code':bind(Path(__file__)),'entry':bind(ROOT/'RUN_ENTRYLOCK.json')}
    handle.remove();model.close()
    write(ROOT/'INFERENCE_RESULT.json',receipt)
    print(json.dumps({'result':bind(ROOT/'INFERENCE_RESULT.json'),'frames':48,'no_detection':receipt['no_detection_frames'],'candidate_count':receipt['candidate_count']}),flush=True)

if __name__=='__main__':main()
