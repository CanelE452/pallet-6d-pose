"""Two frozen source-image predicts; no query/GT selection or head forward."""
from pathlib import Path
import hashlib,json,os,sys,time
HERE=Path(__file__).resolve().parent
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics')
os.environ['MPLCONFIGDIR']=str(HERE/'runtime/matplotlib')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
 assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def state_digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return h.hexdigest()
def main():
 p=json.loads((HERE/'P3_REUSE_PROTOCOL.json').read_text())
 for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
 assert p['code']==bind(Path(__file__))and str(Path(sys.executable).resolve())==p['runtime_python_resolved']
 run=HERE/'P3_FIRST_REPLAY';run.mkdir(exist_ok=False)
 import numpy as np,cv2,torch
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(True)
 rows=[json.loads(l)for l in Path(p['image_manifest']['path']).read_text().splitlines()];assert len(rows)==1000
 first=rows[0];assert sha(first['image'])==first['image_sha256'];rgb=np.load(p['first_lossless_RGB']['path'],allow_pickle=False)
 assert hashlib.sha256(rgb.tobytes()).hexdigest()==first['RGB_pixels_sha256']and np.array_equal(cv2.imread(first['image'])[:,:,::-1],rgb)
 # Legacy PNG decoder must be exact before ANY image model forward.
 sys.path.insert(0,p['feature_package_root']);from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures,canvas_affine
 weights=p['weights'];assert sha(weights['path'])==weights['sha256'];net=FrozenYoloFeatures(weights['path'],device='cuda:0')
 loaded=state_digest(net.yolo.model);net.yolo.model.fuse(verbose=False);net.yolo.model.requires_grad_(False).eval();before=state_digest(net.yolo.model)
 calls=[0];hook=net.yolo.model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
 write(run/'RUN_STARTED.json',dict(protocol=bind(HERE/'P3_REUSE_PROTOCOL.json'),loaded_state_digest=loaded,standard_fused_before_image_state_digest=before,optimizer_updates=0))
 start=time.monotonic();bgr=np.ascontiguousarray(rgb[:,:,::-1]);z1=net.predict(bgr,already_padded=False,cpu_features=True);count1=calls[0]
 z2=net.predict(bgr,already_padded=False,cpu_features=True);gain,offset=canvas_affine(z1['canvas_shape'],z1['input_shape'])
 with np.load(first['P3_cache']['path'],allow_pickle=False)as z:old=z['p3'][0]
 new=z1['p3'].numpy();repeat=z2['p3'].numpy()
 equal_old=np.array_equal(new,old);equal_repeat=np.array_equal(new,repeat)
 assert z1['added_border']==100 and list(z1['input_shape'])==first['input_shape']and list(z1['canvas_shape'])==first['canvas_shape']
 assert gain==first['gain']and np.array_equal(offset,first['offset'])
 after=state_digest(net.yolo.model);assert before==after and sha(weights['path'])==weights['sha256']
 assert all(not m.training for m in net.yolo.model.modules())and all(not q.requires_grad for q in net.yolo.model.parameters())
 result=dict(completed=True,PASS=bool(equal_old and equal_repeat),protocol=bind(HERE/'P3_REUSE_PROTOCOL.json'),code=bind(Path(__file__)),image_manifest=p['image_manifest'],
  images_existing=1000,explicit_source_predict_calls=2,module_forward_hooks=calls[0],first_predict_hooks=count1,repeat_predict_hooks=calls[0]-count1,
  first_PNG_cv2_vs_Pillow12_decode_exact=True,first_P3_vs_existing_cache_bit_exact=equal_old,first_P3_repeat_bit_exact=equal_repeat,
  P3_max_abs_diff_existing=float(np.max(np.abs(new.astype(float)-old.astype(float)))),P3_differing_values_existing=int(np.sum(new!=old)),
  frozen_state_before=before,frozen_state_after=after,standard_inference_fusion_only=True,checkpoint_unchanged=True,model_eval=True,requires_grad=False,
  feature_affine=dict(added_border=100,input_shape=list(z1['input_shape']),canvas_shape=list(z1['canvas_shape']),gain=gain,offset=offset.tolist()),
  runtime=dict(python=sys.version,torch=str(torch.__version__),cv2=cv2.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0)),
  GT_queries_masks_reference_current_reads=0,optimizer_updates=0,PnP=0,detection_candidates_not_used_for_query_or_selection=True,seconds=time.monotonic()-start)
 write(run/'P3_REPLAY_CHECK.json',result);hook.remove();net.close()
 print(json.dumps(dict(result=bind(run/'P3_REPLAY_CHECK.json'),PASS=result['PASS'])),flush=True)
 assert result['PASS'],'Preserve failedprotocol/code/log; no tolerance relaxation/cache overwrite'
if __name__=='__main__':main()
