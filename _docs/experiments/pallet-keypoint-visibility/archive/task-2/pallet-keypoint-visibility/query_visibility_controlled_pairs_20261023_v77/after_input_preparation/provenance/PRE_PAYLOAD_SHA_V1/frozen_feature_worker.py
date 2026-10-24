"""Legacy frozen R0 worker. Framed binary output; never stores full P3.

Inputs are RGB plus unchanged input q for ROI sampling only. No GT/labels/mask
or detection-ranked query replacement is read or used.
"""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,struct,sys,time
HERE=Path(__file__).resolve().parent
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics')
os.environ['MPLCONFIGDIR']=str(HERE/'runtime/matplotlib')
WIRE=sys.stdout.buffer
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,d):
 assert not p.exists();p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def emit(meta,array=None):
 data=b''if array is None else array.tobytes(order='C')
 if array is not None:meta.update(shape=list(array.shape),dtype=str(array.dtype))
 meta['array_bytes']=len(data);h=json.dumps(meta,allow_nan=False,separators=(',',':')).encode();WIRE.write(struct.pack('<Q',len(h)));WIRE.write(h);WIRE.write(data);WIRE.flush()
def state_digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return h.hexdigest()
def main(args):
 protocol=json.loads(Path(args.protocol).read_text());assert sha(args.protocol)==args.protocol_sha
 assert protocol['worker_code']==bind(Path(__file__))
 for b in protocol['bindings']:assert sha(b['path'])==b['sha256'],b['path']
 assert str(Path(sys.executable).resolve())==protocol['legacy_python_resolved']
 import numpy as np,torch,cv2,ast
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(True)
 env={'np':np,'torch':torch,'F':torch.nn.functional}
 nodes=[n for n in ast.parse(Path(protocol['ROI_code']['path']).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in ('native_grid','patches')]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),protocol['ROI_code']['path'],'exec'),env)
 after=[json.loads(x)for x in Path(protocol['after_manifest']['path']).read_text().splitlines()]
 guards=json.loads(Path(protocol['guard_inputs']['path']).read_text());after_i=0;guard_i=0;explicit=0;repeats=0;decoded=0
 sys.path.insert(0,protocol['feature_package_root'])
 with contextlib.redirect_stdout(sys.stderr):
  from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures,canvas_affine
  net=FrozenYoloFeatures(protocol['weights']['path'],device='cuda:0');loaded=state_digest(net.yolo.model);net.yolo.model.fuse(verbose=False);net.yolo.model.requires_grad_(False).eval()
 before=state_digest(net.yolo.model);calls=[0];hook=net.yolo.model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1));torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 write(HERE/'WORKER_STARTED.json',dict(protocol=bind(Path(args.protocol)),loaded_state_digest=loaded,fused_before_state_digest=before,optimizer_updates=0))
 for line in sys.stdin:
  r=json.loads(line)
  if r.get('stop'):break
  if r['stage']=='guard':
   expected=guards[guard_i];guard_i+=1
  else:
   expected=after[after_i];after_i+=1
  assert r['id']==expected['id']and r['image']==expected['image']and r['image_sha256']==expected['image_sha256']
  assert sha(r['image'])==r['image_sha256']
  bgr=cv2.imread(r['image']);assert bgr is not None and list(bgr.shape[:2])==r['raw_hw']
  assert hashlib.sha256(np.ascontiguousarray(bgr[:,:,::-1]).tobytes()).hexdigest()==r['RGB_pixels_sha256'];decoded+=1
  with contextlib.redirect_stdout(sys.stderr):z=net.predict(bgr,already_padded=False,cpu_features=False)
  explicit+=1;gain,offset=canvas_affine(z['canvas_shape'],z['input_shape'])
  assert z['added_border']==100 and list(z['input_shape'])==r['expected_input_shape']and gain==r['expected_gain']and np.array_equal(offset,r['expected_offset'])
  q=np.asarray(r['initial_points'],np.float64)
  if r['backend']=='legacy_gpu':
   payload,valid,L=env['patches'](z,q,r['raw_hw'],gain,offset);a=payload.numpy();kind='ROI'
  else:a=z['p3'].cpu().numpy();valid=None;L=r['roi_side'];kind='P3'
  assert L==r['roi_side']and a.dtype==np.float16 and np.isfinite(a).all()
  repeat=bool(r['stage']=='guard'or after_i==1)
  if repeat:
   with contextlib.redirect_stdout(sys.stderr):zz=net.predict(bgr,already_padded=False,cpu_features=False)
   explicit+=1;repeats+=1;assert torch.equal(z['p3'],zz['p3'])
   if r['backend']=='legacy_gpu':assert np.array_equal(a,env['patches'](zz,q,r['raw_hw'],gain,offset)[0].numpy())
  emit(dict(kind=kind,id=r['id'],gain=gain,offset=offset.tolist(),input_shape=list(z['input_shape']),canvas_shape=list(z['canvas_shape']),added_border=100,
   ROI_valid=None if valid is None else valid.tolist(),repeat_bit_exact=True if repeat else None,P3_values_sha256=hashlib.sha256(z['p3'].cpu().numpy().tobytes()).hexdigest()),a)
 assert guard_i==2 and after_i==len(after)
 after_state=state_digest(net.yolo.model);assert after_state==before and sha(protocol['weights']['path'])==protocol['weights']['sha256']
 assert all(not m.training for m in net.yolo.model.modules())and all(not q.requires_grad for q in net.yolo.model.parameters())
 receipt=dict(completed=True,PASS=True,protocol=bind(Path(args.protocol)),code=bind(Path(__file__)),after_images=after_i,original_guard_images=guard_i,
  explicit_backbone_predict_calls=explicit,repeat_calls=repeats,module_forward_hooks=calls[0],decoded_images_with_canonical_pixel_SHA_check=decoded,
  source_specific_ROI_backend=True,full_P3_archives_stored=0,weights_unchanged=True,state_digest_before=before,state_digest_after=after_state,
  model_eval=True,requires_grad=False,peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,
  runtime=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cv2=cv2.__version__),GT_current_reads=0,fit_PnP=0,detection_candidates_used=False)
 write(HERE/'FROZEN_FEATURE_INFERENCE_RECEIPT.json',receipt);hook.remove();net.close();emit(dict(kind='complete',receipt=bind(HERE/'FROZEN_FEATURE_INFERENCE_RECEIPT.json')))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha',required=True);main(a.parse_args())
