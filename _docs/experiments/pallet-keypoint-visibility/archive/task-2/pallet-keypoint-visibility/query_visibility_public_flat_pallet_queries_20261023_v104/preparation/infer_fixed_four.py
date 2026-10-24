"""Frozen public initializer only. Model construction requires exact root entry."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,time
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;OUT=V/'initializer_only_outputs';PROTOCOL=H/'INITIALIZER_PROTOCOL_BEFORE_FORWARD.json'
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,x):
 with Path(p).open('x')as f:
  for r in x:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
def native_grid():
 n=next(x for x in ast.parse(ROI.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='native_grid');ns={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),str(ROI),'exec'),ns);return ns['native_grid']
def serial(z):return dict(candidates=[{k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in c.items()}for c in z['candidates']],selected_index=z['selected_index'],input_shape=list(z['input_shape']),canvas_shape=list(z['canvas_shape']),added_border=z['added_border'])
def selected_queries(z,hw):
 selected=None if not z['candidates']else int(np.argmax([c['score']for c in z['candidates']]));assert selected==z['selected_index']
 if selected is None:return [dict(native_slot=k,query_xy=None,roi_side=None,original_query_continuous_valid=False,query_nearest_pixel_valid=False,query_rgb96_centre_domain_valid=False,original_input_valid=False,reason='NO_DETECTION_U')for k in range(8)]
 q=np.asarray(z['candidates'][selected]['keypoints_xy'],np.float64);assert q.shape==(9,2)and np.isfinite(q).all();_,_,L=native_grid()(q,hw);h,w=hw;out=[]
 for k,(x,y)in enumerate(q[:8]):
  cont=bool(0<=x<w and 0<=y<h);near=bool(0<=np.floor(x+.5)<w and 0<=np.floor(y+.5)<h);rgb=bool(-.5<=x<w-.5 and-.5<=y<h-.5)
  out.append(dict(native_slot=k,query_xy=[float(x),float(y)],roi_side=L,original_query_continuous_valid=cont,query_nearest_pixel_valid=near,query_rgb96_centre_domain_valid=rgb,original_input_valid=cont and near,reason='VALID_QUERY_NOT_VISIBILITY_OR_CORRECTNESS'if cont and near else'OOB_OR_MISS_U'))
 return out

def state(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return h.hexdigest()
def run(entry):
 e=read(entry);assert e['approved_initializer_only']and e['current_allowed']is False and e['fit_allowed']is False and e['PnP_allowed']is False
 for k in('code','protocol','generated','canonical_RGB_seal'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and not OUT.exists();p=read(PROTOCOL)
 for k in('source_inputs','native_grid','factory','checkpoint','reference_V74_protocol','reference_V74_runner'):check(p[k])
 assert p['code']==e['code']and p['generated']==e['generated']and read(e['generated']['path'])['PASS']
 cs=read(e['canonical_RGB_seal']['path']);assert cs['completed']and cs['PASS']and cs['all4_decoded_RGB_exact_V103'];frames=rows(check(cs['manifest']));assert len(frames)==4 and[r['frame_index']for r in frames]==list(range(4))
 source=rows(p['source_inputs']['path'])
 for r,s in zip(frames,source):assert all(r[k]==v for k,v in s.items())
 OUT.mkdir();(OUT/'P3').mkdir();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['YOLO_CONFIG_DIR']=str(OUT/'runtime/ultralytics');os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
 import torch,cv2,ultralytics
 assert sys.version_info[:2]==(3,10)and torch.__version__=='2.1.1+cu118'and cv2.__version__=='4.9.0'and ultralytics.__version__=='8.4.60'
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(True)
 spec=importlib.util.spec_from_file_location('v104_frozen_initializer',p['factory']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 model=m.FrozenYoloFeatures(p['checkpoint']['path'],device='cuda:0');loaded=state(model.yolo.model);model.yolo.model.fuse(verbose=False);model.yolo.model.requires_grad_(False).eval();before=state(model.yolo.model)
 calls=[0];hook=model.yolo.model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1));write(OUT/'STARTED.json',dict(entry=bind(entry),loaded_state=loaded,after_standard_fusion_state=before,optimizer_updates=0,PnP=0));start=time.monotonic();proposals=[];queries=[];selected_frames=[];first=None;firstp3=None
 def rgb(r):
  a=np.load(check(r['canonical_RGB']),allow_pickle=False);assert a.dtype==np.uint8 and list(a.shape)==r['raw_hw']+[3]and hashlib.sha256(a.tobytes()).hexdigest()==r['decoded_RGB_sha256'];return np.ascontiguousarray(a[:,:,::-1])
 try:
  for i,r in enumerate(frames):
   n=calls[0];z=model.predict(rgb(r),already_padded=False,cpu_features=True);s=serial(z);q=selected_queries(z,r['raw_hw']);gain,offset=m.canvas_affine(z['canvas_shape'],z['input_shape']);p3=z['p3'].numpy();h,w=np.array(z['input_shape'])//8;assert p3.dtype==np.float16 and p3.shape==(64,80,80)and np.isfinite(p3).all()
   path=OUT/'P3'/f'FRAME_{i:02d}.npz';np.savez_compressed(path,p3=p3[:,:h,:w]);base=dict(frame_index=i,input_id=r['input_id'],original_image={'path':r['image'],'sha256':r['image_sha256']},raw_hw=r['raw_hw'],canonical_RGB=r['canonical_RGB'],decoded_RGB_sha256=r['decoded_RGB_sha256'])
   allL=[native_grid()(np.array(c['keypoints_xy']),r['raw_hw'])[2]for c in z['candidates']]
   proposals.append(dict(base,**s,candidate_native8_L=allL,gain=gain,offset=offset.tolist(),forward_module_hooks=calls[0]-n,P3_input_only=bind(path)))
   selected=z['selected_index'];selected_frames.append(dict(base,selected_bbox_xyxy=None if selected is None else z['candidates'][selected]['box_xyxy'].tolist(),initial_points=[x['query_xy']for x in q],L8=[x['roi_side']for x in q],no_detection=selected is None,native_semantic_mapping='UNKNOWN'))
   for x in q:queries.append(dict(query_id=r['input_id']+':native'+str(x['native_slot']),frame_index=i,input_id=r['input_id'],**x,native_semantic_mapping='UNKNOWN',human_gold=False))
   if i==0:first=s;firstp3=p3.copy()
  repeat=model.predict(rgb(frames[0]),already_padded=False,cpu_features=True);assert serial(repeat)==first and np.array_equal(repeat['p3'].numpy(),firstp3)
  assert state(model.yolo.model)==before and sha(p['checkpoint']['path'])==p['checkpoint']['sha256']and all(not x.training for x in model.yolo.model.modules())and all(not x.requires_grad for x in model.yolo.model.parameters())
  assert len(proposals)==4 and len(queries)==32
  jsonl(OUT/'FROZEN_PROPOSALS_ALL4.jsonl',proposals);jsonl(OUT/'SELECTED_FRAMES4_INPUT_ONLY.jsonl',selected_frames);jsonl(OUT/'SELECTED_QUERIES32_INPUT_ONLY.jsonl',queries)
  write(OUT/'RESULT.json',dict(completed=True,PASS=True,entry=bind(entry),frames=4,slots=32,all_candidates_retained=sum(len(x['candidates'])for x in proposals),no_detection_frames=sum(x['selected_index']is None for x in proposals),firstscore_argmax_native_order_preserved=True,native_grid_L_AST_exact=True,all32slots_including_no_detection_OOB=True,explicit_source_predict_calls=4,explicit_repeat_calls=1,total_module_hooks=calls[0],model_state_before=before,model_state_after=state(model.yolo.model),checkpoint_unchanged=True,first_prediction_P3_repeat_exact=True,precision='Exact V74public initializer: matmulTF32off/cudnnTF32on, halfFalse',seconds=time.monotonic()-start,GT_or_CSV_read=False,source_input_only=True,query_correction_fit_current_private_PnP=0,training_admission=False))
  write(OUT/'INITIALIZER_OUTPUT_SEAL.json',dict(completed=True,PASS=True,result=bind(OUT/'RESULT.json'),selected_frames=bind(OUT/'SELECTED_FRAMES4_INPUT_ONLY.jsonl'),selected_queries=bind(OUT/'SELECTED_QUERIES32_INPUT_ONLY.jsonl'),all_candidates=bind(OUT/'FROZEN_PROPOSALS_ALL4.jsonl'),canonical_RGB_seal=e['canonical_RGB_seal'],files=[bind(x)for x in sorted(OUT.rglob('*'))if x.is_file()],new_query_labels_fit_current_PnP=0));print(json.dumps(bind(OUT/'INITIALIZER_OUTPUT_SEAL.json')),flush=True)
 except BaseException as ex:
  write(OUT/'FAILED.json',dict(error=repr(ex),automatic_retry=False));raise
 finally:hook.remove();model.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--entry',type=Path,required=True);run(a.parse_args().entry)
