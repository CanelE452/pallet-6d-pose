"""Actual original initializer queries on the fixed100 public Pallox frames only.
No visibility head, GT-cue association, crop correction, fit, current223 or PnP.
"""
from pathlib import Path
import os,json,hashlib,datetime,ast,argparse,traceback,time
import numpy as np
HERE=Path(__file__).resolve().parent;V94=HERE.parent;BASE=V94.parent
FRAMES=BASE/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl'
RECORDS=BASE/'query_visibility_public_cue_inputs_20261023_v90/input_pack/FRAME_FULL_P3_RECORDS.jsonl'
PRIOR=BASE/'query_visibility_public_cue_inputs_20261023_v90/INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json'
ROI=BASE.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
BACK=BASE/'query_visibility_public_real_20261022_v63/model/deployment_assets/frozen_yolo_features.py'
OUT=V94/'pallox_actual_initializer/FIXED100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def native_function():
 tree=ast.parse(ROI.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_grid'];assert len(nodes)==1
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROI),'exec'),ns);return ns['native_grid']
def selected_queries(z,hw):
 candidates=z['candidates'];selected=None if not candidates else int(np.argmax([c['score']for c in candidates]))
 assert selected==z['selected_index']
 if selected is None:return [dict(native_slot=k,query_xy=None,roi_side=None,original_query_continuous_valid=False,query_nearest_pixel_valid=False,query_rgb96_centre_domain_valid=False,reason='NO_DETECTION_U')for k in range(8)]
 q=np.asarray(candidates[selected]['keypoints_xy'],np.float64);assert q.shape==(9,2)and np.isfinite(q).all()
 _,_,L=native_function()(q,hw);H,W=hw;rows=[]
 for k in range(8):
  x,y=q[k];valid=bool(0<=x<W and 0<=y<H);nearest=bool(0<=np.floor(x+.5)<W and 0<=np.floor(y+.5)<H)
  rgbcentre=bool(-.5<=x<W-.5 and -.5<=y<H-.5);assert nearest==rgbcentre
  rows.append(dict(native_slot=k,query_xy=q[k].tolist(),roi_side=L,original_query_continuous_valid=valid,query_nearest_pixel_valid=nearest,query_rgb96_centre_domain_valid=rgbcentre,reason='VALID_QUERY_NOT_A_VISIBILITY_LABEL'if valid else 'OOB_OR_INVALID_U'))
 return rows
def serial(z):
 return dict(candidates=[{k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in c.items()}for c in z['candidates']],selected_index=z['selected_index'],input_shape=list(z['input_shape']),canvas_shape=list(z['canvas_shape']),added_border=z['added_border'])
def generated():
 q=np.array([[100.+i,120.+2*i]for i in range(9)])
 c=lambda score,pts:dict(score=score,keypoints_xy=pts)
 z=dict(candidates=[c(.8,q),c(.8,q+30)],selected_index=0)
 a=selected_queries(z,[480,640]);assert len(a)==8 and a[0]['query_xy']==[100.,120.]and a[0]['roi_side']==16
 q2=q.copy();q2[8]=[100000,100000];z2=dict(candidates=[c(.8,q2)],selected_index=0)
 assert selected_queries(z2,[480,640])==a
 b=selected_queries(dict(candidates=[],selected_index=None),[480,640]);assert len(b)==8 and all(x['query_xy']is None for x in b)
 q3=q.copy();q3[0]=[-100.,-100.];z3=dict(candidates=[c(.9,q3)],selected_index=0);r=selected_queries(z3,[480,640]);assert not r[0]['original_query_continuous_valid']and r[0]['query_xy']==[-100.,-100.]
 for xy,cv,nv in [([-.25,10.],False,True),([639.75,10.],True,False),([0.,0.],True,True),([-.5,0.],False,True),([639.5,0.],True,False)]:
  qb=q.copy();qb[0]=xy;rb=selected_queries(dict(candidates=[c(.9,qb)],selected_index=0),[480,640])[0]
  assert rb['original_query_continuous_valid']==cv and rb['query_nearest_pixel_valid']==nv and rb['query_rgb96_centre_domain_valid']==nv
 tree=ast.parse(BACK.read_text());fun=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='canvas_affine'];ns={'np':np};exec(compile(ast.Module(body=fun,type_ignores=[]),str(BACK),'exec'),ns)
 g,o=ns['canvas_affine']([680,840],[544,640]);assert g==640/840 and np.array_equal(o,[0.,13.])
 write(HERE/'GENERATED_QUERY_SERIALIZATION_TESTS.json',dict(PASS=True,completed=True,separate_native_continuous_nearest_RGBcentre_bounds=True,explicit_canvas_affine_fixture=True,code=bind(__file__),native_grid=bind(ROI),first_argmax_tie_preserved=True,all8_slots_no_detection_preserved=True,centroid_excluded_from_query_and_L=True,no_q_snap_or_confidence_replacement=True,actual_source_arrays_pixels_weights_forward=0))
def freeze():
 assert read(HERE/'GENERATED_QUERY_SERIALIZATION_TESTS.json')['PASS'];f=[json.loads(x)for x in FRAMES.read_text().splitlines()];r=[json.loads(x)for x in RECORDS.read_text().splitlines()];assert len(f)==len(r)==100
 assert all(a['frame_input_id']==b['frame_input_id']and a['image_sha256']==b['image_sha256']for a,b in zip(f,r))
 prior=read(PRIOR)
 write(HERE/'PROTOCOL_BEFORE_SOURCE_FORWARD.json',dict(schema='V94_ACTUAL_FROZEN_PALLOX100_INITIALIZER_INPUT_ONLY',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),code=bind(__file__),generated=bind(HERE/'GENERATED_QUERY_SERIALIZATION_TESTS.json'),frames=bind(FRAMES),records=bind(RECORDS),prior_protocol=bind(PRIOR),native_grid=bind(ROI),assets=prior['assets'],output=str(OUT),source_frames=100,slots=800,source='Originalfixed50pairs/all100publicPalloxframes,TRAIN_LAB44/VAL_HALL56; previously exposed/usedsource, onephysicalbin',boundary_flags='Nativecontinuous[0,W)x[0,H), nearestfloor(q+.5), andRGB96centre[-.5,W-.5) separately reported. No pack-valid intersection invented in this intake.',affine='Exact frozen canvas_affine(nativecanvas,actualinput) gain/offset compared with everyV90 record.',selection='Same frozen factory/conf.001/firstargmax native raw9; retain all candidates andnative0..7,exclude centroid8; no GTbest/no q orL changes',L='Exact V60 native_grid AST: quarter finite native8span clipped16..maxHW; no GTref',decoder='Same legacycv2 nativePNG; verify each decodedRGB SHA exactly againstV90 canonicalrecord',P3='Compare each actual unpadded FP16 capturedP3 and canvas/shape/pad exactly with existingV90 fullP3. ReuseV90 fullP3 bindings only if all100exact; no new fullP3 duplicate saved.',runtime=dict(python='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',cv2='4.9.0',device='cuda:0',FP32=True,TF32=False,threads=1,deterministic=True),calls=dict(blank_generated_warmup=1,source=100,repeat_first=1,no_autoretry=True),immutable_model='StateSHA aftergeneratedwarmup vs afterallactualsource identical; checkpointfileSHA unchanged',output_budget_bytes=20000000,forbidden=['No visibilityhead/checkpoint, fit, current223, GT cue/pose/depth/mask association, new source labels, PnP.','No source frame/slot drop, target re-selection or GT-based crop.'],training_admission=False))
def state(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return h.hexdigest()
def run(entryfile):
 e=read(entryfile);assert e['approved']is True and e['input_only']is True
 for k in ['protocol','code']:
  assert sha(e[k]['path'])==e[k]['sha256']
 p=read(e['protocol']['path']);assert sha(__file__)==p['code']['sha256']==e['code']['sha256']
 for k in ['frames','records','prior_protocol','native_grid','generated']:assert sha(p[k]['path'])==p[k]['sha256']
 for k in ['backbone_code','backbone_checkpoint']:assert sha(p['assets'][k]['path'])==p['assets'][k]['sha256']
 assert p['output']==str(OUT)and not OUT.exists();OUT.mkdir(parents=True)
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['YOLO_CONFIG_DIR']=str(OUT/'runtime/Ultralytics');os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
 import cv2,torch,importlib.util,resource
 assert torch.__version__==p['runtime']['torch']and cv2.__version__==p['runtime']['cv2']
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 spec=importlib.util.spec_from_file_location('v94_frozen_initializer',p['assets']['backbone_code']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 model=m.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
 f=[json.loads(x)for x in FRAMES.read_text().splitlines()];records=[json.loads(x)for x in RECORDS.read_text().splitlines()];assert len(f)==len(records)==100
 def image(row,rec):
  assert sha(row['image'])==row['image_sha256'];bgr=cv2.imread(row['image'],cv2.IMREAD_COLOR);assert list(bgr.shape[:2])==row['raw_hw']
  assert hashlib.sha256(np.ascontiguousarray(bgr[:,:,::-1]).tobytes()).hexdigest()==rec['decoded_RGB_sha256'];return bgr
 hooks=[0];handle=model.yolo.model.register_forward_pre_hook(lambda *_:hooks.__setitem__(0,hooks[0]+1))
 loaded=state(model.yolo.model);model.predict(np.zeros(tuple(f[0]['raw_hw'])+(3,),np.uint8),already_padded=False,cpu_features=True);warm=state(model.yolo.model);warmhooks=hooks[0]
 write(OUT/'STARTED.json',dict(entry=bind(entryfile),loaded_state_SHA=loaded,warmed_state_SHA=warm,standard_inference_initialization_change=loaded!=warm,generated_warmup_hooks=warmhooks,fit0=True))
 rows=[];queries=[];first=None;firstp3=None;start=time.monotonic()
 for i,(fr,rec)in enumerate(zip(f,records)):
  assert fr['frame_input_id']==rec['frame_input_id'];z=model.predict(image(fr,rec),already_padded=False,cpu_features=True);s=serial(z)
  assert list(z['input_shape'])==rec['actual_input_shape']and list(z['canvas_shape'])==rec['canvas_shape']and z['added_border']==rec['added_border']
  gain,offset=m.canvas_affine(z['canvas_shape'],z['input_shape']);assert gain==rec['gain']and np.array_equal(offset,rec['offset']),'Affine differs fromV90'
  assert sha(rec['full_P3']['path'])==rec['full_P3']['sha256'];cached=np.load(rec['full_P3']['path'],allow_pickle=False)['p3'];h,w=rec['P3_shape'][-2:];new=z['p3'].numpy()[:,:h,:w];assert np.array_equal(cached,new),'P3 differs fromV90 on '+fr['frame_input_id']
  row=dict(frame_index=i,**fr,**s,affine_gain=gain,affine_offset=offset.tolist(),V90_P3_exact=True,reusable_full_P3=rec['full_P3']);rows.append(row)
  for a in selected_queries(z,fr['raw_hw']):queries.append(dict(query_id=fr['frame_input_id']+':native'+str(a['native_slot']),frame_index=i,frame_input_id=fr['frame_input_id'],pair_id=fr['pair_id'],view=fr['view'],split=fr['split'],**a,native_semantic_state='UNKNOWN',human_gold=False))
  if i==0:first=s;firstp3=z['p3'].numpy().copy()
  if(i+1)%20==0:print(json.dumps(dict(source_frames=i+1,seconds=time.monotonic()-start)),flush=True)
 repeat=model.predict(image(f[0],records[0]),already_padded=False,cpu_features=True);assert serial(repeat)==first and np.array_equal(repeat['p3'].numpy(),firstp3)
 assert state(model.yolo.model)==warm and sha(p['assets']['backbone_checkpoint']['path'])==p['assets']['backbone_checkpoint']['sha256']
 assert all(not x.training for x in model.yolo.model.modules())and all(not x.requires_grad for x in model.yolo.model.parameters())
 for name,rr in [('FROZEN_PROPOSALS_ALL100.jsonl',rows),('ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl',queries)]:
  path=OUT/name;assert not path.exists();path.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n'for x in rr))
 files=[bind(OUT/x)for x in ['STARTED.json','FROZEN_PROPOSALS_ALL100.jsonl','ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl']]
 handle.remove();model.close();assert len(queries)==800
 write(OUT/'RESULT.json',dict(schema='V94_ACTUAL_PUBLIC_PALLOX100_ORIGINAL_INITIALIZER_QUERY_INTAKE',completed=True,PASS=True,entry=bind(entryfile),protocol=e['protocol'],files=files,frames=100,slots=800,no_detection_frames=sum(r['selected_index']is None for r in rows),valid_original_slots=sum(q['original_query_continuous_valid']for q in queries),reason_counts={reason:sum(q['reason']==reason for q in queries)for reason in sorted({q['reason']for q in queries})},all100_P3_exact_V90=True,no_new_full_P3_duplicate=True,repeat_first_prediction_andP3_exact=True,warmed_model_state_SHA=warm,after_state_unchanged=True,checkpoint_unchanged=True,explicit_predict_calls=102,total_forward_hooks=hooks[0],generated_warmup_hooks=warmhooks,seconds=time.monotonic()-start,peakRSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_current_head_fit_nativegold_PnP=0,training_admission=False))
 write(OUT/'INTAKE_SEAL.json',dict(completed=True,PASS=True,entry=bind(entryfile),result=bind(OUT/'RESULT.json'),files=[bind(x)for x in sorted(OUT.iterdir())if x.is_file()],new_current_head_fit_nativegold_PnP=0))
 print(json.dumps(dict(result=bind(OUT/'RESULT.json')),ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['generated','freeze','run']);a.add_argument('--entry',type=Path);z=a.parse_args()
 if z.stage=='generated':generated()
 elif z.stage=='freeze':freeze()
 else:
  try:assert z.entry;run(z.entry)
  except BaseException as err:
   if OUT.exists()and not(OUT/'HALTED_FAILURE.json').exists():write(OUT/'HALTED_FAILURE.json',dict(error=repr(err),traceback=traceback.format_exc(),no_autoretry=True))
   raise
