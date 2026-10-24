"""One source-only full-RGB -> matching three-tensor pack; no head or GT reader.

The root explicitly delegated freeze and one execution after generated checks.
GT is hash-bound as a separate provenance file, never deserialized here.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, importlib.util, json, os, shutil, sys, time
import numpy as np

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
PLAN=H/'PRODUCER_PLAN_BEFORE_CORPUS_FEATURES.json'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_ACTUAL_EXECUTION.json'
OUT=V/'matching_feature_pack'
FIELDS=('p3_roi','rgb96','crop_valid96')
STAGES=('BEFORE','COVER','SHAM')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
 assert sha(b['path'])==b['sha256'],b['path']
 return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rr):
 with Path(p).open('x')as f:
  for r in rr:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
 return h.hexdigest()
def validate_metadata(images,queries,occurrences):
 assert len(images)==288 and len(queries)==96 and len(occurrences)==2304
 for i,im in enumerate(images):
  bi,si=divmod(i,3);q=queries[bi]
  assert im['image_index']==i and im['base_index']==bi and im['variant']==STAGES[si]
  assert im['base_id']==q['base_id']and im['split']==q['split']and im['raw_hw']==[640,640]
  assert np.shape(q['query8_xy'])==(8,2)and np.shape(q['L'])==(8,)
  assert np.isfinite(q['query8_xy']).all()and np.isfinite(q['L']).all()and min(q['L'])>0
  for k,o in enumerate(occurrences[i*8:i*8+8]):
   assert o['occurrence_index']==i*8+k and o['image_index']==i and o['aux_corner_index']==k
   assert o['image_id']==im['image_id']and o['base_index']==bi and o['variant']==STAGES[si]and o['split']==im['split']
   assert o['query_xy']==q['query8_xy'][k]and o['L']==q['L'][k]
   assert o['native8_mapping']=='UNKNOWN'and o['source_emulated_not_initializer']is True
 return dict(frames=288,queries=2304,TRAIN=sum(r['split']=='TRAIN'for r in occurrences),VALIDATION=sum(r['split']=='VALIDATION'for r in occurrences))
def pack_frame(rgb,p3,q,L,math,gain,offset):
 assert rgb.dtype==np.uint8 and rgb.shape==(640,640,3)
 x,anyvalid,valid9=math.sample_p3(p3,q,L,rgb.shape[:2],gain,offset,100)
 rgbs=[];valids=[]
 for a,b in zip(q,L):
  c,v=math.crop(rgb,a,b,96);rgbs.append(c);valids.append(v)
 rgbs=np.array(rgbs,np.uint8);valids=np.array(valids,bool)
 centre,continuous,nearest=math.query_validity(q,rgb.shape[:2])
 central4=valids[:,0,47:49,47:49].reshape(8,-1).all(1)
 assert np.array_equal(x[:,64].astype(bool),valid9)
 return dict(p3_roi=x,rgb96=rgbs,crop_valid96=valids),dict(query_center_valid=centre,query_continuous_valid=continuous,query_nearest_valid=nearest,ROI_anyvalid=anyvalid,ROI_center_valid=valid9[:,4,4],RGB_central4_valid=central4,operator_query_available=valid9[:,4,4]&central4)
def freeze():
 assert not OUT.exists();plan=read(PLAN)
 rs=V/'render_preparation/actual_corpus/RENDER_SEAL.json';fs=V/'render_preparation/FINAL_PREPARATION_AND_CORPUS_SEAL.json'
 assert sha(rs)=='6887a2e6d4f1493d023188a27503d279909a35cae275629d02b190e9d4f9c68b'
 assert sha(fs)=='618aba1893ade4086d99a7458b462b41110fbe71a353c67bd61de80e573ef52f'
 r=read(rs);f=read(fs);assert r['completed']and r['PASS_execution']and f['PASS_execution_and_saved_QA']
 m=read(check(r['manifest']));assert m['completed']and m['query_count']==2304
 for b in r['files']:check(b)
 summary=validate_metadata(rows(check(m['full_RGB'])),rows(check(m['query_inputs'])),rows(check(m['input_occurrences'])))
 generated=H/'GENERATED_MATCHING_FEATURE_TESTS.json';proof=read(generated);assert proof['PASS']and proof['code']==bind(__file__)
 deps=plan['immutable_dependencies'];assets={k:deps[k]for k in('backbone_code','backbone_checkpoint','native_sampler_and_RGB')}
 for b in assets.values():check(b)
 gt=read(V/'geometry_query_preparation/GEOMETRY96_SEAL.json')['GT_only_targets'];check(gt) # bytes only; no GT parse.
 import torch,cv2,PIL,ultralytics
 assert torch.__version__=='2.1.1+cu118'and sys.version_info[:2]==(3,10)and np.__version__=='1.26.4'and cv2.__version__=='4.9.0'and PIL.__version__=='12.2.0'
 runtime=dict(python=str(Path(sys.executable).resolve()),torch=torch.__version__,numpy=np.__version__,cv2=cv2.__version__,PIL=PIL.__version__,ultralytics=ultralytics.__version__,torch_code=bind(torch.__file__),ultralytics_code=bind(ultralytics.__file__),CUDA=torch.version.cuda,device='cuda:0',threads=1,TF32=False,autocast=False)
 write(PROTOCOL,dict(completed=True,code=bind(__file__),generated=bind(generated),plan=bind(PLAN),render_seal=bind(rs),final_corpus_seal=bind(fs),source_manifest=r['manifest'],images=m['full_RGB'],queries=m['query_inputs'],occurrences=m['input_occurrences'],renderer_chunks=m['chunks'],assets=assets,runtime=runtime,summary=summary,GT_only_targets=gt,GT_body_read=False,source_forward_calls=288,first_repeat=1,blank_warmup=1,chunk_queries=192,output=str(OUT),maximum_output_bytes=160000000,minimum_free_bytes=250000000,
  authorization='Root delegated implementation, binding freeze, generated checks, and one actual source producer through completion in the current task. Supersedes planning-only execution boundary, not source recipe.',
  model_input_keys=list(FIELDS),GT_native_slot_source_labels_forwarded=False,full_P3_persisted=False,storage_dedup='Each occurrence stored once; no duplicate full RGB or global P3. All changed full RGB extracted independently, even equal local crops.',training_admission=False,current_private_head_fit_PnP_allowed=False))
 entry=H/'SOURCE_FEATURE_EXECUTION_BINDINGS.json';write(entry,dict(delegated_root_authorization=True,source_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(generated),output=str(OUT),head_fit_current_PnP_allowed=False))
 print(json.dumps(dict(protocol=bind(PROTOCOL),execution_bindings=bind(entry))),flush=True)
def authorize(entry):
 e=read(entry);assert e['delegated_root_authorization']and e['source_only']and e['head_fit_current_PnP_allowed']is False
 for k in('code','protocol','generated'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and not OUT.exists()
 p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and read(e['generated']['path'])['PASS']
 for k in('plan','render_seal','final_corpus_seal','source_manifest','images','queries','occurrences'):check(p[k])
 for b in p['assets'].values():check(b)
 for c in p['renderer_chunks']:check(c['file'])
 check(p['GT_only_targets']) # GT-only bytes binding; contents are not parsed.
 assert p['model_input_keys']==list(FIELDS)and p['GT_body_read']is False
 assert shutil.disk_usage(V).free>p['minimum_free_bytes']
 return e,p
def run(entry):
 e,p=authorize(entry)
 import torch,cv2,PIL
 from PIL import Image
 assert str(Path(sys.executable).resolve())==p['runtime']['python']and torch.__version__=='2.1.1+cu118'
 assert cv2.__version__=='4.9.0'and PIL.__version__=='12.2.0'and np.__version__=='1.26.4'
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 os.environ['YOLO_CONFIG_DIR']=str(H/'runtime/ultralytics')
 images=rows(p['images']['path']);queries=rows(p['queries']['path']);occs=rows(p['occurrences']['path']);summary=validate_metadata(images,queries,occs)
 OUT.mkdir();(OUT/'chunks').mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),no_silent_restart=True,source_predict_expected=288))
 start=time.monotonic();back=None
 try:
  math=module(p['assets']['native_sampler_and_RGB']['path'],'v99_pinned_spatial_math');bm=module(p['assets']['backbone_code']['path'],'v99_pinned_frozen_yolo')
  back=bm.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
  with torch.inference_mode():back.predict(np.zeros((640,640,3),np.uint8),already_padded=False,cpu_features=True)
  before=state_sha(back.yolo.model);chunks=[];lineage=[];facts=[];buffer={k:[]for k in FIELDS};repeat_exact=False;renderer=None;renderer_chunk=-1
  torch.cuda.reset_peak_memory_stats()
  for fi,im in enumerate(images):
   bi,si=divmod(fi,3);q=np.array(queries[bi]['query8_xy'],np.float64);L=np.array(queries[bi]['L'],np.float64)
   rgb=np.array(Image.open(check(im['image'])).convert('RGB'),copy=True);assert rgb.shape==(640,640,3)and arrsha(rgb)==im['decoded_RGB_sha256']
   with torch.inference_mode():z=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
   if fi==0:
    with torch.inference_mode():zz=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
    assert torch.equal(z['p3'],zz['p3']);repeat_exact=True
    for k in('added_border','canvas_shape','input_shape'):assert z[k]==zz[k]
   a=math.expected_affine(rgb.shape[:2]);gain,offset=bm.canvas_affine(z['canvas_shape'],z['input_shape'])
   assert z['added_border']==100 and list(z['canvas_shape'])==a['canvas_shape']and list(z['input_shape'])==a['input_shape']
   assert gain==a['gain']and np.array_equal(offset,a['offset'])
   h,w=a['p3_actual_hw'];plane=z['p3'].numpy();assert plane.dtype==np.float16 and plane.shape==(64,80,80)
   assert not plane[:,h:,:].any()and not plane[:,:,w:].any()
   x,fact=pack_frame(rgb,plane[:,:h,:w],q,L,math,gain,offset)
   if bi//8!=renderer_chunk:
    if renderer is not None:renderer.close()
    renderer_chunk=bi//8;renderer=np.load(check(p['renderer_chunks'][renderer_chunk]['file']),allow_pickle=False)
   assert np.array_equal(renderer['query8_xy'][bi%8],q)and np.array_equal(renderer['L'][bi%8],L)
   assert np.array_equal(renderer['rgb96'][bi%8,si],x['rgb96'])and np.array_equal(renderer['crop_valid96'][bi%8,si],x['crop_valid96'])
   for k in range(8):
    o=dict(occs[fi*8+k]);assert o['rgb96_sha256']==arrsha(x['rgb96'][k])and o['crop_valid96_sha256']==arrsha(x['crop_valid96'][k])
    o.update(input_unit_index=fi*8+k,chunk_index=fi//24,index_in_chunk=(fi%24)*8+k,input_facts={n:bool(v[k])for n,v in fact.items()},p3_roi_sha256=arrsha(x['p3_roi'][k]));lineage.append(o)
   for k in FIELDS:buffer[k].append(x[k])
   facts.append(dict(image_index=fi,image_id=im['image_id'],image=im['image'],decoded_RGB_sha256=im['decoded_RGB_sha256'],raw_hw=im['raw_hw'],canvas_shape=list(z['canvas_shape']),input_shape=list(z['input_shape']),P3_shape=[64,h,w],P3_dtype='float16',gain=gain,offset=offset.tolist(),added_border=100,full_P3_stored=False,renderer_RGB96_valid_bitexact=True))
   if(fi+1)%24==0:
    cp=OUT/'chunks'/f'queries_{fi//24:03d}.npz';arrays={k:np.concatenate(buffer[k])for k in FIELDS};assert all(len(v)==192 for v in arrays.values());np.savez_compressed(cp,**arrays)
    chunks.append(dict(chunk_index=len(chunks),start_occurrence=(fi-23)*8,end_occurrence=(fi+1)*8,count=192,file=bind(cp)));buffer={k:[]for k in FIELDS}
   if(fi+1)%24==0:print(json.dumps(dict(frames=fi+1,total=288,queries=len(lineage),seconds=time.monotonic()-start)),flush=True)
   assert sum(f.stat().st_size for f in OUT.rglob('*')if f.is_file())<p['maximum_output_bytes']
  if renderer is not None:renderer.close()
  after=state_sha(back.yolo.model);assert before==after and repeat_exact
  assert all(not v.requires_grad for v in back.yolo.model.parameters())and all(not m.training for m in back.yolo.model.modules())
  back.close();back=None
  for b in p['assets'].values():check(b)
  assert len(lineage)==2304 and len(chunks)==12 and all(not buffer[k]for k in FIELDS)
  lp=OUT/'INPUT_LINEAGE2304_INPUT_ONLY.jsonl';fp=OUT/'FRAME_FEATURE_FACTS288_INPUT_ONLY.jsonl';jsonl(lp,lineage);jsonl(fp,facts)
  gp=OUT/'GT_ONLY_PROVENANCE.json';write(gp,dict(completed=True,GT_only_targets=p['GT_only_targets'],GT_body_read_by_producer=False,GT_forwarded=False,native8_mapping='UNKNOWN',human_gold=False))
  man=dict(completed=True,frame_count=288,query_count=2304,unique_input_units=2304,summary=summary,chunks=chunks,input_lineage=bind(lp),frame_facts=bind(fp),GT_only_provenance=bind(gp),model_input_keys=list(FIELDS),array_schema=dict(p3_roi='float16[N,65,9,9]',rgb96='uint8[N,3,96,96]',crop_valid96='bool[N,1,96,96]'),full_P3_stored=False,source_predict_calls=288,first_repeat_calls=1,blank_warmup_calls=1,model_state_before=before,model_state_after=after,checkpoint=p['assets']['backbone_checkpoint'],runtime=p['runtime'],all2304_RGB96_valid_bitexact=True,source_emulated_queries=True,native8_mapping='UNKNOWN',training_admission=False,new_head_fit_private_current_PnP=0,seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
  write(OUT/'PACK_MANIFEST.json',man);write(OUT/'INPUT_PACK_QA.json',dict(completed=True,PASS=True,all2304_q_L_order_split_exact=True,all2304_RGB96_valid_bitexact=True,all_P3_channel64_valid_exact=True,first_global_P3_repeat_exact=True,model_state_unchanged=True,GT_body_read=False,head_fit_private_current_PnP=0))
  write(OUT/'INPUT_PACK_SEAL.json',dict(completed=True,PASS=True,protocol=bind(PROTOCOL),execution_bindings=bind(entry),manifest=bind(OUT/'PACK_MANIFEST.json'),QA=bind(OUT/'INPUT_PACK_QA.json'),input_lineage=bind(lp),frame_facts=bind(fp),GT_only_provenance=bind(gp),chunks=[c['file']for c in chunks],output_bytes_before_seal=sum(f.stat().st_size for f in OUT.rglob('*')if f.is_file()),new_head_fit_private_current_PnP=0))
  print(json.dumps(dict(completed=True,seal=bind(OUT/'INPUT_PACK_SEAL.json'),seconds=man['seconds'],summary=summary)),flush=True)
 except BaseException as exc:
  if not(OUT/'FAILED.json').exists():write(OUT/'FAILED.json',dict(completed=False,exception_type=type(exc).__name__,exception=str(exc),source_frames_completed=locals().get('fi',-1),preserve_no_automatic_restart=True))
  raise
 finally:
  if back is not None:back.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.freeze:freeze()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --freeze metadata only, then --entry one source producer')
