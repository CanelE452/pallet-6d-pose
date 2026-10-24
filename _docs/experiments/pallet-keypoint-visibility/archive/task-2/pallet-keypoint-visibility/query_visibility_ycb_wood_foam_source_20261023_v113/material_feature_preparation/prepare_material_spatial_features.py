"""C11-exact material-source features. Inert import; preparation is metadata only.

Only an explicit root extraction entry may decode RGB/load frozen DINO. No GT,
teacher packet, reference, depth, mask, pose, head, current or V112 is read.
Old C11 features are reused by immutable binding, never recomputed or copied.
"""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
C11=P/'query_visibility_spatial_target_query_features_20261023_v111'
SOURCE_PROTOCOL=C11/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';SOURCE_PROTOCOL_SHA='e7f12ad8bf61994ec22fd510839d213564f7dd97933e6a89587f6f1063080645'
PRODUCER=C11/'feature_preparation/prepare_spatial_query_features.py';PRODUCER_SHA='cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e'
C10=P/'query_visibility_natural_surface_rgb_head_20261023_v110/feature_preparation/prepare_source_rgb_features.py'
OLD_RECEIPT=C11/'spatial_query_features/FEATURE_COMPLETE.json'
QUERY_PROTOCOL=V/'query_preparation/PRODUCER_PROTOCOL.json';QUERY_PROTOCOL_SHA='f9d358638a5729c59b535182b1c75440a27f58d3b34f5d4f86032da6925966f8'
INPUTS=V/'query_preparation/INPUTS400_QUERY_EMULATION_ONLY.jsonl';INPUTS_SHA='3448555230c78a753adc085afc3afb61932c4484c95a72e09b72d517f9217f13'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';OUT=V/'material_spatial_features'
FIELDS={'input_id','instance_row_index','plan_row_index','image','raw_hw','initial_points','roi_side','query_box_xyxy','original_split','split','source','source_emulation','actual_initializer_query','native_mapping','training_admission'}
OLD_UNITS=11652;OLD_OCCURRENCES=41984;NEW_FRAMES=400;NEW_UNITS=3200

def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(d):require(sha(d['path'])==d['sha256'],'Changed binding: '+d['path']);return Path(d['path'])
def same(a,b):return a.get('path')==b.get('path')and a.get('sha256')==b.get('sha256')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_inputs(rs):
 require(len(rs)==NEW_FRAMES and len({r['input_id']for r in rs})==NEW_FRAMES,'All400 original instance rows required')
 for i,r in enumerate(rs):
  require(set(r)==FIELDS and set(r['image'])=={'path','sha256'},'Strict input-only whitelist')
  require(r['instance_row_index']==i and isinstance(r['plan_row_index'],int)and r['raw_hw']==[480,640],'Original instance order/HW')
  q=np.asarray(r['initial_points'],np.float64);b=np.asarray(r['query_box_xyxy'],np.float64)
  require(q.shape==(8,2)and np.isfinite(q).all()and b.shape==(4,)and np.isfinite(b).all(),'Original q8 geometry')
  require((b[2:]>b[:2]).all()and np.array_equal(b,np.r_[q.min(0),q.max(0)]),'Unchanged unclipped q8 minmax context')
  require(np.isfinite(r['roi_side'])and r['roi_side']>0,'Original positive L')
  require((r['original_split'],r['split'])in(('train','TRAIN'),('val','VALIDATION')),'Original fold mapping only')
  require(r['source']=='PUBLIC_YCB_MATERIAL_GT_JITTER_EMULATION'and r['source_emulation']is True and r['actual_initializer_query']is False and r['native_mapping']=='UNKNOWN'and r['training_admission']is False,'Emulated source scope')
 require(sum(r['split']=='TRAIN'for r in rs)==300 and sum(r['split']=='VALIDATION'for r in rs)==100,'Fixed300TRAIN/100VALIDATION')
 return rs

def local_math(p):
 fn=next(n for n in ast.parse(Path(p['C10_pure_math']['path']).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='pure_local_math')
 ns=dict(ast=ast,np=np,LOCAL_MATH=Path(p['local_crop_code']['path']))
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(C10)+'::pure_local_math_only','exec'),ns)
 return ns['pure_local_math']()

def make_features(r,rgb,producer,c8,c9,tm,crop,qvalid,encode,torch,device):
 q=np.asarray(r['initial_points'],np.float64);centre,continuous,nearest=qvalid(q,r['raw_hw'])
 target,tv=tm.sample_target_context(rgb,r['query_box_xyxy']);lr=[];lv=[];fra=[];central=[];facts=[]
 for k,point in enumerate(q):
  rr,v=crop(rgb,point,r['roi_side'],96);c4=bool(v[:,47:49,47:49].all());lr.append(rr);lv.append(v);central.append(c4)
  fra.append([np.float32(tv.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())])
  facts.append(dict(logical_aux_slot=k,q_continuous=bool(continuous[k]),q_nearest=bool(nearest[k]),q_center_valid=bool(centre[k]),RGB_central4_valid=c4,RGB96_sha256=hashlib.sha256(rr.tobytes()).hexdigest(),valid96_sha256=hashlib.sha256(v.tobytes()).hexdigest()))
 local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device=device),torch.as_tensor(np.stack(lv),device=device))
 context=c9.preprocess_target(torch.as_tensor(target.transpose(2,0,1)[None],device=device),torch.as_tensor(tv[None,None],device=device))
 views=torch.cat((context,local),0);require(tuple(views.shape)==(17,3,224,224),'One context plus8full+8central')
 raw=encode(views);cls=raw['x_norm_clstoken'];patch=raw['x_norm_patchtokens']
 with torch.inference_mode():x,inside=producer.assemble(cls[:1].repeat_interleave(8,0),patch[:1].repeat_interleave(8,0),cls[1:9],patch[1:9],cls[9:],q,[r['query_box_xyxy']]*8,fra,central)
 factual=np.asarray(centre,bool)&np.asarray(central,bool);context_available=np.full(8,bool(tv.any()),bool);available=factual&context_available
 return x.cpu().numpy(),factual,context_available,available,inside,dict(input_id=r['input_id'],instance_row_index=r['instance_row_index'],plan_row_index=r['plan_row_index'],image=r['image'],raw_hw=r['raw_hw'],split=r['split'],original_split=r['original_split'],decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),query_box_xyxy=r['query_box_xyxy'],target_valid_fraction=float(np.float32(tv.mean())),target_RGB224_F32_sha256=hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),queries=facts)

def mapping(rs):
 out=[]
 for i,r in enumerate(rs):
  for k in range(8):
   u=8*i+k
   out.append(dict(local_occurrence_index=u,local_input_unit_index=u,global_occurrence_index=OLD_OCCURRENCES+u,global_input_unit_index=OLD_UNITS+u,instance_row_index=i,plan_row_index=r['plan_row_index'],logical_aux_slot=k,input_id=r['input_id'],query_id=r['input_id']+'::q'+str(k),image=r['image'],raw_hw=r['raw_hw'],query_xy=r['initial_points'][k],L=r['roi_side'],query_box_xyxy=r['query_box_xyxy'],split=r['split'],original_split=r['original_split'],source=r['source'],source_emulated_not_initializer=True,native_mapping='UNKNOWN'))
 return out

def prepare():
 require(sha(SOURCE_PROTOCOL)==SOURCE_PROTOCOL_SHA and sha(PRODUCER)==PRODUCER_SHA,'Pinned C11 operator')
 require(sha(QUERY_PROTOCOL)==QUERY_PROTOCOL_SHA and sha(INPUTS)==INPUTS_SHA,'Frozen400 query metadata')
 sp=read(SOURCE_PROTOCOL);rs=validate_inputs(rows(INPUTS));old=read(OLD_RECEIPT)
 require(old['completed']and old['PASS']and old['features_shape']==[OLD_UNITS,1928]and old['occurrences']==OLD_OCCURRENCES,'Old features receipt shape')
 g=H/'GENERATED_FEATURE_TESTS.json';require(read(g)['PASS']and same(read(g)['code'],bind(__file__)),'Current generated proof')
 p=dict(completed=True,code=bind(__file__),generated=bind(g),query_protocol=bind(QUERY_PROTOCOL),inputs=bind(INPUTS),source_protocol=bind(SOURCE_PROTOCOL),producer=bind(PRODUCER),C10_pure_math=bind(C10),local_crop_code=sp['local_crop_code'],target_crop_code=sp['target_crop_code'],C8_preprocess=sp['C8_preprocess'],C9_target_preprocess=sp['C9_target_preprocess'],weight=sp['weight'],official_repo=sp['official_repo'],official_revision=sp['official_revision'],official_source_bindings=sp['official_source_bindings'],asset_runtime=sp['asset_runtime'],runtime=sp['runtime'],old_receipt=bind(OLD_RECEIPT),old_output=old['output'],old_features_reused_by_binding_not_copied_or_recomputed=True,old_units=OLD_UNITS,old_occurrences=OLD_OCCURRENCES,new_frames=NEW_FRAMES,new_units=NEW_UNITS,new_occurrences=NEW_UNITS,new_features_shape=[NEW_UNITS,1928],new_occurrence_unit='0..3199 exact instance-major then logicalslot0..7',combined_units=OLD_UNITS+NEW_UNITS,combined_occurrences=OLD_OCCURRENCES+NEW_UNITS,new_global_unit_offset=OLD_UNITS,new_global_occurrence_offset=OLD_OCCURRENCES,fold_instances=dict(TRAIN=300,VALIDATION=100),fold_occurrences=dict(TRAIN=2400,VALIDATION=800),max_views=17,feature_layout='T384,Q384,F384,C384,P384,cosTC,cosTF,cosQC,cosQP,targetfrac,localfrac,centralfrac,Qinside',availability='native continuous AND nearest AND non-sentinel AND RGB96 central4 AND context any; Qoutside adds no U',actual_extraction_pending_root_entry_and_conditional_source_admission=True,conditional_source_admission_binding='Pending in actual root entry; hash only, body never read by this input-only producer',labels_GT_reference_pose_depth_masks_private_current_V112_reads=0,head_forward_fit_PnP=0,output=str(OUT))
 write(PROTOCOL,p)
 e=dict(approved=False,stage='EXTRACT_MATERIAL_SPATIAL_FEATURES',actual_source_encoder_extraction_authorized=False,corpus_review_and_conditional_source_admission_confirmed=False,conditional_source_admission=dict(path=None,sha256=None),input_only=True,training_allowed=False,current_allowed=False,GT_body_reads_allowed=False,PnP_allowed=False,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(g),inputs=bind(INPUTS),old_receipt=bind(OLD_RECEIPT),output_directory=str(OUT))
 write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',e)
 print(json.dumps(dict(protocol=bind(PROTOCOL),entry_template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),frames=len(rs),units=NEW_UNITS,actual_encoder_calls=0)))

def authorize(path):
 e=read(path);require(e.get('approved')is True and e.get('stage')=='EXTRACT_MATERIAL_SPATIAL_FEATURES','Explicit root extraction entry required')
 require(e['actual_source_encoder_extraction_authorized']is True and e['corpus_review_and_conditional_source_admission_confirmed']is True,'Awaiting root corpus/admission confirmation')
 require(e['input_only']and e['training_allowed']is False and e['current_allowed']is False and e['GT_body_reads_allowed']is False and e['PnP_allowed']is False,'Input-only extraction scope')
 for k in('code','protocol','generated','inputs','old_receipt'):check(e[k])
 p=read(PROTOCOL);require(same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))and same(p['code'],e['code'])and same(p['generated'],e['generated'])and same(e['inputs'],p['inputs'])and same(e['old_receipt'],p['old_receipt']),'Entry/code/input binding mismatch')
 require(read(e['generated']['path'])['PASS']and same(read(e['generated']['path'])['code'],e['code']),'Stale generated proof')
 # This admission certificate is bound by bytes only; no GT-bearing body is read.
 check(e['conditional_source_admission'])
 for k in('query_protocol','inputs','source_protocol','producer','C10_pure_math','local_crop_code','target_crop_code','C8_preprocess','C9_target_preprocess','asset_runtime','old_receipt'):check(p[k])
 require(sha(SOURCE_PROTOCOL)==SOURCE_PROTOCOL_SHA and sha(PRODUCER)==PRODUCER_SHA and sha(QUERY_PROTOCOL)==QUERY_PROTOCOL_SHA and sha(INPUTS)==INPUTS_SHA,'Frozen implementation/input changed')
 old=read(p['old_receipt']['path']);require(old['completed']and old['PASS']and old['features_shape']==[OLD_UNITS,1928]and old['occurrences']==OLD_OCCURRENCES and same(old['output'],p['old_output']),'Old receipt changed')
 return e,p

def run(entry):
 e,p=authorize(entry);out=Path(e['output_directory']);require(out.resolve()==OUT.resolve(),'Fixed output directory')
 rs=validate_inputs(rows(p['inputs']['path']));out.mkdir(exist_ok=False);write(out/'STARTED.json',dict(entry=bind(entry),labels_GT_body_read=False,no_automatic_restart=True))
 try:
  import torch
  from PIL import Image,__version__ as PILVERSION
  require(torch.__version__==p['runtime']['torch']and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve(),'Exact learned runtime')
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  for d in p['official_source_bindings']:check(d)
  check(p['weight']);check(p['old_output']) # byte verification only; old array never deserialized/copied
  producer=module(p['producer']['path'],'v113_c11_math');c8=module(p['C8_preprocess']['path'],'v113_local_preprocess');c9=module(p['C9_target_preprocess']['path'],'v113_target_preprocess');tm=module(p['target_crop_code']['path'],'v113_target_math');crop,qvalid=local_math(p)
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);encoder.load_state_dict(torch.load(p['weight']['path'],map_location='cpu',weights_only=True),strict=True);encoder=encoder.cuda().float().eval().requires_grad_(False)
  require(sum(v.numel()for v in encoder.parameters())==22058112,'DINO architecture')
  before=producer.state_sha(encoder);require(before==read(p['asset_runtime']['path'])['state_before_SHA256'],'Frozen DINO state')
  calls=0;repeated=False;start=time.monotonic()
  def encode(views):
   nonlocal calls,repeated
   with torch.inference_mode():
    a=encoder.forward_features(views);calls+=1
    if not repeated:
     b=encoder.forward_features(views);calls+=1;require(all(torch.equal(a[k],b[k])for k in('x_norm_clstoken','x_norm_patchtokens')),'First batch repeat differs');repeated=True
   return a
  features=[];factual=[];context=[];available=[];inside=[];facts=[]
  for r in rs:
   rgb=np.array(Image.open(check(r['image'])).convert('RGB'),dtype=np.uint8,copy=True);require(list(rgb.shape)==r['raw_hw']+[3],'Original image HW')
   x,fa,ca,aa,ii,fc=make_features(r,rgb,producer,c8,c9,tm,crop,qvalid,encode,torch,'cuda');features.append(x);factual.extend(fa);context.extend(ca);available.extend(aa);inside.extend(ii);facts.append(fc)
   if len(facts)%25==0:print(json.dumps(dict(frames=len(facts),total=NEW_FRAMES,seconds=time.monotonic()-start)),flush=True)
  x=np.concatenate(features).astype(np.float32);fa=np.asarray(factual,bool);ca=np.asarray(context,bool);aa=np.asarray(available,bool);ii=np.asarray(inside,bool)
  require(x.shape==(NEW_UNITS,1928)and np.isfinite(x).all()and np.array_equal(aa,fa&ca),'New feature schema/availability')
  require(before==producer.state_sha(encoder),'Frozen encoder changed')
  lineage=mapping(rs)
  for row in lineage:
   u=row['local_input_unit_index'];row.update(factual_query_available=bool(fa[u]),context_available=bool(ca[u]),query_available=bool(aa[u]),context_Q_inside=bool(ii[u]))
  packet=out/'SOURCE_MATERIAL_SPATIAL_QUERY_FEATURES.npz'
  np.savez_compressed(packet,features=x,occurrence_unit=np.arange(NEW_UNITS,dtype=np.int32),query_available=aa,factual_query_available=fa,context_available=ca,unit_Q_inside=ii,global_occurrence_index=np.arange(OLD_OCCURRENCES,OLD_OCCURRENCES+NEW_UNITS,dtype=np.int32),global_input_unit_index=np.arange(OLD_UNITS,OLD_UNITS+NEW_UNITS,dtype=np.int32))
  for name,data in [('NEW3200_INPUT_LINEAGE_ONLY.jsonl',lineage),('INPUT_RGB_QUERY_FACTS400_INPUT_ONLY.jsonl',facts)]:
   with(out/name).open('x')as f:
    for row in data:f.write(json.dumps(row,allow_nan=False)+'\n')
  manifest=dict(completed=True,new_features=bind(packet),new_lineage=bind(out/'NEW3200_INPUT_LINEAGE_ONLY.jsonl'),input_facts=bind(out/'INPUT_RGB_QUERY_FACTS400_INPUT_ONLY.jsonl'),new_frames=NEW_FRAMES,new_units=NEW_UNITS,new_occurrences=NEW_UNITS,old_features=p['old_output'],old_receipt=p['old_receipt'],old_reused_by_binding_only=True,combined_units=OLD_UNITS+NEW_UNITS,combined_occurrences=OLD_OCCURRENCES+NEW_UNITS,combined_rule='Concatenate old feature units then new features; old occurrence map unchanged, new local occurrence_unit +11652; global occurrences41984..45183. No labels in input packet.',entry=bind(entry))
  write(out/'PACK_MANIFEST.json',manifest);check(p['weight']);check(p['old_output'])
  write(out/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(packet),manifest=bind(out/'PACK_MANIFEST.json'),entry=bind(entry),protocol=e['protocol'],code=bind(__file__),inputs=p['inputs'],features_shape=list(x.shape),frames=NEW_FRAMES,occurrences=NEW_UNITS,old_features_reused=p['old_output'],combined_units=OLD_UNITS+NEW_UNITS,combined_occurrences=OLD_OCCURRENCES+NEW_UNITS,new_query_lineage=bind(out/'NEW3200_INPUT_LINEAGE_ONLY.jsonl'),input_RGB_spatial_facts=bind(out/'INPUT_RGB_QUERY_FACTS400_INPUT_ONLY.jsonl'),model_state_unchanged=True,frozen_encoder_state_before=before,frozen_encoder_state_after=producer.state_sha(encoder),first_batch_repeat_exact=repeated,encoder_calls=calls,max_views=17,Q_outside_never_adds_U=True,Q_inside_units=int(ii.sum()),query_available=int(aa.sum()),factual_U=int((~fa).sum()),context_U=int((~ca).sum()),extra_context_U=int((fa&~ca).sum()),fold_frames=dict(TRAIN=300,VALIDATION=100),fold_occurrences=dict(TRAIN=2400,VALIDATION=800),decoder=dict(Pillow=PILVERSION,numpy=np.__version__),old_feature_array_deserialized_or_copied=False,labels_GT_reference_depth_masks_pose_private_current_V112_reads=0,head_forward_fit_PnP=0,seconds=time.monotonic()-start))
  print(json.dumps(dict(completed=True,receipt=bind(out/'FEATURE_COMPLETE.json'))))
 except BaseException:
  write(out/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 else:run(a.entry)
