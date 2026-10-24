"""C10 input-only producer. Import/--prepare never read pixels/features/weights.
Only root-approved --entry may construct/run the frozen ImageNet encoder.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
C8=P/'query_visibility_frozen_rgb_head_20261023_v105';C9=P/'query_visibility_target_relative_rgb_head_20261023_v108'
INPUTS=P/'query_visibility_ycb_real_surface_source_20261023_v109/query_preparation/INPUTS1000_QUERY_EMULATION_ONLY.jsonl'
LOCAL_MATH=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
TARGET_MATH=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
C8_FACTORY=C8/'frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'
C9_FACTORY=C9/'context_feature_preparation/prepare_target_context_features.py'
OLD_MAPPING=C9/'context_feature_preparation/OCCURRENCE_MAPPING_INPUT_ONLY.npz'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';OUT=V/'natural_surface_rgb_features';BATCH_ROWS=4
INPUT_FIELDS={'input_id','instance_row_index','image','raw_hw','initial_points','roi_side','query_box_xyxy','split','source','source_emulation','native_mapping'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for s in iter(lambda:f.read(1<<20),b''):h.update(s)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pure_local_math():
 tree=ast.parse(LOCAL_MATH.read_text());ns={'np':np};parts=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('crop','query_validity')];assert len(parts)==2
 exec(compile(ast.Module(body=parts,type_ignores=[]),str(LOCAL_MATH)+'::pure_crop_and_query_only','exec'),ns)
 return ns['crop'],ns['query_validity']

def validate_inputs(rs):
 assert len(rs)==1000 and len({r['input_id']for r in rs})==1000
 for r in rs:
  assert set(r)==INPUT_FIELDS and set(r['image'])=={'path','sha256'}
  assert np.asarray(r['initial_points']).shape==(8,2)and np.isfinite(r['initial_points']).all()
  assert np.isfinite(r['roi_side'])and r['roi_side']>0 and r['raw_hw']==[480,640]
  b=np.asarray(r['query_box_xyxy'],float);assert b.shape==(4,)and np.isfinite(b).all()and(b[2:]>b[:2]).all()
  assert r['source_emulation']is True and r['native_mapping']=='UNKNOWN'and r['split']in('TRAIN','VALIDATION')
 return rs

def compose1541(target512,local1026,targetfraction,cosTC=None):
 n=len(local1026)
 if target512.shape!=(n,512)or local1026.shape!=(n,1026)or targetfraction.shape!=(n,):raise ValueError('Feature shape')
 if any(not torch.isfinite(x).all()for x in(target512,local1026,targetfraction)):raise ValueError('Nonfinite feature')
 full=local1026[:,:512];centre=local1026[:,512:1024]
 tc=F.cosine_similarity(target512,centre,dim=1,eps=1e-12)if cosTC is None else cosTC
 tf=F.cosine_similarity(target512,full,dim=1,eps=1e-12)
 assert tc.shape==(n,)
 out=torch.cat((target512,full,centre,tc[:,None],tf[:,None],targetfraction[:,None],local1026[:,1024:1026]),1)
 assert out.shape==(n,1541)and torch.isfinite(out).all()
 return out

def prepare(design):
 import torchvision
 assert torch.__version__=='2.1.1+cu118'and torchvision.__version__=='0.16.1+cu118'
 g=H/'GENERATED_FEATURE_TESTS.json';assert read(g)['PASS']and read(g)['code']==bind(__file__)
 inputs=validate_inputs(rows(INPUTS));line=[]
 for i,r in enumerate(inputs):
  for k,q in enumerate(r['initial_points']):line.append(dict(global_occurrence_index=33984+i*8+k,input_unit_index=3652+i*8+k,query_row_index=i,logical_query_index=k,query_id=r['input_id']+'::q'+str(k),input_id=r['input_id'],image=r['image'],raw_hw=r['raw_hw'],query_xy=q,L=r['roi_side'],query_box_xyxy=r['query_box_xyxy'],split=r['split'],source=r['source'],source_emulated_not_initializer=True))
 lf=H/'NEW8000_INPUT_LINEAGE_ONLY.jsonl'
 with lf.open('x')as f:
  for r in line:f.write(json.dumps(r,separators=(',',':'))+'\n')
 c8r=read(C8/'frozen_rgb_features/FEATURE_COMPLETE.json');c9r=read(C9/'target_context_features/FEATURE_COMPLETE.json');assert c8r['completed']and c8r['PASS']and c9r['completed']and c9r['PASS']
 c8p=read(C8/'frozen_rgb_feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json')
 p=dict(completed=True,design=bind(design),code=bind(__file__),generated=bind(g),new_inputs=bind(INPUTS),new_lineage=bind(lf),local_math=bind(LOCAL_MATH),target_math=bind(TARGET_MATH),C8_factory=bind(C8_FACTORY),C9_factory=bind(C9_FACTORY),C8_receipt=bind(C8/'frozen_rgb_features/FEATURE_COMPLETE.json'),C9_receipt=bind(C9/'target_context_features/FEATURE_COMPLETE.json'),C8_features=c8r['output'],C9_features=c9r['output'],old_mapping=bind(OLD_MAPPING),pretrained=c8p['pretrained'],weight_provenance=c8p['weight_download_provenance'],torchvision_resnet_code=c8p['torchvision_resnet_code'],schema=dict(features='float32[11652,1541]',occurrences=41984,old_occurrences=33984,new_occurrences=8000,old_units=3652,new_units=8000,new_contexts=1000,layout=['target512','localfull512','central512','cos_target_central','cos_target_localfull','target_fraction','local_fraction','central_fraction'],old_components='Literal saved C8full+central+fractions and C9target+targetfraction+cosTC; only new cosTF arithmetic, no old CNN rerun.',new_components='Exact V90 crop/query_validity pureAST reused by V94/V104; C8 full96+central32 preprocessing/frozenImageNet ResNet; V107 rect224/C9target preprocessing. Counts/denominators for fractions computed on CPU then castF32.',availability='Old=factual_old AND context_old exactly as C9. New=q continuous&nearest excluding(-1,-1) AND RGB96 central4native-valid AND target-context-any-supported. Partial crops retained; unavailable is U, never negative.'),batch_input_rows=BATCH_ROWS,output=str(OUT),runtime=dict(python=str(Path(sys.executable).resolve()),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,TF32=False,AMP=False,deterministic=True),forbidden_reads=['teacher_packets','projected_GT_reference','GT_masks','depth','pose','native_semantic_class','labels','current223'],preparation_source_pixels_features_parameter_load_forward_fit=0)
 write(PROTOCOL,p);write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',dict(approved_actual_source_feature_extraction=False,source_only=True,current_allowed=False,fit_allowed=False,PnP_allowed=False,code=bind(__file__),protocol=bind(PROTOCOL),design=bind(design),generated=bind(g),output=str(OUT)))
 print(json.dumps(dict(protocol=bind(PROTOCOL),entry_template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),command=f'{sys.executable} {Path(__file__).resolve()} --entry ROOT_APPROVED_ENTRY.json')),flush=True)

def run(entry):
 e=read(entry);assert e['approved_actual_source_feature_extraction']is True and e['source_only']and e['current_allowed']is False and e['fit_allowed']is False and e['PnP_allowed']is False
 for k in('code','protocol','design','generated'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)
 p=read(PROTOCOL);assert p['code']==e['code']and p['design']==e['design']and p['generated']==e['generated']and read(check(e['generated']))['PASS']and not OUT.exists()
 assert Path(e['output']).resolve()==OUT.resolve()
 import torchvision
 from PIL import Image,__version__ as PILVERSION
 assert str(Path(sys.executable).resolve())==p['runtime']['python']and torch.__version__==p['runtime']['torch']and torchvision.__version__==p['runtime']['torchvision']and torch.cuda.is_available()
 for k in('new_inputs','new_lineage','local_math','target_math','C8_factory','C9_factory','C8_receipt','C9_receipt','C8_features','C9_features','old_mapping','pretrained','weight_provenance','torchvision_resnet_code'):check(p[k])
 inputs=validate_inputs(rows(p['new_inputs']['path']));crop,qvalid=pure_local_math();targetmath=module(p['target_math']['path'],'c10_target_math');c8=module(p['C8_factory']['path'],'c10_c8_factory');c9=module(p['C9_factory']['path'],'c10_c9_factory')
 with np.load(p['C8_features']['path'],allow_pickle=False)as f:oldlocal=np.array(f['features'],copy=True);oldunit=np.array(f['occurrence_unit'],copy=True);oldfact=np.array(f['query_available'],copy=True)
 with np.load(p['C9_features']['path'],allow_pickle=False)as f:oldtarget=np.array(f['features'],copy=True);oldunit9=np.array(f['occurrence_unit'],copy=True);oldavail=np.array(f['query_available'],copy=True);oldcontext=np.array(f['context_available'],copy=True);oldfact9=np.array(f['factual_query_available'],copy=True)
 assert oldlocal.shape==(3652,1026)and oldtarget.shape==(3652,1027)and np.array_equal(oldunit,oldunit9)and np.array_equal(oldfact,oldfact9)and np.array_equal(oldavail,oldfact&oldcontext)
 assert np.array_equal(oldlocal[:,512:1024],oldtarget[:,512:1024])and np.array_equal(oldlocal[:,1025],oldtarget[:,1026])
 OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(entry=bind(entry),no_silent_restart=True));started=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with torch.inference_mode():oldfeatures=compose1541(torch.as_tensor(oldtarget[:,:512],device='cuda'),torch.as_tensor(oldlocal,device='cuda'),torch.as_tensor(oldtarget[:,1025],device='cuda'),torch.as_tensor(oldtarget[:,1024],device='cuda')).cpu().numpy()
  assert np.array_equal(oldfeatures[:,:512],oldtarget[:,:512])and np.array_equal(oldfeatures[:,512:1536],oldlocal[:,:1024])and np.array_equal(oldfeatures[:,1536],oldtarget[:,1024])and np.array_equal(oldfeatures[:,1538],oldtarget[:,1025])and np.array_equal(oldfeatures[:,1539:],oldlocal[:,1024:])
  encoder=c8.load_imagenet_backbone().cuda().eval().requires_grad_(False);before=c8.state_sha(encoder);torch.cuda.reset_peak_memory_stats();features=[];facts=[];factual=[];contexts=[];repeat=False;calls=0
  for start in range(0,1000,BATCH_ROWS):
   batch=inputs[start:start+BATCH_ROWS];localrgb=[];localvalid=[];targetrgb=[];targetvalid=[];targetfrac=[];fractions=[];batchfacts=[]
   for bi,r in enumerate(batch):
    path=check(r['image']);rgb=np.array(Image.open(path).convert('RGB'),dtype=np.uint8,copy=True);assert list(rgb.shape)==r['raw_hw']+[3]
    tg,tv=targetmath.sample_target_context(rgb,r['query_box_xyxy']);targetrgb.append(tg.transpose(2,0,1));targetvalid.append(tv[None]);targetfrac.append(np.float32(tv.mean()))
    qs=np.asarray(r['initial_points'],np.float64);centre,continuous,nearest=qvalid(qs,r['raw_hw']);fcts=[]
    for k,q in enumerate(qs):
     rr,v=crop(rgb,q,r['roi_side'],96);localrgb.append(rr);localvalid.append(v);fractions.append([np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]);central4=bool(v[:,47:49,47:49].all());a=bool(centre[k]and central4);factual.append(a);contexts.append(bool(tv.any()))
     fcts.append(dict(logical_query_index=k,q_continuous=bool(continuous[k]),q_nearest=bool(nearest[k]),q_center_valid=bool(centre[k]),RGB_central4_valid=central4,factual_query_available=a,context_available=bool(tv.any()),RGB96_sha256=hashlib.sha256(rr.tobytes()).hexdigest(),valid96_sha256=hashlib.sha256(v.tobytes()).hexdigest()))
    batchfacts.append(dict(query_row_index=start+bi,input_id=r['input_id'],image=r['image'],decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),target_RGB224_float32_sha256=hashlib.sha256(np.ascontiguousarray(tg).tobytes()).hexdigest(),target_valid_fraction=float(np.float32(tv.mean())),target_support_any=bool(tv.any()),queries=fcts))
   localrgb=torch.as_tensor(np.stack(localrgb),device='cuda');localvalid=torch.as_tensor(np.stack(localvalid),device='cuda');localviews,_=c8.preprocess_views(localrgb,localvalid)
   targetviews=c9.preprocess_target(torch.as_tensor(np.stack(targetrgb),device='cuda'),torch.as_tensor(np.stack(targetvalid),device='cuda'));allviews=torch.cat((targetviews,localviews),0);nb=len(batch);nl=8*nb
   with torch.inference_mode():
    raw=encoder(allviews);calls+=1
    if start==0:assert torch.equal(raw,encoder(allviews));repeat=True;calls+=1
    localfeatures=c8.assemble_features(raw[nb:],torch.as_tensor(np.asarray(fractions,np.float32),device='cuda'))
    traw=F.normalize(raw[:nb].float(),p=2,dim=1,eps=1e-12);tf=torch.as_tensor(np.asarray(targetfrac,np.float32),device='cuda');traw=torch.where((tf>0)[:,None],traw,torch.zeros_like(traw))
    z=compose1541(traw.repeat_interleave(8,0),localfeatures,tf.repeat_interleave(8))
   features.append(z.cpu().numpy());facts.extend(batchfacts)
   if(start//BATCH_ROWS+1)%25==0:print(json.dumps(dict(input_rows=start+nb,total=1000,seconds=time.monotonic()-started)),flush=True)
  newfeatures=np.concatenate(features).astype(np.float32);assert newfeatures.shape==(8000,1541)and np.isfinite(newfeatures).all()
  factual=np.asarray(factual,bool);contexts=np.asarray(contexts,bool);newavail=factual&contexts
  features=np.concatenate((oldfeatures,newfeatures));unit=np.r_[oldunit,np.arange(3652,11652)].astype(np.int32);fact=np.r_[oldfact,factual];ctx=np.r_[oldcontext,contexts];available=np.r_[oldavail,newavail]
  assert features.shape==(11652,1541)and unit.shape==fact.shape==ctx.shape==available.shape==(41984,)and np.array_equal(available,fact&ctx)
  after=c8.state_sha(encoder);assert before==after and repeat and not any(m.training for m in encoder.modules())and all(not p.requires_grad for p in encoder.parameters())
  output=OUT/'SOURCE_RGB_FEATURES.npz';np.savez_compressed(output,features=features,occurrence_unit=unit,query_available=available,factual_query_available=fact,context_available=ctx,global_occurrence_index=np.arange(41984,dtype=np.int32))
  fp=OUT/'NEW1000_RGB_AND_QUERY_FACTS_INPUT_ONLY.jsonl'
  with fp.open('x')as f:
   for r in facts:f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(output),protocol=e['protocol'],entry=bind(entry),code=e['code'],design=e['design'],new_query_lineage=p['new_lineage'],new_RGB_query_facts=bind(fp),old_C8_receipt=p['C8_receipt'],old_C9_receipt=p['C9_receipt'],features_shape=[11652,1541],occurrences=41984,old_occurrences=33984,new_occurrences=8000,old_CNN_calls=0,new_context_rows=1000,old_components_copied_exact=True,first_batch_repeat_exact=True,model_state_unchanged=True,frozen_encoder_state_before=before,frozen_encoder_state_after=after,encoder_batch_calls=calls,counts=dict(old_factual_U=int((~oldfact).sum()),old_final_U=int((~oldavail).sum()),new_factual_U=int((~factual).sum()),new_context_U=int((~contexts).sum()),new_final_U=int((~newavail).sum())),seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),decoder=dict(Pillow=PILVERSION,numpy=np.__version__),teacher_labels_masks_depth_reference_reads=0,new_head_fit_private_current_PnP=0))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'FEATURE_COMPLETE.json'))),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--design',type=Path);ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:assert a.design is not None;prepare(a.design)
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare --design PATH or --entry ROOT_APPROVED_ENTRY')
