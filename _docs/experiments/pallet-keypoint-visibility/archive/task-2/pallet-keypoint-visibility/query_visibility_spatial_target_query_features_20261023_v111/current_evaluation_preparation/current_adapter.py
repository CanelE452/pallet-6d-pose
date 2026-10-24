"""C11 fixed current operator. Generated path never opens actual current/model data."""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import argparse,ast,csv,hashlib,importlib.util,json,math,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PROTOCOL=V/'ROOT_CURRENT_COMPARISON_OPERATOR_BEFORE_INPUT_READ.json'
PROTOCOL_SHA='fc9dd76c816029a6ee3b935ebb141e2229fd26f06c3c2a48e433075c9f8d1cac'
ADMISSION=V/'ROOT_SOURCE_ADMISSION_FOR_FROZEN_CURRENT_EVAL.json'
ADMISSION_SHA='38478176086f78e9476986ee46e151cfb4573d96573bc4fb2fc41a4630ba0c84'
FEATURE_PROTOCOL=V/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
FEATURE_PROTOCOL_SHA='e7f12ad8bf61994ec22fd510839d213564f7dd97933e6a89587f6f1063080645'
PRODUCER=V/'feature_preparation/prepare_spatial_query_features.py'
PRODUCER_SHA='cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e'
HEAD=V/'train_spatial_target_query_head.py';HEAD_SHA='f34ce3b44cc491593bbe6b63fa498c0417a5ba70f54318787094ac0802efa5de'
V94=P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation'
DECODER=V94/'infer_same_query_once.py';DECODER_SHA='53fe3d7031aa9a610638465d12d8c17c0778ba0ed289a6a03e4f3540c0d372bd'
EVALUATOR=V94/'evaluate_saved_only.py';EVALUATOR_SHA='0045abb266c4e3341aa2bf4b02f387f24ec5f04b5c19b74b200000e5e55113b5'
READER=P/'query_visibility_physical_cue_aux_20261023_v91/private_evaluation/infer_native_center_once.py';READER_SHA='df126041c752c02ab5842765fcfcde294cf02ee666f5c604417b812b3991a5bb'
METRIC=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py';METRIC_SHA='d079650b65aef777675af552da61335a28b73f978580413d74d143433019bef6'
CURRENT_KEYS={'current_model_input','prior_input_rows','prior_preparation_seal','prior_inference_seal','prior_input_gate','prepared_inputs','feature_metadata'}
EVAL_KEYS={'evaluation_manifest','manual_provenance','baseline_v91_scores','baseline_v91_inference_seal','historical_v91_metrics','original_baseline_metrics'}
NEW='V111_C11_RAW';NEW_GEO='V111_C11_PLUS_OLD_GEOMETRY';OLD='V91_RAW';OLD_GEO='V91_PLUS_OLD_GEOMETRY';MODELS=('v61','legacy',OLD,OLD_GEO,NEW,NEW_GEO)
BOX_MODE='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def require(x,why):
 if not x:raise ValueError(why)
def same(a,b):return isinstance(a,dict)and isinstance(b,dict)and a.get('path')==b.get('path')and a.get('sha256')==b.get('sha256')
def contains(x,b):return same(x,b)or(isinstance(x,dict)and any(contains(z,b)for z in x.values()))or(isinstance(x,list)and any(contains(z,b)for z in x))
def check(b):require(sha(b['path'])==b['sha256'],'Changed bound bytes: '+b['path']);return Path(b['path'])
def write_new(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(p,expected,name):
 require(sha(p)==expected,'Pinned code changed');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prior_reader():return module(READER,READER_SHA,'c11_pinned_prior_reader')
def metric_base():return module(METRIC,METRIC_SHA,'c11_pinned_metric_core')

def pure_core():
 """Exact V94 decoder/evaluator function AST; imports no current readers or data."""
 require(sha(DECODER)==DECODER_SHA and sha(EVALUATOR)==EVALUATOR_SHA,'Pinned readout changed')
 ns=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(require=require),NEW=NEW,NEW_GEO=NEW_GEO,OLD=OLD,OLD_GEO=OLD_GEO,MODELS=MODELS,Path=Path,csv=csv)
 names={'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'}
 functions=[n for n in ast.parse(EVALUATOR.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 functions += [n for n in ast.parse(DECODER.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='decode_rows']
 require(len(functions)==11,'Pure reader function coverage changed')
 exec(compile(ast.Module(body=functions,type_ignores=[]),str(EVALUATOR)+'::pure_V94_functions','exec'),ns)
 return SimpleNamespace(**ns)

def initial_context_box(points):
 q=np.asarray(points,np.float64)
 require(q.shape==(8,2),'Exactly frozen q8 required')
 good=np.isfinite(q).all(1)&~(q==[-1.,-1.]).all(1)
 if not good.any():return None
 lo=q[good].min(0);hi=q[good].max(0)
 return np.r_[lo,hi] if (hi>lo).all() else None

def source_guard(protocol,e,freeze,complete,gates,audit):
 require(same(e['source_freeze'],protocol['source_freeze'])and same(e['source_gates'],protocol['source_gates']),'Wrong preselected source')
 require(freeze.get('completed')is True and freeze['selected_update']==1800 and freeze['threshold_logit']==0,'Frozen selected head changed')
 require(same(freeze['checkpoint'],e['checkpoint'])and same(freeze['trainer'],bind(HEAD)),'Checkpoint/trainer mismatch')
 require(freeze.get('current_used_for_fit_or_selection')is False and freeze.get('PnP')==0,'Source selection leakage')
 require(complete.get('completed')is True and complete['optimizer_updates']==2000 and same(complete['model_freeze'],e['source_freeze']),'Incomplete sourcefit')
 require(gates.get('completed')is True and gates.get('all_gates_pass')is True and len(gates['gates'])==12 and all(x['PASS']is True for x in gates['gates'].values()),'12 sourcegates required')
 require(audit.get('completed')is True and audit.get('PASS')is True and not audit.get('blockers',[])and(contains(audit,e['source_freeze'])or contains(audit,e['source_fit_complete'])),'Independent source review missing')

def validate_entry(path,stage,code_path=None):
 e=read(path);require(e.get('approved')is True and e.get('stage')==stage,'Root stage approval required')
 require(e.get('current_used_for_selection')is False and e.get('training_allowed')is False and e.get('PnP_allowed')is False,'Forbidden scope')
 require(e.get('context_box_mode')==BOX_MODE,'Frozen fallback only; no GT/legacybox')
 for k in('code','protocol','generated','source_freeze','source_fit_complete','source_gates','source_audit','source_admission'):check(e[k])
 require(same(e['code'],bind(__file__ if code_path is None else code_path))and same(e['protocol'],bind(PROTOCOL))and sha(PROTOCOL)==PROTOCOL_SHA,'Code/operator changed')
 if stage=='EVALUATE_SAVED':require(same(e['common_code'],bind(__file__)),'Shared inference/reader changed')
 proof=read(e['generated']['path']);require(proof.get('PASS')is True and contains(proof,e['code']),'Generated proof changed')
 require(sha(PRODUCER)==PRODUCER_SHA and sha(HEAD)==HEAD_SHA,'Source implementation changed')
 source_guard(read(PROTOCOL),e,read(e['source_freeze']['path']),read(e['source_fit_complete']['path']),read(e['source_gates']['path']),read(e['source_audit']['path']))
 require(same(e['source_admission'],bind(ADMISSION))and sha(ADMISSION)==ADMISSION_SHA and sha(FEATURE_PROTOCOL)==FEATURE_PROTOCOL_SHA,'Source admission/feature protocol changed')
 admission=read(ADMISSION);require(admission['completed']and admission['frozen_model_admitted_for_once_current_comparison']and admission['source_all12_gates_pass']and same(admission['source_model_freeze'],e['source_freeze'])and same(admission['source_saved_review'],e['source_audit']),'Source admission mismatch')
 require(set(e['bindings'])==CURRENT_KEYS|EVAL_KEYS,'Explicit current/eval bindings required')
 # No current descriptor SHA/body was opened above; all are deferred until source PASS.
 return e,e['bindings']

def infer(entry):
 e,b=validate_entry(entry,'INFER');out=Path(e['output_directory']);out.mkdir(parents=True,exist_ok=False)
 write_new(out/'STARTED.json',dict(entry=bind(entry),current_GT_read=False,box_mode=BOX_MODE,optimizer_updates=0,PnP=0))
 import torch
 from PIL import Image,__version__ as PILVERSION
 require(torch.__version__=='2.1.1+cu118'and torch.cuda.is_available(),'Frozen source runtime required')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 old=module(READER,READER_SHA,'c11_prior_reader');data,base,q,L=old.load_current(b)
 frames=old.read(b['prepared_inputs']['path']);producer=module(PRODUCER,PRODUCER_SHA,'c11_spatial_math')
 fp=read(FEATURE_PROTOCOL)
 require(Path(sys.executable).resolve()==Path(fp['runtime']['executable']).resolve(),'Exact source executable required')
 for k in('weight','C8_preprocess','C9_target_preprocess','target_crop_code'):check(fp[k])
 for a in fp['official_source_bindings']:check(a)
 c8=producer.mod(fp['C8_preprocess']['path'],'c11_current_local_preprocess');c9=producer.mod(fp['C9_target_preprocess']['path'],'c11_current_target_preprocess');tm=producer.mod(fp['target_crop_code']['path'],'c11_current_target_crop')
 sys.path.insert(0,fp['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
 encoder=dinov2_vits14_reg(pretrained=False);encoder.load_state_dict(torch.load(fp['weight']['path'],map_location='cpu',weights_only=True),strict=True);encoder=encoder.cuda().float().eval().requires_grad_(False)
 require(sum(x.numel()for x in encoder.parameters())==22058112,'DINO architecture changed')
 encoder_before=producer.state_sha(encoder);require(encoder_before==read(fp['asset_runtime']['path'])['state_before_SHA256'],'Source encoder state changed');headmod=module(HEAD,HEAD_SHA,'c11_current_head')
 check(e['checkpoint']);payload=torch.load(e['checkpoint']['path'],map_location='cpu',weights_only=True)
 require(payload['update']==1800 and same(payload['trainer'],bind(HEAD))and same(payload['design'],read(e['source_freeze']['path'])['design']),'Wrong sourcecheckpoint')
 head=headmod.SpatialTargetQueryHead().cuda().eval().requires_grad_(False);head.load_state_dict(payload['state_dict'],strict=True)
 head_before=producer.state_sha(head);features=[];facts=[];context=[];local=[];inside=[];encoder_calls=0;repeat=False;start=time.monotonic()
 for i,frame in enumerate(frames):
  qs=q[i*8:(i+1)*8];box=initial_context_box(qs);rgbsha=None
  if box is None:target=np.zeros((224,224,3),np.float32);tv=np.zeros((224,224),bool)
  else:
   require(sha(frame['image'])==frame['image_sha256'],'Full context image changed')
   rgb=np.array(Image.open(frame['image']).convert('RGB'),dtype=np.uint8,copy=True);require(list(rgb.shape[:2])==frame['raw_hw'],'Native frameHW changed')
   rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();target,tv=tm.sample_target_context(rgb,box)
  lr=data['rgb96'][i*8:(i+1)*8];lv=data['crop_valid96'][i*8:(i+1)*8]
  localviews,_=c8.preprocess_views(torch.as_tensor(np.ascontiguousarray(lr),device='cuda'),torch.as_tensor(np.ascontiguousarray(lv),device='cuda'))
  targetview=c9.preprocess_target(torch.as_tensor(target.transpose(2,0,1)[None],device='cuda'),torch.as_tensor(tv[None,None],device='cuda'))
  views=torch.cat((targetview,localviews),0)
  with torch.inference_mode():
   raw=encoder.forward_features(views);encoder_calls+=1
   if not repeat:
    again=encoder.forward_features(views);encoder_calls+=1
    require(torch.equal(raw['x_norm_clstoken'],again['x_norm_clstoken'])and torch.equal(raw['x_norm_patchtokens'],again['x_norm_patchtokens']),'DINO first repeat differs');repeat=True
   cls,patch=raw['x_norm_clstoken'],raw['x_norm_patchtokens'];fra=np.array([[np.float32(tv.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]for v in lv],np.float32);c4=lv[:,:,47:49,47:49].all((1,2,3))
   xx,ii=producer.assemble(cls[:1].repeat_interleave(8,0),patch[:1].repeat_interleave(8,0),cls[1:9],patch[1:9],cls[9:],qs,[box if box is not None else[np.nan]*4]*8,fra,c4)
  features.append(xx.cpu().numpy());context.extend([bool(tv.any())]*8);local.extend(c4.tolist());inside.extend(ii.tolist())
  facts.append(dict(frame_index=i,id=frame['id'],image=dict(path=frame['image'],sha256=frame['image_sha256']),raw_hw=frame['raw_hw'],decoded_RGB_sha256=rgbsha,box_mode=BOX_MODE,box_xyxy=None if box is None else box.tolist(),context_support_any=bool(tv.any()),target_valid_fraction=float(np.float32(tv.mean())),target_RGB224_float32_sha256=hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),RGB96_original_cached_sha256=hashlib.sha256(np.ascontiguousarray(lr).tobytes()).hexdigest(),valid96_original_cached_sha256=hashlib.sha256(np.ascontiguousarray(lv).tobytes()).hexdigest(),Q_inside=ii.tolist()))
  if(i+1)%25==0:print(json.dumps(dict(frames=i+1,total=223,seconds=time.monotonic()-start)),flush=True)
 x=np.concatenate(features).astype(np.float32);prior=np.array([r['valid']for r in base]);local=np.asarray(local,bool);context=np.asarray(context,bool);inside=np.asarray(inside,bool);op=prior&local&context
 require(x.shape==(1784,1928)and np.isfinite(x).all(),'Feature output malformed')
 with torch.inference_mode():
  a=head(torch.as_tensor(x,device='cuda'));again=head(torch.as_tensor(x,device='cuda'));require(torch.equal(a,again),'Head repeat differs');rawz=a.cpu().numpy();z=np.where(op,rawz,0).astype(np.float32);pv=torch.sigmoid(-torch.as_tensor(z,device='cuda')).cpu().numpy()
 require(encoder_before==producer.state_sha(encoder)and head_before==producer.state_sha(head),'Frozen state changed')
 rows=pure_core().decode_rows(base,z,pv,op)
 for i,r in enumerate(rows):r.update(context_available=bool(context[i]),local_central4_available=bool(local[i]),context_Q_inside=bool(inside[i]),context_box_mode=BOX_MODE,context_bbox_origin='INITIAL_Q8_DERIVED_NOT_SELECTED_DETECTION')
 valid=np.array([r['valid']for r in rows]);require(prior.sum()==1739 and not np.any(valid&~prior),'Prior45U changed')
 with(out/'REAL_QUERY_SCORES.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'RAW_QUERY_SCALAR_OUTPUTS.npz',query_logit=z,raw_head_logit_before_U=rawz,query_available=op,prior_valid=prior,valid=valid,p_visible_FP32=pv,features1928=x,context_available=context,local_central4_available=local,context_Q_inside=inside,query_xy=data['query_xy'],side_L=data['side_L'],original_query_xy=q,original_side_L=L,id=np.array([r['id']for r in base]),native_corner=np.array([r['native_corner']for r in base],np.int8))
 with(out/'CONTEXT_FRAME_FACTS_INPUT_ONLY.jsonl').open('x')as f:
  for r in facts:f.write(json.dumps(r,allow_nan=False)+'\n')
 write_new(out/'INPUT_PREPARATION_AUDIT.json',dict(completed=True,frames=223,slots=1784,prior_valid=1739,prior_U=45,original_q_L_order_cachedRGB96_valid_exact=True,prior_artifact_bindings={k:b[k]for k in CURRENT_KEYS},box_mode=BOX_MODE,NPZ_box_xyxy_deserialized=False,selected_detection_bbox_verified=False,box_clipping_expansion_GT_reference_used=False,context_available_N=int(context.sum()),local_central4_available_N=int(local.sum()),extra_operator_U=int((prior&~op).sum()),Q_outside_not_added_U=True,Q_inside_N=int(inside.sum()),original_P3_used_by_classifier=False,context_full_RGB_decodes=sum(r['decoded_RGB_sha256']is not None for r in facts),decoder=dict(Pillow=PILVERSION,numpy=np.__version__),input_fields_for_neural_head='Image-only1928features; no ID/native/GT/reference/geometry',current_GT_read=False))
 check(e['checkpoint']);check(fp['weight'])
 write_new(out/'INFERENCE_SEAL.json',dict(completed=True,frames=223,slots=1784,prior_valid=1739,prior_U=45,valid=int(valid.sum()),U=int((~valid).sum()),additional_U=int((prior&~valid).sum()),original45U_preserved=True,entry=bind(entry),protocol=e['protocol'],source_model_freeze=e['source_freeze'],source_gate_results=e['source_gates'],source_postfit_audit=e['source_audit'],checkpoint=e['checkpoint'],code=bind(__file__),preparation=bind(out/'INPUT_PREPARATION_AUDIT.json'),outputs=[bind(out/n)for n in('REAL_QUERY_SCORES.csv','RAW_QUERY_SCALAR_OUTPUTS.npz','CONTEXT_FRAME_FACTS_INPUT_ONLY.jsonl')],raw_E_threshold_logit=0.,D_rule='z<0',probability_is_display_only=True,rounded_pvisible_half_vs_rawsign_mismatch_N=int((valid&((pv>=.5)!=(z<0))).sum()),model_unchanged=True,encoder_state_SHA256=encoder_before,head_state_SHA256=head_before,first_encoder_repeat_and_head_repeat_bitexact=True,frozen_DINO_forward_calls=encoder_calls,head_forward_calls=2,runtime_torch=torch.__version__,current_used_for_selection=False,current_GT_read=False,new_initializer=0,optimizer_updates=0,PnP=0,geometry_route_separate=True,seconds=time.monotonic()-start))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--infer-entry',required=True);a=ap.parse_args();infer(a.infer_entry)
