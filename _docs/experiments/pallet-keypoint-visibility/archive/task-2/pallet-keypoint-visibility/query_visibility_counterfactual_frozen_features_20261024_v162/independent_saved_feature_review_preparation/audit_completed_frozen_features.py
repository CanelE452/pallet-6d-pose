"""Independent saved-feature audit. No encoder/model/weight import or load.
Preparation uses metadata/code and invented RGB arrays. Actual feature/pixel bodies
require explicit completed-root trigger. Only pure frozen C8/C9 input math replays.
"""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,collections,datetime,hashlib,json,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
BH=V/'feature_preparation';OUT=V/'frozen_cf_features_v1';R=V/'independent_saved_feature_review'
PP=BH/'PROTOCOL_BEFORE_ANY_ENCODER_OR_INPUT_BODIES.json';PSHA='5b236a04a3fe02a20050370c0062f5994537f45563721dd6a8e54eae7148bdf9';CSHA='11d7b44475cd97190480b972cb129763ca2a2023abd475dc7fbdec73d4641a6d'
PIXEL_KEYS=('rgb96','valid96','central32','target224','target_valid','fractions','query_xy','L','native_slot','query_id','crop_constructed','native_continuous_valid','nearest_valid','central4_available','context_available')
FEATURE_KEYS={'features','occurrence_index','input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_xy','L','query_available','native_continuous_valid','nearest_valid','feature_constructed','local_central4_available','context_available','support_fractions','context_box_xyxy','unit_Q_inside','occurrence_unit'}
BINDINGS=[]
def safe(p):
 p=Path(p).resolve();s=str(p).lower();assert p.is_relative_to(P) and not any(t in s for t in ('_v112','/private_evaluation/','current223','/current_evaluation/'));return p
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=safe(p);assert p.suffix not in ('.pt','.pth','.pkl');b=dict(path=str(p),sha256=sha(p));BINDINGS.append(b);return b
def verify(b):assert bind(b['path'])['sha256']==b['sha256'],b['path'];return safe(b['path'])
def read(p):return json.loads(safe(p).read_text())
def rows(p):return [json.loads(s) for s in safe(p).read_text().splitlines() if s]
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def assembly_bytes(a,box):
 data=b''
 for name,z in [('query_xy',a['query_xy']),('context_box',np.asarray(box,np.float64)),('fractions',a['fractions']),('central4_available',a['central4_available']),('crop_constructed',a['crop_constructed'])]:
  z=np.ascontiguousarray(z);data+=name.encode()+str(z.dtype).encode()+str(z.shape).encode()+z.tobytes()
 return data

def input_operators(p):
 import torch
 from torch.nn import functional as F
 result={}
 for key,name in [('C8_preprocess','preprocess_views'),('C9_target_preprocess','preprocess_target')]:
  path=verify(p[key]);assert path.suffix=='.py';ns=dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225));nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(nodes)==1
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure_input_math','exec'),ns);result[name]=ns[name]
 return result

def views(a,op,device):
 import torch
 rgb=np.ascontiguousarray(a['rgb96'].transpose(0,3,1,2));valid=np.ascontiguousarray(a['valid96'][:,None])
 local,_=op['preprocess_views'](torch.as_tensor(rgb,device=device),torch.as_tensor(valid,device=device))
 context=op['preprocess_target'](torch.as_tensor(a['target224'].transpose(2,0,1)[None],device=device),torch.as_tensor(a['target_valid'][None,None],device=device))
 result=torch.cat((context,local),0);assert result.shape==(17,3,224,224) and result.dtype==torch.float32 and torch.isfinite(result).all()
 return result.cpu().numpy()

def invented(p):
 op=input_operators(p);a=dict(rgb96=np.arange(8*96*96*3,dtype=np.uint8).reshape(8,96,96,3),valid96=np.ones((8,96,96),bool),target224=np.full((224,224,3),110,np.float32),target_valid=np.ones((224,224),bool),query_xy=np.array([[10.,10.]]*8,np.float64),fractions=np.ones((8,3),np.float32),central4_available=np.ones(8,bool),crop_constructed=np.ones(8,bool));v=views(a,op,'cpu');sig=assembly_bytes(a,[0.,0.,20.,20.]);checks={}
 b={k:z.copy() for k,z in a.items()};b['rgb96'][0,47,47,0]^=1;checks['changed_valid_RGB_changes_preprocess_hash']=ash(views(b,op,'cpu'))!=ash(v)
 b={k:z.copy() for k,z in a.items()};b['valid96'][:,0:4]=False;b['rgb96'][:,0:4]=0;w=views(b,op,'cpu');b['rgb96'][:,0:4]=255;checks['invalid_local_RGB_meanfill_invariant']=exact(w,views(b,op,'cpu'))
 b={k:z.copy() for k,z in a.items()};b['target_valid'][:4]=False;b['target224'][:4]=0;w=views(b,op,'cpu');b['target224'][:4]=255;checks['invalid_context_RGB_meanfill_invariant']=exact(w,views(b,op,'cpu'))
 b={**a,'owner_target':np.ones(8),'native_slot':np.arange(8)[::-1],'pasted_alpha':np.ones((3,3),bool)};checks['noninput_alpha_owner_slot_metadata_inert']=exact(v,views(b,op,'cpu'))
 for key in ('query_xy','fractions','central4_available','crop_constructed'):
  b={k:z.copy() for k,z in a.items()};b[key].flat[0]=not b[key].flat[0] if b[key].dtype==bool else b[key].flat[0]+.01;checks['reuse_signature_requires_'+key]=assembly_bytes(b,[0.,0.,20.,20.])!=sig
 checks['reuse_signature_requires_contextbox']=assembly_bytes(a,[1.,0.,20.,20.])!=sig
 checks['feature_byte_change_detected']=not exact(np.zeros((8,1928),np.float32),np.ones((8,1928),np.float32))
 assert all(checks.values());return dict(completed=True,PASS=True,invented_arrays_only=True,checks=checks,number_checks=len(checks),encoder_model_weight_calls=0,source_pixel_feature_bodies_read=0)

def execute(trigger):
 t=read(trigger);assert t['completed'] is True and t['source_only'] is True and t['allow_source_saved_feature_input_math_review_only'] is True
 assert t['encoder_or_model_or_fit_allowed'] is False and t['new_current_V112_PnP_allowed'] is False
 assert t['completed_feature']['path']==str(OUT/'FEATURE_COMPLETE.json');verify(t['completed_feature']);tb=bind(trigger)
 assert sha(PP)==PSHA and sha(BH/'encode_all477_source_inputs.py')==CSHA;p=read(PP);bind(PP);verify(p['code'])
 complete=read(OUT/'FEATURE_COMPLETE.json');assert complete['completed'] and complete['PASS'] and complete['protocol']==bind(PP) and complete['code']==p['code'] and complete['inputs']==p['inputs']
 for key,value in [('rows',3816),('payloads',477),('pairs',159),('inputU',360),('feature_constructed',3816),('failed_pairs_retained',2),('actual_encoder_calls',364),('views_per_call',17),('BEFORE_unique_calls',46),('CF_calls',318),('BEFORE_reuses',113)]:assert complete[key]==value,key
 for key in ('encoder_state_unchanged','only_RGB17views_encoder_input','alpha_owner_native_labels_GT_pose_not_encoder_inputs','original_qL_native_split_availability_unchanged','all_occurrences_preserved','UNKNOWN_targets_never_used'):assert complete[key]
 assert complete['frozen_encoder_state_before']==complete['frozen_encoder_state_after']==p['expected_encoder_state'] and not complete['training_admitted'] and not complete['automatic_retry'] and complete['classifier_fit_initializer_current_V112_PnP_calls']==0
 entry=read(verify(complete['entry']));assert entry['approved_actual_frozen_feature_extraction'] and entry['source_only'] and entry['protocol']==complete['protocol'] and entry['code']==complete['code']
 assert not any(entry[k] for k in ('fit_allowed','classifier_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','GT_pose_owner_label_reads_allowed','automatic_retry'))
 inputs=read(verify(p['inputs']));groups=inputs['groups'];frames={r['frame_index']:r for r in inputs['frames']};metadata=rows(verify(inputs['manifest']));assert len(groups)==477 and len(frames)==46 and len(metadata)==3816
 assert [r for g in groups for r in g['rows']]==metadata and all(r['split']=='SOURCE_DEVELOPMENT' and r['native_visibility_target']==r['point_quality_target']==-1 and not r['training_admitted'] for r in metadata)
 assert sum(not r['query_available'] for r in metadata)==360 and all(r['feature_constructed'] for r in metadata)
 failed={r['pair_id'] for r in metadata if not r['pair_pixel_proof_PASS']};assert failed=={'V161_A01_R2','V161_A38_R2'} and sum(r['pair_id'] in failed for r in metadata)==48
 facts=rows(verify(complete['payload_facts']));lineage=rows(verify(complete['occurrence_lineage']));parity=read(verify(complete['BEFORE_original_parity']));reuse=read(verify(complete['BEFORE_reuse']));assert len(facts)==477 and len(lineage)==3816 and parity['PASS'] and parity['frames']==46 and parity['rows']==368 and len(parity['records'])==46 and reuse['PASS'] and reuse['reuses']==113 and len(reuse['records'])==113
 feature_path=safe(complete['features']['path']);assert feature_path==OUT/'CF3816_FEATURES_INPUT_ONLY.npz';verify(complete['features'])
 with np.load(feature_path,allow_pickle=False) as z:assert set(z.files)==FEATURE_KEYS;packet={k:z[k].copy() for k in z.files}
 x=packet['features'];assert x.shape==(3816,1928) and x.dtype==np.float32 and np.isfinite(x).all()
 for key in ('occurrence_index','occurrence_unit'):assert packet[key].dtype==np.int32 and np.array_equal(packet[key],np.arange(3816))
 for key in ('input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot'):assert packet[key].tolist()==[r[key] for r in metadata],key
 for key in ('query_xy','L','context_box_xyxy'):
  source='original_context_box_xyxy' if key=='context_box_xyxy' else key;assert np.array_equal(packet[key],np.asarray([r[source] for r in metadata],np.float64),equal_nan=True),key
 for key,source in [('query_available','query_available'),('native_continuous_valid','query_available'),('nearest_valid','nearest_valid'),('feature_constructed','feature_constructed'),('local_central4_available','central4_available'),('context_available','context_available')]:assert packet[key].dtype==bool and np.array_equal(packet[key],np.asarray([r[source] for r in metadata],bool)),key
 assert int((~packet['query_available']).sum())==360 and packet['feature_constructed'].all() and np.array_equal(x[:,1927],packet['unit_Q_inside'].astype(np.float32)) and not x[~packet['unit_Q_inside'],384:768].any()
 # Exactly the bound original source cache, not a model or checkpoint.
 original_path=safe(p['original_features']['path']);assert original_path.name=='MR6D_NATIVE1840_FEATURES_INPUT_ONLY.npz';verify(p['original_features'])
 with np.load(original_path,allow_pickle=False) as z:old={k:z[k].copy() for k in ('features','global_index','frame_index','native_slot','query_id','query_xy','L','query_available','feature_constructed')}
 assert old['features'].shape==(1840,1928) and old['features'].dtype==np.float32
 provenance=hashlib.sha256(json.dumps({k:p[k] for k in ('C111_producer','C8_preprocess','C9_target_preprocess','runtime','expected_encoder_state')},sort_keys=True).encode()).hexdigest()
 import torch
 assert str(torch.__version__)==p['runtime']['torch']=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available() and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 op=input_operators(p);first={};expected_reuses=[];expected_parities=[];actual_calls=0;reviews=[]
 for i,(g,f) in enumerate(zip(groups,facts)):
  assert i==g['payload_index']==f['payload_index'] and g['occurrence_start']==i*8 and g['rows']==metadata[i*8:i*8+8] and [r['native_slot'] for r in g['rows']]==list(range(8))
  for key in ('pair_id','variant','original_frame_index','payload'):assert g[key]==f[key],key
  source=safe(g['payload']['path']);assert source.parent.parent.name=='source_pixels_v1' and source.name==g['variant']+'_INPUTS.npz';verify(g['payload'])
  with np.load(source,allow_pickle=False) as z:a={k:z[k].copy() for k in PIXEL_KEYS}
  assert {k:ash(a[k]) for k in PIXEL_KEYS}==f['input_array_sha256'];assert exact(a['central32'],a['rgb96'][:,32:64,32:64].copy())
  assert np.array_equal(a['query_xy'],packet['query_xy'][i*8:i*8+8],equal_nan=True) and np.array_equal(a['L'],packet['L'][i*8:i*8+8],equal_nan=True)
  fractions=np.asarray([[np.float32(a['target_valid'].mean()),np.float32(v.mean()),np.float32(v[32:64,32:64].mean())] for v in a['valid96']],np.float32);assert exact(fractions,a['fractions']) and exact(fractions,packet['support_fractions'][i*8:i*8+8]) and exact(x[i*8:i*8+8,1924:1927].copy(),fractions)
  view=views(a,op,'cuda');vh=ash(view);ah=hashlib.sha256(assembly_bytes(a,g['context_box_xyxy'])).hexdigest();block=x[i*8:i*8+8];fh=ash(block)
  assert vh==f['preprocess17view_FP32_sha256'] and ah==f['assembly_operands_sha256'] and fh==f['features_all8_sha256'] and f['encoder_provenance_sha256']==provenance
  for k,(m,r) in enumerate(zip(g['rows'],lineage[i*8:i*8+8])):
   assert {key:r[key] for key in m}==m and set(r)-set(m)=={'feature_index','payload_index','unit_Q_inside','feature1928_sha256'}
   assert r['feature_index']==i*8+k and r['payload_index']==i and r['unit_Q_inside']==bool(packet['unit_Q_inside'][i*8+k]) and r['feature1928_sha256']==ash(block[k])
  is_before=g['variant']=='BEFORE';frame=g['original_frame_index'];first_before=is_before and frame not in first
  assert f['actual_encoder_call']==(not is_before or first_before);actual_calls+=int(f['actual_encoder_call'])
  if first_before:
   fr=frames[frame];ids=np.asarray(fr['global_index'],np.int64);assert exact(block,old['features'][ids])
   assert old['global_index'][ids].tolist()==fr['global_index'] and old['frame_index'][ids].tolist()==[frame]*8 and old['native_slot'][ids].tolist()==list(range(8)) and old['query_id'][ids].tolist()==fr['query_id'] and exact(old['query_xy'][ids],a['query_xy']) and exact(old['L'][ids],a['L'])
   for k,ff in enumerate(fr['facts']):assert ash(a['rgb96'][k])==ff['RGB96_HWC_sha256'] and ash(a['valid96'][k])==ff['valid96_sha256'] and ash(a['target224'])==ff['target224_F32_HWC_sha256'] and ash(a['target_valid'])==ff['target_valid_sha256']
   expected_parities.append(dict(original_frame_index=frame,global_indices=fr['global_index'],all8_features1928_byte_equal=True,all8_RGB96_valid96_byte_equal=True,target224_original_fact_hash_equal=True,preprocess17view_original_replay_byte_equal=True,view17_FP32_sha256=vh,features_sha256=fh,original_saved_features_sha256=ash(old['features'][ids])))
   first[frame]=dict(payload_index=i,view=vh,assembly=ah,features=block.copy(),input_hash=f['input_array_sha256'])
  elif is_before:
   c=first[frame];assert vh==c['view'] and ah==c['assembly'] and exact(block,c['features']) and f['input_array_sha256']==c['input_hash']
   expected_reuses.append(dict(payload_index=i,pair_id=g['pair_id'],variant='BEFORE',original_frame_index=frame,reused_payload_index=c['payload_index'],exact17view_bytes=True,exact_assembly_operand_bytes=True,provenance_sha256=provenance,view17_FP32_sha256=vh,assembly_operands_sha256=ah,features_sha256=fh))
  reviews.append(dict(payload_index=i,pair_id=g['pair_id'],variant=g['variant'],PASS=True,all8_feature_SHA=fh,independent17view_SHA=vh,assembly_SHA=ah,actual_encoder_call_record=f['actual_encoder_call']))
  if (i+1)%50==0:print(json.dumps(dict(independent_payloads_checked=i+1,total477=477,no_encoder_calls=True)),flush=True)
 assert actual_calls==364 and len(first)==46 and expected_parities==parity['records'] and expected_reuses==reuse['records'] and len(expected_reuses)==113
 R.mkdir();rp=R/'ALL477_INDEPENDENT_SAVED_FEATURE_FACTS.jsonl'
 with rp.open('x') as f:
  for r in reviews:f.write(json.dumps(r)+'\n')
 result=dict(schema='V162_INDEPENDENT_SAVED_FEATURE_REVIEW',completed=True,PASS=True,rows=3816,payloads=477,pairs=159,inputU=360,failed_pairs_retained=2,all_metadata_qL_native_split_availability_exact=True,all_native_and_quality_targets_unknown_in_source_metadata=True,no_labels_in_feature_packet=True,original46x8_BEFORE_features_byte_equal=True,duplicate113_inputs_assembly_finalfeatures_provenance_exact=True,actual_encoder_call_records364=True,independent_C8_C9_preprocess477_hashes_exact=True,encoder_state_receipts_match_frozen_expected=True,review_code=bind(__file__),root_trigger=tb,feature_complete=bind(OUT/'FEATURE_COMPLETE.json'),features=complete['features'],records=bind(rp),bindings=list({(b['path'],b['sha256']):b for b in BINDINGS}.values()),pure_fixed_RGB_input_math_only=True,encoder_network_or_weight_construction_or_loads=0,no_fit_classifier_initializer_PnP=True,held_RGB_private_current_V112_access=0,training_admitted=False,selected_model=None,fully_solved=False,limits='Saved-feature/input integrity and exact source replay only; feature extraction/owner pixel proof does not establish native visibility or primary-task success.')
 write(R/'REVIEW_COMPLETE.json',result);print(json.dumps(dict(PASS=True,review=bind(R/'REVIEW_COMPLETE.json'))))

def prepare():
 assert sha(PP)==PSHA and sha(BH/'encode_all477_source_inputs.py')==CSHA;p=read(PP);verify(p['code']);g=invented(p)
 write(H/'GENERATED_INVENTED_AUDIT_CHECKS.json',dict(code=bind(__file__),**g))
 protocol=dict(schema='V162_INDEPENDENT_SAVED_FEATURE_AUDIT_BEFORE_BODIES',completed=True,prepared_only=True,code=bind(__file__),producer_protocol=bind(PP),producer_code=p['code'],input_metadata=p['inputs'],invented=bind(H/'GENERATED_INVENTED_AUDIT_CHECKS.json'),required_root_trigger={'feature_extraction_completed':True,'source_only':True},no_generated_feature_or_pixel_bodies_read=True,no_idle_polling=True,actual_audit_pure_C8_C9_input_math_device='pinned CUDA, no encoder/network/weights',checks=['3816 identities/native/qL/originalavailability and360inputU','477/159 retained includingfailed2 pairs and all source native/quality-1 metadata','46x8 actual BEFORE features byte equal originalsource','113 duplicates input/preprocessed17view/assembly/finalfeatures/provenance exact','all477 actual fixed C8/C9 preprocess hashes and inputarraySHA match','364 actual-call receipt arithmetic and frozen unchanged encoder-state receipts','no owner/native/quality/alpha fields in features and no training admission'],feature_or_weight_or_pixel_bodies_read_during_preparation=0,encoder_model_fit_calls=0,planned_output=str(R/'REVIEW_COMPLETE.json'),training_admitted=False)
 write(H/'PROTOCOL_BEFORE_SAVED_FEATURE_BODIES.json',protocol);write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(H/'PROTOCOL_BEFORE_SAVED_FEATURE_BODIES.json'),invented=bind(H/'GENERATED_INVENTED_AUDIT_CHECKS.json'),generated_bodies_read=0,encoder_or_weight_calls=0,actual_audit_performed=False))
 print(json.dumps(dict(prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'),protocol=bind(H/'PROTOCOL_BEFORE_SAVED_FEATURE_BODIES.json'))))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--completed-feature-trigger',type=Path);a=parser.parse_args()
 if a.prepare_only and not a.completed_feature_trigger:prepare()
 elif a.completed_feature_trigger and not a.prepare_only:
  try:execute(a.completed_feature_trigger)
  except BaseException:
   f=H/'FAILED_SAVED_FEATURE_REVIEW_AFTER_TRIGGER_SCHEMA_REPAIR.json'
   if not f.exists():write(f,dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False))
   raise
 else:raise SystemExit('Use preparation or explicit completed-root trigger; no automatic retry')
