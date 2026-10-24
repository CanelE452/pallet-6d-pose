"""INERT V155 preparation for a separate future frozen-current diagnostic phase.

--prepare reads code and JSON binding metadata only. A future --entry requires
a NEW phase policy and resolved protocol outside V155; V155 cannot authorize it.
No source/private tensor bodies or model objects are accessed by --prepare.
"""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
BLOCKING_POLICY=V/'ROOT_POLICY_BEFORE_FROZEN_PROBE_AND_INDEPENDENT_AUXILIARY_CONTROL.json'
BLOCKING_SHA='6de9504b7a875659e4d8702ea707ae14fbaff6ea73e788a33316a1b14426b63b'
OLD=P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149'
OLD_CODE=OLD/'head_operator_preparation/infer_frozen_public_native_benchmark.py'
OLD_SHA='37372228ee5bc55651e8e8d7e818adccae6d41e4515704822106e4d87e901a71'
OLD_PROTOCOL=OLD/'head_operator_preparation/PROTOCOL_BEFORE_ANY_PAYLOAD.json'
OLD_PROTOCOL_SHA='ed3103f99badd7b9a09a2d919485c807eec9c04343d8345a1d9b2c9bb650a568'
SHARED=P/'query_visibility_mr6d_full_public_source_20261024_v153/head_preparation/separated_query_head.py'
SHARED_SHA='cc7c0964a1e0bfbb0d5cc9549fef47e5b72486701dc4ab8919fc934eeaedeb7b'
INDEPENDENT=V/'independent_head_preparation/independent_query_head.py'
SOURCE154=P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154'
CONTROL154=SOURCE154/'runs/C35_C36_SEPARATE_QUERY_FIXED2000/CONTROL_COMPLETE.json'
CONTROL154_SHA='45279eeb0bdb6d21c73d796062a279516f187970134e8c9ffa983e3ce17d476c'
DRAFT=H/'FUTURE_PROTOCOL_DRAFT_NO_CURRENT_AUTHORIZATION.json'
MODELS=('C35','C36','C37','C38')
ARMS=('C35_MATCHED_VISIBILITY_ONLY_CONTROL','C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY',
      'C37_WARM_VISIBILITY_INDEPENDENT_AUX_FIXED8000','C38_FRESH_VISIBILITY_INDEPENDENT_AUX_FIXED8000')
N=1784
FORBIDDEN=('GT_labels','RGB_masks','new_encoder_initializer','fit_optimizer','calibration_threshold_choice',
           'checkpoint_arm_model_choice','admission_or_source_gate_change','V112','PnP','q_L_correction','automatic_retry')
INHERITED=('original_operator','original_operator_protocol','original_C11_feature_seal','original_C16_inference_seal',
 'C16_freeze','C16_checkpoint','C11_head_class','feature_packet','C16_original_raw_packet','feature_values_sha256',
 'input_whitelist','metadata_whitelist','C16_output_score_whitelist','CSV_fields','runtime')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check(b):
 path=Path(b['path']).resolve();assert path.is_relative_to(P)and'_v112'not in str(path).lower()
 assert bind(path)==b;return path
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def source_state_sha(model):
 d=hashlib.sha256()
 for k,t in sorted(model.state_dict().items()):
  a=t.detach().cpu().contiguous().numpy();d.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return d.hexdigest()
def placeholder(path):return dict(path=str(path),sha256='FUTURE_ROOT_BIND_AFTER_BOTH_SOURCE_FITS_AND_INDEPENDENT_AUDIT_COMPLETE')

def future_verified_quality_label(row):
 """Pure prospective predicate, used only for invented fixtures in V155.

 Future evaluator must separately prove manual-reference and native authority;
 manual_click provenance alone is insufficient. Input availability stays separate.
 """
 if row.get('human_state')!='DIRECT_VISIBLE' or row.get('annotation_coordinate_source')!='manual_click':return -1
 if row.get('manual_reference_authority_checked')is not True or row.get('same_native_mapping_checked')is not True:return -1
 try:error=float(row['initial_reference_error_px'])
 except (KeyError,TypeError,ValueError):return -1
 if not np.isfinite(error)or error<0:return -1
 return 1 if error<=3 else 0 if error>20 else -1

def prepare():
 assert sha(BLOCKING_POLICY)==BLOCKING_SHA and sha(OLD_CODE)==OLD_SHA and sha(OLD_PROTOCOL)==OLD_PROTOCOL_SHA
 assert sha(SHARED)==SHARED_SHA and sha(CONTROL154)==CONTROL154_SHA
 blocking=read(BLOCKING_POLICY);assert not blocking['new_private_model_inference_allowed_this_phase']
 previous=read(OLD_PROTOCOL);control=read(CONTROL154);assert control['completed']and control['PASS']and control['selected_model']is None
 fixed={}
 for i,(name,arm)in enumerate(zip(MODELS,ARMS)):
  if i<2:
   fitbinding=control['arms'][arm]['fit'];fit=read(check(fitbinding));freezebinding=fit['model_freeze'];freeze=read(check(freezebinding))
   resultbinding=fit['source_results'];result=read(check(resultbinding))
   assert freeze['parameters']==224643 and freeze['update']==2000 and not freeze['current_allowed']and freeze['selected_model']is None
   checkpoint=freeze['checkpoint'];expected=result['final_state'];status=freeze['source_checks_all_pass']
  else:
   directory=V/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000'/arm
   fitbinding=placeholder(directory/'FIT_COMPLETE.json');freezebinding=placeholder(directory/'MODEL_FREEZE.json');resultbinding=placeholder(directory/'SOURCE_RESULTS.json')
   checkpoint=placeholder(directory/'FINAL_CHECKPOINT.pt');expected='FUTURE_ROOT_BIND_FROZEN_STATE_SHA';status='PENDING_BOTH_SOURCE_FITS_NO_READ_OR_CHOICE'
  fixed[name]=dict(arm=arm,parameters=224643 if i<2 else 471043,fixed_update=2000 if i<2 else 8000,parent_C32_update=8000,
    seed=1540 if i<2 else 1550,auxiliary_weight=i if i<2 else 1,visibility_initialization='C32'if i<3 else'FRESH',
    source_fit=fitbinding,source_freeze=freezebinding,source_results=resultbinding,source_checkpoint=checkpoint,source_state_sha=expected,
    source_checks_all_pass=status,head_code=bind(SHARED)if i<2 else bind(INDEPENDENT),
    class_name='SeparatedQueryHead'if i<2 else'IndependentAuxiliaryQueryHead',threshold_logit=0,source_current_allowed=False)
 # No private binding is hashed/opened here: inherited descriptors only.
 draft=dict(completed=True,prepared_only=True,ready_for_actual_execution=False,code=bind(__file__),blocking_V155_policy=bind(BLOCKING_POLICY),
   old_head_code=bind(OLD_CODE),old_head_protocol=bind(OLD_PROTOCOL),models=list(MODELS),models_fixed=fixed,
   original={k:previous[k]for k in INHERITED},source_controls=[bind(CONTROL154),placeholder(V/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000/CONTROL_COMPLETE.json')],
   required_source_audits=[dict(path=str(SOURCE154/'independent_saved_fit_review/REVIEW.json'),sha256='FUTURE_ROOT_BIND_COMPLETED_AUDIT'),
     placeholder(V/'independent_saved_fit_review/REVIEW.json')],
   mandatory_order='Future separate phase policy+resolved protocol+all source audits -> original C16 ALL1784 singlebatch repeat2 raw/safe/PV/12metadata byteparity DURABLE seal -> only then candidate checkpoint hashes/load and all four full1784repeat2 -> prediction seal -> separately authorized saved label evaluation.',
   all_rows=1784,frames=223,valid=1739,U=45,additional_U=0,G42='CONSTANT_ZERO_ONLY',no_compaction_or_chunking=True,
   outputs='Visibility primary z<0=D / z>=0=E / originalUfirst; auxiliary logits saved separately, never primary visibility gates. Real point quality remains unlabelled.',
   future_eval_models=['v61','legacy','C16',*MODELS],future_primary_cohort='Original common paired1237; same-slot1248 secondary; full1784 inventory/U/S/unknown preserved.',
   future_eval_contract='Reuse original V149/V143/V94 saved mapping/metrics. Replay v61/legacy/C16 historical rows exactly; WOOD/SQUARE D/E confusion, precision, recall andU with paired recover/damage. No model, threshold, checkpoint, qerrorfilter or crop choice. Legacy training overlap and development-exposed current223 must remain disclosed.',
   future_quality_eval=dict(separate_saved_evaluation_entry_required=True,
    authority='First prove original human manual2D reference and same-native mapping are authoritative from original annotation/provenance; manual_click alone does not certify this. No body is read or authority presumed in V155.',
    primary_eligible='DIRECT_VISIBLE plus manual_click, independently documented reference authority and same-native mapping, finite original query-reference error. Preserve fullknown and original-inputvalid denominators separately.',
    target='near<=3px =>1; far>20px=>0; 3<error<=20 unknown; missing/uncertified unknown. No threshold tuning or relabel to score.',
    occluded_amodal='External/self occluded or amodal reference uncertainty is outside primary realquality gold and retained separately; do not auto-label far or good from those coordinates.',
    manual_near_D376='Keep exact original V94.calculate_all predicate: manual_click & DIRECT_VISIBLE & error<=3 & query_valid, WOOD153/SQUARE223 total376. Report unchanged alongside new separate quality analysis.',
    entity='Original frame-level selected-entity decision has one independent frame unit; do not count eight slots as independent frames. Query-quality is per native query and does not imply frame identity.',
    no_fusion='Primary visibility is never gated by entity/quality, manual error, or quality-eligible masks; all1784 and original45U remain.',
    current_quality_gold_created_or_authority_assumed=False),
   current_private_payload_hash_or_body_reads=0,current_actual_model_instantiations_or_calls=0,selected_model=None)
 write(DRAFT,draft)
 template=dict(approved_future_frozen_benchmark=False,benchmark_only_not_admitted=True,source_admission_unchanged=True,selected_model=None,
   code=bind(__file__),draft_origin=bind(DRAFT),future_phase_policy=dict(path='FUTURE_OTHER_PHASE/ROOT_DIAGNOSTIC_POLICY_BEFORE_CURRENT_BODIES.json',sha256='FUTURE_ROOT_SEAL'),
   resolved_protocol=dict(path='FUTURE_OTHER_PHASE/RESOLVED_FOUR_HEAD_PROTOCOL.json',sha256='FUTURE_ROOT_SEAL'),
   output_directory='FUTURE_OTHER_PHASE/frozen_four_head_current_benchmark_v1',**{k:False for k in FORBIDDEN})
 write(H/'FUTURE_ROOT_ENTRY_TEMPLATE_NOT_AUTHORIZED.json',template)
 # Invented arrays only: byte identity and invalid-slot decoder; no actual model objects.
 x=np.array([0.,-0.,np.nan],np.float32);y=x.copy();assert eq(x,y)and not eq(x,np.array([-0.,-0.,np.nan],np.float32))
 raw=np.array([-100.,0.,1.],np.float32);av=np.array([False,True,True]);assert np.where(av,(raw>=0).astype(np.int8),-1).tolist()==[-1,1,1]
 q=dict(human_state='DIRECT_VISIBLE',annotation_coordinate_source='manual_click',manual_reference_authority_checked=True,same_native_mapping_checked=True)
 assert [future_verified_quality_label(dict(q,initial_reference_error_px=v))for v in(0,3,3.1,20,20.1,float('nan'))]==[1,1,-1,-1,0,-1]
 assert future_verified_quality_label(dict(q,human_state='EXTERNAL_OCCLUDED',initial_reference_error_px=30))==-1
 assert future_verified_quality_label(dict(q,same_native_mapping_checked=False,initial_reference_error_px=30))==-1
 assert future_verified_quality_label(dict(q,manual_reference_authority_checked=False,initial_reference_error_px=0))==-1
 proof=dict(completed=True,PASS=True,invented_arrays_only=True,strict_signedzero_and_nan_bytes=True,Ufirst_raw0=True,no_actual_models_instantiated=True,private_payload_hash_or_body_reads=0,
    realquality_prospective_visible_manual_authority_native_mapping_and3_20_boundaries_checked=True,occluded_uncertified_reference_never_primary_quality_gold=True)
 write(H/'GENERATED_INERT_GUARDS.json',proof)
 ready=dict(completed=True,prepared_only=True,ready_for_actual_execution=False,code=bind(__file__),draft_protocol=bind(DRAFT),
  entry_template=bind(H/'FUTURE_ROOT_ENTRY_TEMPLATE_NOT_AUTHORIZED.json'),generated=bind(H/'GENERATED_INERT_GUARDS.json'),
  current_private_payload_body_or_hash_reads=0,model_instantiations_or_calls=0,
  reason='V155 explicitly prohibits current inference; future separate policy/resolved source freezes+audits required. No --entry was run.')
 write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)

def authorize_future(entry):
 e=read(entry);assert e['approved_future_frozen_benchmark']is True and e['benchmark_only_not_admitted']and e['source_admission_unchanged']and e['selected_model']is None
 assert all(e[k]is False for k in FORBIDDEN)
 assert e['code']==bind(__file__)and e['draft_origin']==bind(DRAFT)
 # An entry under V155, or the existing V155 policy, can never authorize private reads.
 fp=check(e['future_phase_policy']);rp=check(e['resolved_protocol'])
 assert fp.parent!=V and rp.parent==fp.parent and fp!=BLOCKING_POLICY
 policy=read(fp);p=read(rp);d=read(DRAFT)
 assert policy['phase']!='V155'and policy['approved_current_frozen_diagnostic']is True
 assert policy['benchmark_only_not_admitted']and policy['source_admission_unchanged']and policy['selected_model']is None
 assert policy['models_fixed_before_current_body_reads']==p['models_fixed']
 assert p['ready_for_actual_execution']is True and p['code']==bind(__file__)and p['draft_origin']==bind(DRAFT)
 assert p['models']==list(MODELS)and p['original']==d['original']and p['future_phase_policy']==e['future_phase_policy']
 assert len(p['source_controls'])==len(p['source_audits'])==2
 controls=[]
 for index,(cb,ab)in enumerate(zip(p['source_controls'],p['source_audits'])):
  assert cb['path']==d['source_controls'][index]['path']and ab['path']==d['required_source_audits'][index]['path']
  if index==0:assert cb==d['source_controls'][0]
  c=read(check(cb));a=read(check(ab));assert c['completed']and c['PASS']and c['selected_model']is None
  assert a['completed']and a['PASS']and a['control_complete']==cb and a.get('selected_model')is None
  controls.append(c)
 for name in MODELS:
  f=p['models_fixed'][name];old=d['models_fixed'][name]
  for k in('arm','parameters','fixed_update','parent_C32_update','seed','auxiliary_weight','visibility_initialization','head_code','class_name','threshold_logit','source_current_allowed'):assert f[k]==old[k]
  if name in('C35','C36'):
   for k in('source_fit','source_freeze','source_results','source_checkpoint','source_state_sha','source_checks_all_pass'):assert f[k]==old[k]
  fit=read(check(f['source_fit']));fr=read(check(f['source_freeze']));res=read(check(f['source_results']))
  assert controls[0 if name in('C35','C36')else 1]['arms'][f['arm']]['fit']==f['source_fit']
  assert fit['completed']and fit['optimizer_updates']==f['fixed_update']and fit['model_freeze']==f['source_freeze']and fit['source_results']==f['source_results']
  assert fr['completed']and fr['parameters']==f['parameters']and fr['update']==f['fixed_update']and fr['threshold_logit']==0 and fr['all_geometry_input_zero']
  assert fr['checkpoint']==f['source_checkpoint']and fr['source_checks_all_pass']==f['source_checks_all_pass']
  assert not fr['current_allowed']and fr['selected_model']is None and not res['adopted']and not res['current_allowed']and res['selected_model']is None
  assert res['final_state']==f['source_state_sha'];check(f['head_code'])
  # No candidate checkpoint hash/open/load at authorization time.
 assert not Path(e['output_directory']).exists()and Path(e['output_directory']).resolve().parent==fp.parent
 return e,p

def run_future(entry):
 e,p=authorize_future(entry);original=p['original'];out=Path(e['output_directory']);out.mkdir()
 write(out/'STARTED_BEFORE_PRIVATE_PAYLOADS.json',dict(completed=False,entry=bind(entry),resolved_protocol=e['resolved_protocol'],candidate_checkpoint_hash_or_loads=0,selected_model=None))
 parity=False;calls=0
 try:
  # First actual private payload access can occur only after the future authorization above.
  operator=module(check(original['original_operator']),'future_exact_V124_operator')
  inherited=module(check(read(DRAFT)['old_head_code']),'future_V149_pure_helpers')
  a=operator.load_saved_inputs(check(original['feature_packet']))
  assert hashlib.sha256(a['features1928'].tobytes()).hexdigest()==original['feature_values_sha256']
  assert tuple(a)==tuple(original['input_whitelist'])
  keys=tuple(original['metadata_whitelist']);scorekeys=tuple(original['C16_output_score_whitelist'])
  with np.load(check(original['C16_original_raw_packet']),allow_pickle=False)as f:saved={k:f[k].copy()for k in keys+scorekeys}
  import torch
  runtime=original['runtime'];assert Path(sys.executable).resolve()==Path(runtime['executable']).resolve()and torch.__version__==runtime['torch']and np.__version__==runtime['numpy']and torch.cuda.is_available()
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  hm=module(check(original['C11_head_class']),'future_original_C16_class');freeze=read(check(original['C16_freeze']))
  payload=torch.load(check(original['C16_checkpoint']),map_location='cpu',weights_only=True)
  assert payload['update']==freeze['selected_update']==4500 and payload['trainer']==freeze['trainer']and payload['head_class_code']==original['C11_head_class']
  baseline=hm.SpatialTargetQueryHead().cuda().eval().requires_grad_(False);baseline.load_state_dict(payload['state_dict'],strict=True);del payload
  before=operator.state_sha(baseline);s16=read(check(original['original_C16_inference_seal']))
  assert before==s16['head_state_before']==s16['head_state_after']
  raw,z,pv=operator.predict_generated_or_actual(baseline,a['features1928'],a['valid'],torch,'cuda')
  checks=inherited.verify_original_reference(a,saved,raw,z,pv);assert before==operator.state_sha(baseline)
  assert int(a['valid'].sum())==1739 and int((~a['valid']).sum())==45
  check(original['C16_checkpoint']);check(original['feature_packet']);check(original['C16_original_raw_packet'])
  parityfile=out/'C16_ORIGINAL_ALL1784_BYTE_PARITY_SEAL.json'
  write(parityfile,dict(completed=True,PASS=True,entry=bind(entry),all1784_single_batch=True,repeat_calls=2,all_raw_safe_probability_metadata_bytes_exact=checks,
   state_before=before,state_after=before,valid=1739,U=45,additional_U=0,candidate_checkpoint_hash_load_or_forward_calls=0,selected_model=None))
  parity=True;del baseline,saved;models={}
  for name in MODELS:
   fixed=p['models_fixed'][name]
   assert read(parityfile)['completed']and read(parityfile)['PASS']
   source=module(check(fixed['head_code']),'future_fixed_'+name+'_class')
   payload=torch.load(check(fixed['source_checkpoint']),map_location='cpu',weights_only=True)
   for k in('update','parent_C32_update','seed','arm','auxiliary_weight'):
    expected=fixed['fixed_update']if k=='update'else fixed[k];assert payload[k]==expected
   if name in('C37','C38'):assert payload['visibility_initialization']==fixed['visibility_initialization']and payload['architecture']==fixed['head_code']
   model=getattr(source,fixed['class_name'])().cuda().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True);del payload
   assert sum(x.numel()for x in model.parameters())==fixed['parameters']and not model.geometry.weight.any()
   before=source_state_sha(model);assert before==fixed['source_state_sha'];records=[]
   def visibility_only(features):
    outputs=model(features,features.new_zeros((N,42)))
    records.append(torch.stack([outputs[k]for k in('external_occlusion','target_entity','coarse_point_quality')],1).cpu().numpy())
    return outputs['external_occlusion']
   raw,z,pv=operator.predict_generated_or_actual(visibility_only,a['features1928'],a['valid'],torch,'cuda');calls+=2
   assert len(records)==2 and records[0].shape==(N,3)and eq(records[0],records[1])and eq(raw,records[0][:,0].copy())
   assert before==source_state_sha(model)and not model.geometry.weight.any()
   pred=np.where(a['valid'],(z>=0).astype(np.int8),-1);assert (pred<0).sum()==45
   packet=out/(name+'_RAW_HEAD_OUTPUTS.npz');csvfile=out/(name+'_HEAD_ONLY_SCORES.csv')
   np.savez_compressed(packet,raw_head_logit_before_U=raw,query_logit=z,p_visible_FP32=pv,raw_three_logits=records[0],prediction_D0_E1_Uminus1=pred,**{k:a[k]for k in keys})
   inherited.write_minimal_csv(csvfile,a,z,pv)
   check(fixed['source_checkpoint']);check(fixed['source_freeze'])
   models[name]=dict(raw_packet=bind(packet),scores_csv=bind(csvfile),source_freeze=fixed['source_freeze'],source_checkpoint=fixed['source_checkpoint'],
    parameters=fixed['parameters'],fixed_update=fixed['fixed_update'],source_checks_all_pass=fixed['source_checks_all_pass'],source_current_allowed_false_unchanged=True,
    state_before=before,state_after=before,full1784_repeat2_all3_byteexact=True,head_calls=2,valid=1739,U=45,additional_U=0,auxiliary_logits_never_primary_gates=True,selected_model=None)
   del model;torch.cuda.empty_cache()
  assert tuple(models)==MODELS and calls==8
  check(original['feature_packet']);check(original['C16_original_raw_packet'])
  write(out/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),entry=bind(entry),protocol=e['resolved_protocol'],future_phase_policy=e['future_phase_policy'],
   C16_parity=bind(parityfile),models=models,frames=223,slots=1784,valid=1739,U=45,additional_U=0,original_q_L_ID_order_and_metadata_bytes_preserved=True,
   source_admission_unchanged=True,benchmark_only_not_admitted=True,selected_model=None,all_four_reported_no_arm_choice=True,
   encoder_R0_RGB_mask_GT_optimizer_fit_PnP_q_correction_calls=0,new_candidate_head_calls=8,original_C16_head_calls=2,
   requires_separate_saved_evaluation_entry_before_label_reads=True))
 except BaseException:
  write(out/'FAILED_EXECUTION.json',dict(completed=False,C16_parity_completed=parity,candidate_head_calls=calls,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry',type=Path)
 args=parser.parse_args();prepare()if args.prepare else run_future(args.entry)
