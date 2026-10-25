"""Bind the fixed V223 SOURCE-only contrast from V222 metadata, without payload reads."""
import json,hashlib
from pathlib import Path
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';H=R/'V223_SPLIT_TARGET_TRAINING_B'
S=P/'query_visibility_source_target_address_consistency_20261025_v223'
OLD=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
def safe(p):
 p=Path(p)
 assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower()
 for q in (*reversed(p.parents),p):assert not q.is_symlink()
 return p
def bind(p):
 p=safe(p);assert p.suffix in {'.json','.jsonl','.py','.md'}
 z=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(z).hexdigest(),'size_bytes':len(z)}
def read(p):return json.loads(safe(p).read_bytes())
def new(p,o):
 with safe(p).open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
 return bind(p)
def unique(rr):
 out={}
 for b in rr:
  assert 'v112' not in b['path'].lower()
  if b['path'] in out:assert out[b['path']]['sha256']==b['sha256']
  out[b['path']]=b
 return list(out.values())
ready=read(H/'FINAL_READY.json');assert ready['completed'] is True and ready['PASS'] is True
qa=R/'V223_SPLIT_TARGET_TRAINER_QA_A/REVIEW.json';assert read(qa)['PASS'] is True
entry=read(OLD/'ROOT_ACTUAL_FIXED_SOURCE_PAIR_FIT_ENTRY.json')
entry.pop('approved_source_pair_fit_or_probe',None);entry.pop('throughput_probe_complete',None)
entry.update(approved_source_split_target_fit=True,source_only=True,automatic_retry=False,mode='FIXED_SPLIT_TARGET_FIT',
 code=bind(H/'train_split_point_part_source.py'),contract=bind(H/'source_training_contract.py'),
 protocol=bind(H/'PROTOCOL_BEFORE_ANY_V223_SOURCE_PAYLOAD_OR_FIT.json'),
 split_head=bind(H/'split_point_part_head.py'),
 root_design_protocol=bind(R/'V223_ROOT_PREPARATION/PROTOCOL_BEFORE_SPLIT_TARGET_MODEL_OR_PAYLOAD.json'),
 output_directory=str(S/'paired_source_fit_v1'))
def all_bindings(z):
 if isinstance(z,dict):
  if {'path','sha256'} <= set(z):yield (z['path'],z['sha256'])
  for v in z.values():yield from all_bindings(v)
 elif isinstance(z,list):
  for v in z:yield from all_bindings(v)
ready_graph=set(all_bindings(ready))
assert all((entry[k]['path'],entry[k]['sha256']) in ready_graph for k in ['code','contract','protocol','split_head'])
assert entry['root_design_protocol']['sha256']=='5b38c7206197d1ba5158d619d306146b7108dc91d2572aec998ddbd09ea01c44'
entry['root_scope']=new(S/'ROOT_SCOPE_FIXED_FIT.json',{'completed':True,'approved_source_split_target_fit':True,
 'mode':'FIXED_SPLIT_TARGET_FIT','independent_code_QA':bind(qa),'root_design_protocol':entry['root_design_protocol'],
 'same_C65_SOURCE_continuity_not_protected_score_selection':True,'same_V222_exact_data_and_CF_admission':entry['CF_target_admission'],
 'new_protected_model_inference':False,'new_PnP':False,'new_GT_ROI_reference_or_mask_inputs':False,
 'C_held360_metadata_exception':bind(R/'V223_ROOT_PREPARATION/ROOT_SCOPE_EXCEPTION_AND_CONTAINMENT.json'),
 'protected_metadata_zero_claim':False,'exception_values_used_in_model_design_or_fit':False,
 'both_final_states_before_new_DEV_outputs':True})
entry['root_recipe']=new(S/'ROOT_FIXED_SOURCE_PAIR_RECIPE.json',{'completed':True,'approved_frozen_recipe':True,
 'budget_selection_basis':'PROSPECTIVE_FIXED1100_NO_SCORE_OR_PROBE','mode':'FIXED_SPLIT_TARGET_FIT','steps':1100,
 'learning_rate':.001,'weight_decay':.0001,'natural_weak_weight':.25,'CF_weak_weight':.25,'CF_parents_per_step':4,
 'CF_sampling':'UNIFORM_WITH_REPLACEMENT','CF_parent_BCE_weights_BEFORE_COVER_SHAM':[.25,.5,.25],
 'old_batch':192,'strong_batch':32,'natural_batch':16,'CF_both_arms_batch':12,'total_each_batch':252,
 'threshold_logit':0.,'new_hinge':False,'new_backbone':False,'common_draw_seeds':[1810,1811,1812],
 'independent_CF_draw_seed':1813,'CF_common_to_both_arms':True,'early_stop_or_checkpoint_selection':False,
 'arms':['C68SHARED_POINT_PART_LOGIT','C69SEPARATE_POINT_PART_LOGITS'],
 'strong_logit_route':['PART','POINT'],'weak_CF_logit_route':['PART','PART'],
 'expected_parameters_each':121561,'control_unused_POINT_parameters':3202,
 'point0_target_not_query_pixel_or_finite_part_gold':True,'weak_part_target_not_native_point_gold':True})
old_drop={str(OLD/'ROOT_SCOPE_FIXED_FIT.json'),str(OLD/'ROOT_FIXED_SOURCE_PAIR_RECIPE.json'),
 str(OLD/'throughput_probe_v1/PROBE_COMPLETE.json'),str(R/'V222_TRAINER_CODE_QA_A/REVIEW.json')}
entry['metadata_whitelist']=unique([b for b in entry['metadata_whitelist'] if b['path'] not in old_drop]+[
 entry['root_scope'],entry['root_recipe'],entry['root_design_protocol'],bind(qa)])
entry['code_whitelist']=unique(entry['code_whitelist']+[entry['split_head']])
assert len(entry['fit_binary_whitelist'])==169 and len(entry['postfreeze_binary_whitelist'])==39
assert not {b['path'] for b in entry['fit_binary_whitelist']} & {b['path'] for b in entry['postfreeze_binary_whitelist']}
result=new(S/'ROOT_ACTUAL_FIXED_SOURCE_PAIR_FIT_ENTRY.json',entry)
print(json.dumps({'completed':True,'entry':result,'fixed_steps_each':1100,'model_parameters_expected':121561,
 'fit_binary_bindings':169,'postfreeze_binary_bindings':39,'payloads_read_in_assembly':0}))
