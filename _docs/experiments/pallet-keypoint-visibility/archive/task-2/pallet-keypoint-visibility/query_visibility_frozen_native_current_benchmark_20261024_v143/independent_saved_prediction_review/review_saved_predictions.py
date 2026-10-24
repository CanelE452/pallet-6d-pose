"""Independent saved prediction/metadata arithmetic only. No labels or model objects."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import torch
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_frozen_native_current_benchmark_20261024_v143')
OUT=V/'frozen_current_head_benchmark_v1';H=V/'independent_saved_prediction_review'
def bind(p):return dict(path=str(Path(p)),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def scalar(a,b):return (math.isnan(a)and math.isnan(b))or a==b
seal=read(OUT/'PREDICTION_SEAL.json');assert bind(OUT/'PREDICTION_SEAL.json')['sha256']=='74f403323879e42f08b0561e483c2939df91321922365c1c4960838f754a27b9'
assert seal['completed']and seal['PASS']and seal['selected_model']is None and seal['benchmark_only_not_admitted']and not seal['source_admission_changed']
protocol=read(check(seal['protocol']));policy=read(check(seal['policy']));parity=read(check(seal['C16_parity']));oldseal=read(check(seal['original_C16_inference_seal']))
check(seal['code']);check(seal['entry']);check(seal['readiness'])
assert parity['completed']and parity['PASS']and all(parity['all_raw_safe_probability_metadata_bytes_exact'].values())
assert parity['state_before']==parity['state_after']==oldseal['head_state_before']==oldseal['head_state_after']
assert parity['all1784_single_batch']and parity['repeat_calls']==2 and parity['repeat_exact']
assert parity['candidate_checkpoint_body_reads_or_forward_calls']==parity['GT_RGB_encoder_initializer_reads_or_calls']==0
assert seal['full_batch_rows']==1784 and seal['repeat_calls_each']==2 and seal['original_C16_head_calls']==2 and seal['new_candidate_head_calls']==4
assert seal['frames']==223 and seal['slots']==1784 and seal['valid']==1739 and seal['U']==45 and seal['additional_U']==0
assert seal['GT_read']is False and seal['RGB_or_mask_read']is False and seal['new_encoder_or_initializer_calls']==seal['optimizer_or_fit_updates']==0
assert seal['calibration_threshold_checkpoint_arm_model_selection']is False and seal['PnP_or_q_L_correction']==0
metadata=tuple(protocol['metadata_whitelist']);scorekeys=tuple(protocol['C16_output_score_whitelist']);fields=tuple(protocol['CSV_fields'])
assert len(metadata)==12 and len(scorekeys)==3 and len(fields)==9
with np.load(check(protocol['C16_original_raw_packet']),allow_pickle=False)as f:baseline={k:np.array(f[k],copy=True)for k in metadata+scorekeys}
# Select original input metadata members only; the features1928 and original C11
# score members are neither hashed nor deserialized by this independent review.
with np.load(protocol['feature_packet']['path'],allow_pickle=False)as f:original={k:np.array(f[k],copy=True)for k in metadata}
assert all(eq(original[k],baseline[k])for k in metadata)
assert baseline['id'].shape==baseline['native_corner'].shape==(1784,)
assert len(set(baseline['id'].tolist()))==223 and len(set(zip(baseline['id'].tolist(),baseline['native_corner'].tolist())))==1784
assert np.array_equal(baseline['native_corner'],np.tile(np.arange(8,dtype=np.int8),223))
assert (baseline['id'].reshape(223,8)==baseline['id'].reshape(223,8)[:,:1]).all()
for k in metadata[:6]:assert baseline[k].shape==(1784,)and baseline[k].dtype==np.bool_
assert np.array_equal(baseline['valid'],baseline['query_available']) and np.array_equal(baseline['valid'],baseline['prior_valid']&baseline['context_available']&baseline['local_central4_available'])
assert int(baseline['valid'].sum())==1739 and int((~baseline['valid']).sum())==45
assert baseline['query_xy'].dtype==baseline['side_L'].dtype==np.float32 and baseline['original_query_xy'].dtype==baseline['original_side_L'].dtype==np.float64
assert baseline['query_xy'].shape==baseline['original_query_xy'].shape==(1784,2) and baseline['side_L'].shape==baseline['original_side_L'].shape==(1784,)
assert eq(baseline['query_xy'],baseline['original_query_xy'].astype(np.float32))and eq(baseline['side_L'],baseline['original_side_L'].astype(np.float32))
def scores(a):
 for k in scorekeys:assert a[k].shape==(1784,)and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
 raw,z,pv=[a[k]for k in scorekeys]
 assert eq(z,np.where(a['valid'],raw,0).astype(np.float32))
 assert eq(pv,torch.sigmoid(-torch.as_tensor(z)).numpy())
 assert not z[~a['valid']].any()and(pv[~a['valid']]==.5).all()
 return np.where(a['valid'],(z>=0).astype(np.int8),-1).astype(np.int8)
scores(baseline);models={}
for name in ('C29','C32'):
 m=seal['models'][name];fixed=policy['models_fixed_before_private_feature_label_prediction_body_reads'][name]
 assert m['source_freeze']==fixed['source_freeze']and m['source_checkpoint']==fixed['source_checkpoint']
 freeze=read(check(m['source_freeze']));assert freeze['source_eligible']is False and freeze['current_allowed']is False and freeze['all_geometry_input_zero']is True
 assert m['source_eligible']is False and m['selected_model']is None and m['fixed_update']==8000 and m['threshold_logit']==0
 assert m['parameters']==224513 and m['state_before']==m['state_after']and m['head_repeat_exact']and m['head_calls']==2
 assert all(m[k]is True for k in ('benchmark_only_not_admitted','source_current_allowed_false_unchanged','G42_input_all_zero','geometry_weights_zero_before_after','checkpoint_file_unchanged'))
 with np.load(check(m['raw_packet']),allow_pickle=False)as f:
  assert set(f.files)==set(metadata+scorekeys+('prediction_D0_E1_Uminus1',));a={k:np.array(f[k],copy=True)for k in f.files}
 assert all(eq(a[k],baseline[k])and eq(a[k],original[k])for k in metadata)
 pred=scores(a);assert eq(pred,a['prediction_D0_E1_Uminus1'])and int((pred==-1).sum())==45
 csvpath=check(m['scores_csv'])
 with csvpath.open(newline='')as f:
  reader=csv.DictReader(f);assert tuple(reader.fieldnames)==fields;rows=list(reader)
 assert len(rows)==1784
 for i,r in enumerate(rows):
  assert r['id']==str(a['id'][i])and int(r['native_corner'])==int(a['native_corner'][i])
  for k,val in [('query_x',a['original_query_xy'][i,0]),('query_y',a['original_query_xy'][i,1]),('roi_side',a['original_side_L'][i]),('raw_query_aux_hidden_logit',a['query_logit'][i]),('raw_p_visible_FP32',a['p_visible_FP32'][i])]:assert scalar(float(r[k]),float(val)),(name,i,k)
  assert r['valid']==str(bool(a['valid'][i]))and r['authoritative_decision']==({0:'D',1:'E',-1:'UNKNOWN'}[int(pred[i])])
 assert Path(seal['C16_parity']['path']).stat().st_mtime_ns<=Path(m['raw_packet']['path']).stat().st_mtime_ns
 models[name]=dict(slots=1784,valid=1739,U=45,all12_metadata_dtype_shape_bytes_exact_original_C16_and_input=True,raw_safe_probability_FP32_exact_algebra=True,prediction_int8_raw0_Ufirst_exact=True,CSV9fields_all1784_exact=True,prediction_counts_label_free={str(k):int((pred==k).sum())for k in(0,1,-1)},source_freeze=m['source_freeze'],source_checkpoint_hash_inherited_not_rehashed=m['source_checkpoint'],state_before_after_equal_in_producer_seal=True,repeat2_and_checkpoint_unchanged_are_producer_receipt_claims_no_reforward=True)
assert seal['outputs']==[b for m in seal['models'].values()for b in (m['raw_packet'],m['scores_csv'])]
result=dict(completed=True,PASS=True,code=bind(__file__),prediction_seal=bind(OUT/'PREDICTION_SEAL.json'),C16_parity=seal['C16_parity'],protocol=seal['protocol'],original_C16_packet=protocol['C16_original_raw_packet'],all_original1784_ID_q_L_masks_metadata_exact=True,original_feature_packet_members_read=list(metadata),original_feature_values_or_C11_scores_read=False,baseline_C16_raw_safe_probability_algebra_checked=True,models=models,source_selectednull_and_ineligible_unchanged=True,selected_model=None,labels_RGB_weights_features_model_instantiation_forward_fit_PnP_reads_or_calls=0,torch_CPU_elementwise_sigmoid_arithmetic_only=True,qualification='Prediction integrity only, no accuracy claim before original annotation join. Model repeats/state/file invariance are verified as internally consistent producer receipts, not independently rerun model/weight hashes.',V112_access=0)
f=H/'REVIEW.json'
with f.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(f),models=models)),flush=True)
