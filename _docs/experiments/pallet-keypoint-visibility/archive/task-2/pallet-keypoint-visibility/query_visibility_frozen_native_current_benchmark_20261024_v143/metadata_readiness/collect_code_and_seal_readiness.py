"""Metadata/code-only readiness: never deserialize/hash any current packet or checkpoint."""
from pathlib import Path
import ast, hashlib, json
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_frozen_native_current_benchmark_20261024_v143'
H=V/'metadata_readiness'
V111=P/'query_visibility_spatial_target_query_features_20261023_v111'
V124=P/'query_visibility_training_duration_control_20261023_v124'
V133=P/'query_visibility_native_geometry_control_20261024_v133'
POLICY=V/'ROOT_FROZEN_BENCHMARK_POLICY_BEFORE_PRIVATE_BODY_READS.json'
def read_meta(p):
 p=Path(p);assert p.suffix=='.json' and '_v112' not in str(p);return json.loads(p.read_text())
def bind_meta_or_code(p):
 p=Path(p);assert p.suffix in ('.json','.py') and '_v112' not in str(p)
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def exists_only(binding):
 p=Path(binding['path']);assert '_v112' not in str(p)
 return dict(**binding,exists=p.is_file(),size_bytes_metadata_only=p.stat().st_size if p.is_file() else None,expected_sha_source='inherited completed metadata only; payload NOT hashed or opened')
policy=read_meta(POLICY)
assert bind_meta_or_code(POLICY)['sha256']=='33856e89ab3a563e7ee80d2d2daf846d64199122340c02e7299f66fc9f51cd1b'
c11sealpath=V111/'private_evaluation/C11_FROZEN_CURRENT223/INFERENCE_SEAL.json'
c16sealpath=V124/'private_evaluation/C16_FROZEN_CURRENT223/INFERENCE_SEAL.json'
prep=V111/'private_evaluation/C11_FROZEN_CURRENT223/INPUT_PREPARATION_AUDIT.json'
c11=read_meta(c11sealpath);c16=read_meta(c16sealpath);audit=read_meta(prep)
freeze=read_meta(V124/'runs/C16_EXACT_C15_DURATION_CONTROL_8000/MODEL_FREEZE.json')
assert c11['completed'] and c11['model_unchanged'] and c11['slots']==1784
assert c16['completed'] and c16['PASS'] and c16['feature_shape']==[1784,1928] and c16['valid']==1739 and c16['U']==45 and c16['additional_U']==0
assert c16['source_model_freeze']==bind_meta_or_code(V124/'runs/C16_EXACT_C15_DURATION_CONTROL_8000/MODEL_FREEZE.json')
assert c16['checkpoint']==freeze['checkpoint'] and freeze['selected_update']==4500
assert c16['C11_inference_seal']==bind_meta_or_code(c11sealpath)
assert c16['C11_current_packet'] in c11['outputs']
assert c16['head_state_before']==c16['head_state_after'] and c16['model_unchanged']
operator=V124/'current_evaluation_preparation/frozen_head_current.py'
code=operator.read_text();tree=ast.parse(code)
whitelist=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WHITELIST' for t in n.targets)))
assert list(whitelist)==c16['input_fields_read']
predict=ast.get_source_segment(code,next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='predict_generated_or_actual'))
assert 'x=torch.as_tensor(features,device=device);a=model(x);b=model(x)' in predict
assert 'range(' not in predict and 'torch.equal(a,b)' in predict
assert 'z=np.where(available,raw,0).astype(np.float32)' in predict
head=V111/'train_spatial_target_query_head.py';geometry=V133/'source_fit_preparation/train_matched_geometry.py'
headsrc=head.read_text();geosrc=geometry.read_text()
assert 'torch.nn.Linear(3464,64)' in headsrc and 'features.shape[1]!=1928' in headsrc
assert 'torch.nn.Linear(3464,64)' in geosrc and 'self.geometry=torch.nn.Linear(42,64,bias=False)' in geosrc
assert 'h=self.head[0](visual)+self.geometry(geometry42)' in geosrc
models={}
for name,entry in policy['models_fixed_before_private_feature_label_prediction_body_reads'].items():
 f=read_meta(entry['source_freeze']['path']);assert bind_meta_or_code(entry['source_freeze']['path'])==entry['source_freeze']
 assert f['checkpoint']==entry['source_checkpoint'] and f['update']==8000 and f['parameters']==224513
 assert f['source_eligible'] is False and f['current_allowed'] is False and f['all_geometry_input_zero'] is True and f['threshold_logit']==0
 models[name]=dict(source_freeze=entry['source_freeze'],source_checkpoint=exists_only(entry['source_checkpoint']),fixed_update=8000,parameters=224513,original_source_ineligible=True,original_current_allowed=False,all_geometry_input_zero=True)
result=dict(
 completed=True,PASS=True,readiness='METADATA_AND_CODE_READY_PENDING_MANDATORY_ACTUAL_C16_BITPARITY',
 policy=bind_meta_or_code(POLICY),code=bind_meta_or_code(__file__),
 evidence=dict(C11_feature_seal=bind_meta_or_code(c11sealpath),C11_input_preparation_metadata=bind_meta_or_code(prep),C16_inference_seal=bind_meta_or_code(c16sealpath),C16_freeze=bind_meta_or_code(V124/'runs/C16_EXACT_C15_DURATION_CONTROL_8000/MODEL_FREEZE.json'),C16_exact_operator=bind_meta_or_code(operator),C16_operator_protocol=c16['operator'],C11_visual_head_class=bind_meta_or_code(head),C11_feature_assembly=bind_meta_or_code(V111/'feature_preparation/prepare_spatial_query_features.py'),candidate_GeometryQueryHead_class=bind_meta_or_code(geometry)),
 actual_current_cache=exists_only(c16['C11_current_packet']),
 cached_feature_values_sha256_inherited=c16['feature_values_SHA256'],
 input_schema=dict(features1928=dict(shape=[1784,1928],dtype='float32',finite=True,one_row_per_original_native_slot=True,occurrence_or_unit_remap='NONE; identity row order inherited'),
    availability={k:dict(shape=[1784],dtype='bool') for k in ('query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside')},
    query_xy=dict(shape=[1784,2],dtype='float32'),side_L=dict(shape=[1784],dtype='float32'),original_query_xy=dict(shape=[1784,2],dtype='float64'),original_side_L=dict(shape=[1784],dtype='float64'),id=dict(shape=[1784],dtype='Unicode',frames=223),native_corner=dict(shape=[1784],dtype='int8',native_first8_unchanged=True)),
 input_whitelist=list(whitelist),
 feature_layout=[dict(slice=[0,384],name='T normalized target-context CLS'),dict(slice=[384,768],name='Q normalized target-context patch bilinear at original query'),dict(slice=[768,1152],name='F normalized full96 CLS'),dict(slice=[1152,1536],name='C normalized central32 CLS'),dict(slice=[1536,1920],name='P normalized mean local central2x2 patch tokens'),dict(slice=[1920,1924],name='cos(T,C),cos(T,F),cos(Q,C),cos(Q,P)'),dict(slice=[1924,1927],name='target224/full96/central32 valid fractions'),dict(slice=[1927,1928],name='Q_inside')],
 mask_contract=dict(valid_equals_query_available=True,valid_equals_prior_AND_context_AND_localcentral4=True,prior_valid=1739,prior_U=45,final_valid=1739,final_U=45,additional_U=0,context_available=1784,local_central4_available=1739,Q_inside=1784,Q_outside_is_not_additional_U=True,local_central4='original valid96[:,:,47:49,47:49].all(1,2,3)',current_valid96_payload_not_required_for_head_reuse=True,raw_pixel_masks_not_in_features_packet='only inherited availability flags + validity fractions; no RGB/mask body needed',upstream_current_model_input_binding_only=audit['prior_artifact_bindings']['current_model_input']),
 C16_parity_control=dict(source_checkpoint=exists_only(c16['checkpoint']),selected_source_update=4500,parameters=221825,source_visual_head_state_sha256_inherited=c16['head_state_before'],stored_outputs=[exists_only(x)for x in c16['outputs']],raw_baseline='RAW_HEAD_OUTPUTS.npz::raw_head_logit_before_U float32[1784]',safe_baseline='RAW_HEAD_OUTPUTS.npz::query_logit float32[1784]; np.where(valid,raw,0)',probability='RAW_HEAD_OUTPUTS.npz::p_visible_FP32 = CPU torch.sigmoid(-safe_z), display only',operator='ALL1784 float32 rows in exact original order, one model(x) GEMM batch; repeat once; no compaction or 512 chunking',raw_output_parity='must match all1784 saved raw_head_logit_before_U bytes including invalid rows before candidate forward; safe logits and U flags also exact',runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',numpy='1.26.4',device='cuda',precision='FP32',threads=1,deterministic_algorithms=True,cudnn_benchmark=False,matmul_TF32=False,cudnn_TF32=False,CUBLAS_WORKSPACE_CONFIG=':4096:8'),weight_and_model_state_before_after_unchanged=True),
 candidates=models,
 compatibility=dict(visual_input1928_same=True,visual_expansion3464_same=True,candidate_extra42x64_biasfree_branch='allzero inputs; do not compute privateG42',candidate_parameters=224513,C16_parameters=221825,strict_C16_checkpoint_into_GeometryQueryHead=False,load_C16_for_parity='original SpatialTargetQueryHead exact class; checkpoint state_dict strict',load_candidates='GeometryQueryHead exact class, FINAL_RESUME_CHECKPOINT.pt state_dict strict; assert geometry weights remain zero; fixed G0[1784,42]',fixed_sign='available and z<0 ->D, available and z>=0 ->E; original invalid ->U',no_new_availability_criterion=True,source_current_allowed_false_unchanged=True,selected_model=None),
 required_execution_order=['Root-reviewed prospective protocol and explicit root entry before any private payload hash/read or checkpoint load.','Verify bound metadata then private packet/checkpoint hashes under explicit entry; load only input whitelist and C16 output whitelist, no labels.','C16 exact original class/all1784 operator: require raw+safe byteparity against original saved C16, repeat parity and frozen/file state checks; fail closed on mismatch.','Only after parity PASS, forward BOTH predeclared C29/C32 on identical original1784features plus fixedG0, exact all1784 batches/repeats; preserve original45U and seal both results.','Only after complete prediction seal may separate original-mapping evaluation read labels/baseline predictions; no eligibility/adoption/threshold/checkpoint/arm choice.'],
 forbidden_payload_reads_so_far=dict(current_feature_body=0,current_label_body=0,current_prediction_body=0,current_RGB_or_mask_body=0,checkpoint_body=0,current_payload_or_checkpoint_file_hash=0,model_import_instantiation_or_forward=0),
 preparation_search_disclosure='Initial filename/text search included existing ROOT_RESULTS_KO aggregate report lines; no per-slot private input/prediction/annotation table or payload was opened. Subsequent audit narrowed to code and completed metadata seals only.',
 no_current_performance_or_model_admission_claim=True,source_selectednull_unchanged=True,V112_access=0,PnP=0)
out=H/'READINESS.json'
with out.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),cache_exists=result['actual_current_cache']['exists'],fixed_candidates=list(models))))
