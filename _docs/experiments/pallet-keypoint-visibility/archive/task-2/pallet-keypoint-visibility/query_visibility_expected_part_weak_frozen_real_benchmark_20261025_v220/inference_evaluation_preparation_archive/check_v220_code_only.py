"""Functional invented fixtures and numerical-kernel AST checks; no protected bodies."""
from pathlib import Path
import importlib.util,json,hashlib,ast,copy,sys
import numpy as np
H=Path(__file__).parent

def module(n,path):
 sp=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=module('v220_pred_toy',H/'infer_both_frozen_expected_part_heads.py');e=module('v220_eval_toy',H/'evaluate_saved_expected_part_heads_real.py');results=[]
def test(n,v):assert bool(v),n;results.append(n)
def reject(f):
 try:f()
 except (AssertionError,ValueError):return True
 return False
b=dict(path=str(H/'INVENTED_NOT_ACCESSED.json'),sha256='a'*64);same=lambda:dict(b)
sourcekeys=dict(code=same(),protocol=same(),kernels=dict(base_head=same(),candidate=same(),spatial_head=same()))
fit=dict(completed=True,PASS=True,source_only=True,from_scratch=True,DEV_not_fit=True,protected_access=0,selected_arm=None,code=same(),entry=same(),parameters_each=118359,steps_each=4000,all2_freeze=same(),arms={a:dict(checkpoint=same())for a in p.SOURCE_ARM_MAP.values()})
review=dict(completed=True,PASS=True,fit_complete=same(),initial_seed1810_state_exact=True,all8000_saved_log_rows_verified=True,results={a:dict(PASS=True,parameters=118359,state_tensors=36,all_finite=True,optimizer_all_steps4000=True)for a in p.SOURCE_ARM_MAP.values()})
entry=dict(fit_complete=same(),source_fit_entry=same());proto=dict(source_fit_trainer=same(),source_fit_protocol=same(),head=same(),candidate=same(),spatial_head=same(),source_both_final_freeze=same())
barrier=dict(completed=True,PASS=True,DEV_or_real_output_emitted=False,checkpoints={a:same()for a in p.SOURCE_ARM_MAP.values()})
test('C219_authentic_COMPLETE_checkpoint_schema_positive_both_runners',p.validate_source_metadata(fit,review,entry,proto,sourcekeys,barrier)and e.validate_source_metadata(fit,review,entry,proto,sourcekeys,barrier))
bad=copy.deepcopy(fit);bad['steps_each']=3999;test('Reject_nonfinal_or_wrong_source_budget',reject(lambda:p.validate_source_metadata(bad,review,entry,proto,sourcekeys,barrier)))
bad=copy.deepcopy(barrier);bad['DEV_or_real_output_emitted']=True;test('Reject_missing_bothfinal_barrier',reject(lambda:e.validate_source_metadata(fit,review,entry,proto,sourcekeys,bad)))
bad=copy.deepcopy(review);bad['fit_complete']['sha256']='b'*64;test('Reject_other_fit_independentQA',reject(lambda:p.validate_source_metadata(fit,bad,entry,proto,sourcekeys,barrier)))
oldp=dict(prior_dense_prediction_code=same(),prior_dense_prediction_protocol=same(),input_complete=same(),original_raw_complete=same())
old=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,code=same(),protocol=same(),input_complete=same(),original_raw_complete=same(),arms={a:dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=bool(i),weak_loss_weight=.25 if i else 0.,outputs={d:same()for d in e.DOMAINS})for i,a in enumerate(e.PRIOR_DENSE_ARMS)})
test('Prior_C62_C63_savedonly_exactflags_join',set(e.prior_dense_outputs(old,oldp))=={'C62','C63'})
bad=copy.deepcopy(old);bad['arms'][e.PRIOR_DENSE_ARMS[1]]['weak_bridge']=False;test('Prior_dense_flag_drift_rejected',reject(lambda:e.prior_dense_outputs(bad,oldp)))
test('Full_existing14_plus2_models_and_public15_no_baselineforward',e.MODELS==('v61','legacy')+tuple('C'+str(i)for i in range(52,66))and len(e.OLD_TAGS)+2==14 and e.TAGS==('C64','C65'))
av=np.array([True,True,True,False]);logit=np.array([-np.nextafter(np.float32(0),np.float32(1)),0.,np.nan,np.nan],np.float32);test('Fixed_zero_decoder_finite_and_originalU',e.direct_decode(logit,av).tolist()==[0,1,-1,-1])
# Exact inherited numerical kernels; names/tags are administrative only.
counts={}
for old,new,names in [('infer_both_frozen_dense_point_heads.py','infer_both_frozen_expected_part_heads.py',('decode','validate_input','native_indices','context_kernel','build_real_context','infer')),('evaluate_saved_dense_point_heads_real.py','evaluate_saved_expected_part_heads_real.py',('decode','direct_decode','joint','qarray','packet_schema','load_packet','current_associations','evaluate_public','prior_outputs','prior_direct_outputs','prior_grouped_outputs','prior_native_outputs','prior_spatial_outputs'))]:
 oldtree=ast.parse((H/'inherited_V213_frozen'/old).read_text());newtree=ast.parse((H/new).read_text());od={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)};nd={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
 for name in names:
  a=ast.dump(od[name],include_attributes=False);z=ast.dump(nd[name],include_attributes=False)
  # public paired cohort label increases model-count text only.
  z=z.replace('ALL15_MODEL_PAIRED_AVAILABLE','ALL13_MODEL_PAIRED_AVAILABLE')
  assert a==z,name
 counts[new]=len(names)
test('19_original_input_context_decoder_metric_kernel_AST_exact',sum(counts.values())==19)
# Self ES membership retained, original D/E guard retained, no primary quality filter.
text=(H/'evaluate_saved_expected_part_heads_real.py').read_text();test('Existing_SELF_ES_primary1237_and_reference461_manual376_guards_retained',"paired_same_query_RGB_D_vs_ES"in text and "len(eligible)==1237"in text and "==461"in text and "manualnear376"in text and 'primary_quality_owner_reference_error_filter=False'in text)
x=dict(completed=True,PASS=True,invented_only=True,count=len(results),checks=results,unchanged_kernel_AST_counts=counts,source_or_protected_arrays_headers_scores_labels_weights_or_model_reads=0,operational_model_instantiations_or_forward=0)
with (H/'GENERATED_CODE_ONLY_CHECKS.json').open('x')as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))
