"""Bounded code-only V203 adaptation. No source/real payload or result body reads."""
from pathlib import Path
import ast,hashlib,os,stat
H=Path(__file__).parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203'
PRED='infer_both_frozen_dense_point_heads.py';EVAL='evaluate_saved_dense_point_heads_real.py'

def rb(path):
    p=Path(path);assert p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold()and p.suffix=='.py'
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def replace_function(source,name,new):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name)
    lines=source.splitlines();return '\n'.join(lines[:node.lineno-1]+new.strip().splitlines()+lines[node.end_lineno:])+'\n'
def one(source,old,new):assert source.count(old)==1,(old,source.count(old));return source.replace(old,new)

PRED_PREP=r'''
def prepare():
 root=read(SCOPE);require(root['approved_actual_forward_or_evaluation']is False and root['status']=='PROSPECTIVE_CODE_PREPARATION_ONLY','Code-only new scope')
 inherited=read(INHERITED_PROTOCOL);fitp=read(FIT_PROTOCOL)
 write(CHECKS,generated_checks())
 p=dict(inherited)
 p.update(completed=True,prepared_only=True,code=bind(Path(__file__)),scope=bind(SCOPE),inherited_protocol=bind(INHERITED_PROTOCOL),checks=bind(CHECKS),
   arms=[dict(name=a,use_sam=s,spatial_q8=n,weak_bridge=ARM_WEAK[a][0],weak_loss_weight=ARM_WEAK[a][1])for a,s,n in ARMS],
   source_fit_protocol=None,source_fit_trainer=None,source_fit_complete=None,source_both_final_freeze=None,independent_source_fit_review=None,
   source_fit_complete_path=str(FITPHASE/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/COMPLETE.json'),
   source_fit_trainer_path=str(FITPHASE/'fit_preparation_a/train_dense_point_weak_bridge.py'),
   source_factory_authority=bind(FIT_PROTOCOL),head=fitp['source_bindings']['existing_head'],candidate=fitp['source_bindings']['candidate'],spatial_head=fitp['spatial_head'],
   source_context_adapter=fitp['context_adapter'],source_context_protocol=fitp['context_protocol'],output=str(OUT),parameters_each=118359,steps_each=4000,
   primary_a_priori='C63',ablation_a_priori='C62',required_source_fit_review_flags=[],
   original_input_metadata_bodies_read_during_prepare=False,existing_input_bindings_copied_without_body_or_fresh_asset_hash=True,
   context_contract='Both same native/mask/actualXY16, original input-only ROWS group8; V202 F64 subtract/divide clip[-8,8] then FP32; self/U zero. No GT/reference/point/role feature.',
   dynamic_source_binding_rule='Root actual entry after V212 final READY, source terminal and independent exact-fit PASS fills all five source bindings; nulls fail before real input/weight bodies.',
   selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p)
 template=H/'ROOT_ACTUAL_PREDICTION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
 write(template,dict(approved_both_frozen_direct_prediction=False,all_two_final4000_frozen=False,labels_old_scores_metrics_or_fit_allowed=False,
   code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],fit_complete=None,independent_source_fit_review=None,
   source_fit_protocol=None,source_fit_trainer=None,source_both_final_freeze=None,source_fit_entry=None,source_terminal_evidence=None,root_confirmed_source_terminal_exit0=False))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],entry_template=bind(template),actual_array_header_asset_hash_weight_or_output_reads=0,actual_forward=0,invented_CPU_toy_forward_only=True)
 write(H/'PREDICTION_PREPARATION_COMPLETE.json',ready);print(json.dumps(ready))
'''

SOURCE_HELPERS=r'''
def resolve_source_bindings(e,p):
 p=dict(p)
 for ek,pk in [('fit_complete','source_fit_complete'),('independent_source_fit_review','independent_source_fit_review'),('source_fit_protocol','source_fit_protocol'),('source_fit_trainer','source_fit_trainer'),('source_both_final_freeze','source_both_final_freeze')]:
  b=e.get(ek);require(isinstance(b,dict)and set(b)=={'path','sha256'}and isinstance(b['sha256'],str)and len(b['sha256'])==64,'Concrete root source binding '+ek)
  require(FITPHASE in safe(b['path']).parents,'Current V212 source closure only');p[pk]=b
 require(p['source_fit_complete']['path']==p['source_fit_complete_path']and p['source_fit_trainer']['path']==p['source_fit_trainer_path'],'Exact prospective V212 source paths')
 return p

def validate_source_metadata(fit,review,e,p,source_entry,barrier):
 require(fit['completed']and fit['PASS']and fit['source_only']and fit['from_scratch']and fit['all2_frozen_before_any_SOURCE_outputs'],'Both source finals before source outputs')
 require(review['completed']and review['PASS']and review['fit_complete']==e['fit_complete'],'Independent PASS binds exact current fit')
 require(fit['code']==source_entry['code']==p['source_fit_trainer']and fit['entry']==e['source_fit_entry']and source_entry['protocol']==fit['protocol']==p['source_fit_protocol'],'Exact source trainer/entry/protocol')
 require(fit['head']==p['head']and fit['candidate']==p['candidate']and fit['spatial_head']==p['spatial_head']and fit['context_adapter']==p['source_context_adapter'],'Same fixed C61 architecture/context')
 require(fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000 and fit['protected_real_COMMON_VAL_access']==0 and set(fit['arms'])=={a for a,_,_ in ARMS},'Both fixed RAW source arms4000')
 require(fit['all2_freeze']==p['source_both_final_freeze']and barrier['completed']and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a,_,_ in ARMS},'Both final checkpoint barrier exact')
 for a,_,_ in ARMS:
  r=fit['arms'][a];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is True and type(r['weak_bridge'])is bool and r['weak_bridge']is ARM_WEAK[a][0]and r['weak_loss_weight']==ARM_WEAK[a][1],'Fixed source target-only/weak-bridge roles')
 return True
'''

EVAL_PREP=r'''
def prepare():
    require(platform.python_version()=='3.10.20'and np.__version__=='1.26.4','Pinned CPU metric runtime')
    inherited=read(INHERITED_PROTOCOL);pred=read(PRED_PROTOCOL);ready=read(PRED_PREP)
    require(ready['completed']and ready['PASS']and ready['prepared_only'],'Prepared code-only predictor')
    for b in inherited['operator_codes']:check(b)
    write(CHECKS,generated_checks());p=dict(inherited)
    for key in ('prior_ten_evaluation_complete','prior_ten_evaluation_code','prior_ten_evaluation_protocol'):p.pop(key,None)
    p.update(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),inherited_protocol=bind(INHERITED_PROTOCOL),checks=bind(CHECKS),
      input_complete=pred['input_complete'],original_raw_complete=pred['original_raw_complete'],source_fit_complete_path=pred['source_fit_complete_path'],source_fit_protocol=None,
      spatial_head=pred['spatial_head'],source_context_adapter=pred['source_context_adapter'],prediction_producer_code=bind(PRED_CODE),prediction_producer_protocol=bind(PRED_PROTOCOL),prediction_producer_preparation=bind(PRED_PREP),
      prior_spatial_prediction_code=inherited['prediction_producer_code'],prior_spatial_prediction_protocol=inherited['prediction_producer_protocol'],
      prior_twelve_evaluation_complete_path=str(OLD203/'saved_real_head_evaluation_v1/COMPLETE.json'),prior_twelve_evaluation_code=inherited['code'],prior_twelve_evaluation_protocol=bind(INHERITED_PROTOCOL),
      prediction_seal_path=str(PRED),arms=list(ARMS),fixed_arm_spatial_q8=ARM_SPATIAL,models=list(MODELS),output=str(OUT),
      packet_contract='New C62/C63 and prior C54..C61 exact16keys; C52/C53 exact13. Original META6 unchanged; current14/publicweak13, no selection.',
      public_cohorts=['FULL360','KNOWN_CENTER_TRUTH','ALL13_MODEL_PAIRED_AVAILABLE','KNOWN_TRUTH_AND_ALL13_PAIRED_AVAILABLE'],
      primary_a_priori='C63',ablation_a_priori='C62',actual_annotation_score_metric_array_header_fresh_asset_hash_reads_during_prepare=0)
    p['exposure']='Repeated exposed current223/publicheld45, not blind. Source includes procedural/open-real weak data; no protected current/publicheld/COMMON/VAL fitting or result-guided choice. Both arms carry regardless source metrics.'
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_SAVED_REAL_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_saved_real_evaluation=False,all_two_real_predictions_frozen=False,evaluation_only=True,no_model_data_architecture_threshold_selection=True,no_forward_fit_RGB_features_weights_PnP_qL_or_V112=True,
      code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),prediction_seal=None,prior_prediction_seal=dict(path=str(PRIOR_PRED),sha256=None),
      prior_direct_prediction_seal=dict(path=str(PRIOR_DIRECT_PRED),sha256=None),prior_grouped_prediction_seal=dict(path=str(PRIOR_GROUPED_PRED),sha256=None),prior_native_prediction_seal=dict(path=str(PRIOR_NATIVE_PRED),sha256=None),
      prior_spatial_prediction_seal=dict(path=str(PRIOR_SPATIAL_PRED),sha256=None),prior_twelve_evaluation_complete=dict(path=p['prior_twelve_evaluation_complete_path'],sha256=None),source_fit_complete=None,independent_source_fit_review=None,output=str(OUT)))
    ready=dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),no_actual_real_annotation_or_score_bodies_or_hashes=True,new_model_fit_forward_threshold_calls=0)
    write(PREP,ready);print(json.dumps(ready))
'''

def main():
    ppath=OLD/'prediction_preparation_a/infer_both_frozen_spatial_q8_heads.py';pb=rb(ppath);assert hashlib.sha256(pb).hexdigest()=='105b2ed484b04577ca605787fb1821aff8c30ed064ed22a8ffb21f4e3b34b7c4'
    epath=OLD/'evaluation_preparation_a/evaluate_saved_spatial_q8_heads_real.py';eb=rb(epath);assert hashlib.sha256(eb).hexdigest()=='ec373683d22bf899cba8428ca286d20ae9eae69a80dfbcfd071f76beb67dc142'
    p=pb.decode().replace('V203 code-only','V213 code-only')
    p=one(p,"FITPHASE=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202'","FITPHASE=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'")
    p=one(p,"SCOPE=V/'ROOT_PROSPECTIVE_FROZEN_REAL_BENCHMARK_SCOPE.json'","SCOPE=V/'ROOT_PROSPECTIVE_DENSE_POINT_HEADS_FROZEN_BENCHMARK_SCOPE.json'")
    p=one(p,"INHERITED_PROTOCOL=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200/prediction_preparation_a/PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'","INHERITED_PROTOCOL=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203/prediction_preparation_a/PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'")
    p=one(p,"FIT_PROTOCOL=FITPHASE/'fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'","FIT_PROTOCOL=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'")
    p=p.replace('frozen_C60_C61_real_predictions_v1','frozen_C62_C63_real_predictions_v1')
    p=one(p,"ARMS=(('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL',False,False),('C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL',False,True))","ARMS=(('C62_DENSE_POINT_RAW_Q8',False,True),('C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',False,True))\nARM_WEAK={ARMS[0][0]:(False,0.),ARMS[1][0]:(True,.25)}")
    p=replace_function(p,'prepare',SOURCE_HELPERS+'\n'+PRED_PREP)
    p=one(p,"for key in ['decoder_spec'", "p=resolve_source_bindings(e,p)\n for key in ['decoder_spec'")
    start=p.index(" require(all(review.get(k)is True")
    end=p.index(" source=read(RAW10)",start)
    p=p[:start]+" source_entry=read(checked(e['source_fit_entry']));barrier=read(checked(p['source_both_final_freeze']))\n validate_source_metadata(fit,review,e,p,source_entry,barrier)\n"+p[end:]
    p=p.replace("qr=review['results'][arm]['checkpoint'];require(qr['PASS']is True and qr['parameters']==118359 and qr['state_tensors']==36 and qr['update4000']is True and qr['use_sam']is False and qr['spatial_q8']is spatial_q8 and qr['final_hash_exact']is True,'Independent checkpoint typed integrity')","require(len(model.state_dict())==36,'Exact36 source state tensors')")
    p=one(p,"and payload['protocol']==p['source_fit_protocol'],'Exact final checkpoint metadata')","and payload['protocol']==p['source_fit_protocol']and payload['weak_bridge']is ARM_WEAK[arm][0]and payload['weak_loss_weight']==ARM_WEAK[arm][1],'Exact final checkpoint metadata')")
    p=p.replace("updates=4000,source_final_state_sha=f['final_state_sha'],use_sam=use_sam,spatial_q8=spatial_q8,outputs=outputs","updates=4000,source_final_state_sha=f['final_state_sha'],use_sam=use_sam,spatial_q8=spatial_q8,weak_bridge=ARM_WEAK[arm][0],weak_loss_weight=ARM_WEAK[arm][1],outputs=outputs")
    p=one(p,"source_fit_complete=e['fit_complete'],source_fit_integrity_review=", "source_fit_protocol=p['source_fit_protocol'],source_fit_trainer=p['source_fit_trainer'],source_both_final_freeze=p['source_both_final_freeze'],source_fit_complete=e['fit_complete'],source_fit_integrity_review=")
    assert "qr=review['results']"not in p
    e=eb.decode()
    e=e.replace("OLD200=P/", "OLD203=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203'\nOLD200=P/",1)
    e=one(e,"INHERITED_PROTOCOL=OLD200/'evaluation_preparation_a/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'","INHERITED_PROTOCOL=OLD203/'evaluation_preparation_a/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'")
    e=e.replace("SCOPE = V/'ROOT_PROSPECTIVE_FROZEN_REAL_BENCHMARK_SCOPE.json'","SCOPE = V/'ROOT_PROSPECTIVE_DENSE_POINT_HEADS_FROZEN_BENCHMARK_SCOPE.json'")
    e=e.replace("V/'prediction_preparation_a/infer_both_frozen_spatial_q8_heads.py'",f"H/'{PRED}'").replace("V/'prediction_preparation_a/PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'","H/'PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'").replace("V/'prediction_preparation_a/PREPARATION_COMPLETE.json'","H/'PREDICTION_PREPARATION_COMPLETE.json'")
    e=e.replace('frozen_C60_C61_real_predictions_v1','frozen_C62_C63_real_predictions_v1')
    e=one(e,"ARMS = ('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL')","ARMS = ('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')")
    e=one(e,"ARM_SPATIAL=dict(zip(ARMS,(False,True)))","ARM_SPATIAL=dict(zip(ARMS,(True,True)))\nARM_WEAK={ARMS[0]:(False,0.),ARMS[1]:(True,.25)}\nPRIOR_SPATIAL_ARMS=('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL')\nPRIOR_SPATIAL_TAGS=('C60','C61')\nPRIOR_SPATIAL_PRED=OLD203/'frozen_C60_C61_real_predictions_v1/PREDICTION_SEAL.json'")
    e=one(e,"TAGS = ('C60','C61')","TAGS = ('C62','C63')")
    e=e.replace('PRIOR_NATIVE_TAGS+TAGS','PRIOR_NATIVE_TAGS+PRIOR_SPATIAL_TAGS+TAGS').replace('OLD_TAGS=REGIONAL_TAGS+PRIOR_DIRECT_TAGS+PRIOR_GROUPED_TAGS+PRIOR_NATIVE_TAGS','OLD_TAGS=REGIONAL_TAGS+PRIOR_DIRECT_TAGS+PRIOR_GROUPED_TAGS+PRIOR_NATIVE_TAGS+PRIOR_SPATIAL_TAGS')
    e=e.replace("'C60_direct_total_U'","'C62_direct_total_U'").replace("'C61_direct_total_U'","'C63_direct_total_U'")
    e=e.replace('prior_ten','prior_twelve').replace('historical_ten','historical_twelve').replace('old_ten','old_twelve').replace('Old10','Old12').replace('old10','old12').replace('All12','All14').replace('all12','all14').replace('ALL11','ALL13').replace('public11','public13')
    e=e.replace('len(MODELS)==12 and len(BASELINES+OLD_TAGS)==10 and len(ALLTAGS)==10 and len(DIRECT_TAGS)==8','len(MODELS)==14 and len(BASELINES+OLD_TAGS)==12 and len(ALLTAGS)==12 and len(DIRECT_TAGS)==10')
    e=e.replace('C60/C61 fixed spatial-q8 RAW','C62/C63 same spatial-q8 RAW dense-point/weak-bridge')
    e=replace_function(e,'prepare',EVAL_PREP)
    extra="""def prior_spatial_outputs(seal,p):
    require(seal['completed']and seal['PASS']and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'Prior V203 both saved predictions complete')
    require(seal['code']==p['prior_spatial_prediction_code']and seal['protocol']==p['prior_spatial_prediction_protocol']and seal['input_complete']==p['input_complete']and seal['original_raw_complete']==p['original_raw_complete'],'Exact old V203 saved-only provenance')
    require(set(seal['arms'])==set(PRIOR_SPATIAL_ARMS),'Both old spatial arms retained')
    for i,arm in enumerate(PRIOR_SPATIAL_ARMS):
        r=seal['arms'][arm];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is bool(i)and set(r['outputs'])==set(DOMAINS),'Old spatial ablation flags unchanged')
    return {tag:seal['arms'][arm]['outputs']for tag,arm in zip(PRIOR_SPATIAL_TAGS,PRIOR_SPATIAL_ARMS)}

"""
    pos=e.index('def generated_checks():');e=e[:pos]+extra+e[pos:]
    e=one(e,"p=read(PROTOCOL);require(read(check(p['checks']))", "p=read(PROTOCOL);require(read(check(p['checks']))")
    e=one(e,"seal=read(check(e['prediction_seal']));outputs=validate_prediction_seal(seal,p)","seal=read(check(e['prediction_seal']));outputs=validate_prediction_seal(seal,p)\n    p['source_fit_protocol']=seal['source_fit_protocol']")
    e=e.replace("fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']","fit['all2_frozen_before_any_SOURCE_outputs']")
    marker="        prior_twelve=read(check(p['prior_twelve_evaluation_complete']));"
    assert marker in e
    e=e.replace(marker,"        require(e['prior_spatial_prediction_seal']['path']==str(PRIOR_SPATIAL_PRED),'Exact V203 saved spatial seal');outputs.update(prior_spatial_outputs(read(check(e['prior_spatial_prediction_seal'])),p))\n        require(e['prior_twelve_evaluation_complete']['path']==p['prior_twelve_evaluation_complete_path'],'Exact V203 old12 evaluation');p['prior_twelve_evaluation_complete']=e['prior_twelve_evaluation_complete']\n"+marker)
    e=e.replace('old_C52_to_C59_saved_only_rows17152=True','old_C52_to_C61_saved_only_rows21440=True')
    for text,name in [(p,PRED),(e,EVAL)]:
        ast.parse(text)
        with(H/name).open('x')as f:f.write(text)
    print('DERIVED_CODE_ONLY',hashlib.sha256(p.encode()).hexdigest(),hashlib.sha256(e.encode()).hexdigest())

if __name__=='__main__':main()
