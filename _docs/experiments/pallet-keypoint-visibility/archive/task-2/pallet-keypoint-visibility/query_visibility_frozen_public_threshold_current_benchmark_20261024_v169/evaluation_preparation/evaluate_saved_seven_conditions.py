"""Saved-only V169 seven-condition benchmark; no model or threshold fitting.

--prepare reads code/protocol and the completed prediction seal metadata only.
Label, saved-score and frozen annotation-ledger bodies require a separate entry.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from collections import Counter
import argparse,ast,csv,hashlib,json,math,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_REQUIREMENTS_AND_DIAGNOSTIC_POLICY.json'
POLICY_SHA='faa8d31eecc7d35976ed1e5dee100fda6e537123c25ea3557a571e1f024b43f7'
OLD=P/'query_visibility_separated_frozen_current_benchmark_20261024_v156'
OLD_CODE=OLD/'evaluation_preparation/evaluate_saved_seven.py'
OLD_PROTOCOL=OLD/'evaluation_preparation/PROTOCOL.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_CURRENT_EVALUATION_BODIES.json'
PRED=V/'frozen_two_head_four_condition_predictions_v1/PREDICTION_SEAL.json'
PRED_SHA='480d9a6439b978908b23b2fdb0d83e7d84f1bbfd1bcad8939cc40079871b0d1f'
OUT=V/'saved_seven_condition_evaluation_v1'
CONDITIONS=('C45_ORIGINAL0','C45_PUBLICCAL_FULL','C46_ORIGINAL0','C46_PUBLICCAL_FULL')
BASELINES=('v61','legacy','C16');NEW_MODELS=CONDITIONS;MODELS=(*BASELINES,*CONDITIONS)
FIELDS=('id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision')
META=('query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside','query_xy','side_L','original_query_xy','original_side_L','id','native_corner')
CORE=P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py'
DECODER=CORE.parent/'infer_same_query_once.py'
ADAPTER=P/'query_visibility_training_duration_control_20261023_v124/current_evaluation_preparation/frozen_head_current.py'
REPORT=P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/evaluate_current_saved.py'
METRIC=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
PRIOR_CODE=P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149/saved_evaluation_preparation/evaluate_saved_benchmark.py'

def require(x,why):
    if not x:raise ValueError(why)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    require('_v112' not in b['path'].lower(),'Forbidden phase path');require(bind(b['path'])==b,'Changed binding '+b['path']);return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def byte_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def functions(path,names,ns,rename=None):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Pure function coverage')
    if rename:
        class Aliases(ast.NodeTransformer):
            def visit_Constant(self,node):
                return ast.copy_location(ast.Constant(rename[node.value]),node) if isinstance(node.value,str) and node.value in rename else node
        nodes=[Aliases().visit(n) for n in nodes]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::pure','exec'),ns)
def pure():
    ns=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(require=require),Path=Path,csv=csv,require=require,
      NEW=CONDITIONS[0],OLD=CONDITIONS[1],NEW_GEO='UNREPORTED_GEOMETRY0',OLD_GEO='UNREPORTED_GEOMETRY1',MODELS=MODELS,MINIMAL_SCORE_FIELDS=FIELDS)
    functions(CORE,{'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'},ns)
    functions(DECODER,{'decode_rows'},ns);functions(ADAPTER,{'enrich_minimal_baseline'},ns)
    core=SimpleNamespace(**ns)
    base=ModuleType('original_visibility_metric');base.__dict__.update(math=math,Counter=Counter,D='DIRECT_VISIBLE',E='EXTERNAL_OCCLUDED',S='SELF_OCCLUDED',
      TASKS={'D_vs_E':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED'],'D_vs_S':['DIRECT_VISIBLE','SELF_OCCLUDED'],'D_vs_ES':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED']},GROUPS=['WOOD_PRIMARY105','SQUARE118'])
    functions(METRIC,{'yes','ratio','call','calculate'},base.__dict__)
    ex=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(NEW=CONDITIONS[0],OLD=CONDITIONS[1],NEW_GEO='UNREPORTED_GEOMETRY0',OLD_GEO='UNREPORTED_GEOMETRY1',MODELS=MODELS))
    functions(REPORT,{'error_bins','matrices'},ex)
    helpers=dict(np=np,math=math,csv=csv,Path=Path,read=read,check=check,require=require,byte_equal=byte_equal,META=META,FIELDS=FIELDS)
    functions(PRIOR_CODE,{'load_packet','records','exact_csv_rows'},helpers)
    tr=dict(Counter=Counter,NEW_MODELS=NEW_MODELS,BASELINES=BASELINES,require=require)
    functions(OLD_CODE,{'transitions'},tr,dict(zip(('C35','C36','C37','C38'),CONDITIONS)))
    return core,base,SimpleNamespace(**ex),SimpleNamespace(**helpers),tr['transitions']
def decision_state(pred):return 'OTHER_UNRESOLVED' if pred<0 else 'VISIBLE' if pred==0 else 'ABSENT_NONVISIBLE'
def decision_arrays(raw,available,t):
    require(raw.dtype==np.float32 and raw.shape==available.shape and available.dtype==bool,'Raw schema')
    score=raw.astype(np.float64)-np.float64(t)
    pred=np.where(available,(raw.astype(np.float64)>=np.float64(t)).astype(np.int8),-1)
    require(np.array_equal(pred,np.where(available,(score>=0).astype(np.int8),-1)),'Float64 sign equivalence')
    return score,pred
def condition_records(core,helpers,prior,condition,reference):
    minimal=core.readcsv(check(condition['scores_csv']));a=helpers.load_packet(check(condition['raw_packet']))
    require(len(minimal)==1784 and all(tuple(r)==FIELDS for r in minimal),'Original nine-field CSV schema')
    require(all(byte_equal(a[k],reference[k]) for k in META),'All original input metadata bytes')
    raw=a['raw_head_logit_before_U'];require(raw.dtype==np.float32 and raw.shape==(1784,) and np.isfinite(raw).all(),'Raw FP32 original logit')
    require(a['raw_three_logits'].shape==(1784,3) and byte_equal(a['raw_three_logits'][:,0].copy(),raw),'Original raw alias')
    require(byte_equal(a['query_logit'],np.where(a['valid'],raw,0).astype(np.float32)),'Original safe alias')
    t=a['threshold_logit_float64'];require(t.shape==() and t.dtype==np.float64 and float(t)==condition['threshold_float64'],'Frozen scalar')
    score,pred=decision_arrays(raw,a['valid'],float(t));require(byte_equal(score,a['decision_score64']) and byte_equal(pred,a['prediction_D0_E1_Uminus1']),'Saved authoritative decision')
    rawminimal=[]
    for i,r in enumerate(minimal):
        require(core.key(r)==(str(a['id'][i]),str(int(a['native_corner'][i]))),'Original native order')
        require(core.boolean(r['valid'])==bool(a['valid'][i]),'Input availability')
        require(float(r['raw_query_aux_hidden_logit'])==float(a['query_logit'][i]) and float(r['raw_p_visible_FP32'])==float(a['p_visible_FP32'][i]),'Source raw/probability CSV alias')
        require(r['authoritative_decision']==('UNKNOWN' if pred[i]<0 else 'D' if pred[i]==0 else 'E'),'Saved CSV threshold decision')
        # The legacy metadata enricher accepts only raw0; feed a temporary raw0
        # decision while restoring metadata, never alter the actual CSV/NPZ.
        rawminimal.append(dict(r,authoritative_decision=('D' if a['query_logit'][i]<0 else 'E') if a['valid'][i] else 'UNKNOWN'))
    rows=core.enrich_minimal_baseline(core,prior,rawminimal)
    for i,r in enumerate(rows):
        # In-memory legacy-kernel aliases are decision scores, explicitly not
        # persisted raw logits or calibrated probabilities. Keep float64 here.
        r['raw_query_aux_hidden_logit']=repr(float(score[i]));r['authoritative_decision']='UNKNOWN' if pred[i]<0 else 'D' if pred[i]==0 else 'E'
        r['decision_visible']='' if pred[i]<0 else str(pred[i]==0)
    index=core.score_index(rows,'raw_query_aux_hidden_logit')
    require(all(core.response(index[core.key(r)])==decision_state(int(pred[i])) for i,r in enumerate(rows)),'Exact kernel response/saved pred parity')
    return rows,a

def generated():
    core,base,_,_,transitions=pure();checks={}
    require(core.response(dict(valid=False,z=-100,geo=True))=='OTHER_UNRESOLVED','U first')
    checks['historical_Ufirst_and_zero_boundary']=core.response(dict(valid=True,z=0,geo=False))=='ABSENT_NONVISIBLE'
    rr=[dict(id='I',human_state=base.D,present_at_fixed_0_5='True',legacy_stored_state='U'),dict(id='I',human_state=base.E,present_at_fixed_0_5='False',legacy_stored_state='S')]
    x=base.calculate(rr,'legacy','D_vs_E',{},.5);checks['legacy_U_and_out_of_taskS_denominators']=x['N']==2 and x['U_visible']==1 and x['out_of_task_S_occluded']==1
    raw=np.array([-100.,-1.,-0.,0.,1.],np.float32);av=np.array([False,True,True,True,True])
    for t in (0.,-10.97239875793457,-9.855539321899414,np.nextafter(np.float64(1),np.inf)):
        score,pred=decision_arrays(raw,av,t)
        require([core.response(dict(valid=bool(av[i]),z=float(score[i]),geo=False)) for i in range(5)]==[decision_state(int(x)) for x in pred],'Float64 saved-decision/kernel parity')
    checks['four_condition_f64_kernel_equivalence_and_inputU']=True
    checks['no_float32_threshold_downcast']=decision_arrays(np.array([1],np.float32),np.array([True]),np.nextafter(np.float64(1),np.inf))[1].tolist()==[0]
    checks['geometry_route_separate']=core.response(dict(valid=True,z=-1,geo=True),False)=='VISIBLE' and core.response(dict(valid=True,z=-1,geo=True),True)=='ABSENT_NONVISIBLE'
    # Exercise the transplanted function, including its require dependency.
    # Four invented rows: each original material group has one D and one E.
    toy=[dict(id=group+state,group=group,human_state=state,
        paired_same_query_RGB_D_vs_E='True',same_slot_pipeline_D_vs_E='True')
        for group in base.GROUPS for state in (base.D,base.E)]
    calls=[]
    def correct_call(row,model):
        calls.append((row['id'],model))
        return 'VISIBLE' if row['human_state']==base.D else 'ABSENT_NONVISIBLE'
    panels=transitions(core,base,toy,correct_call)
    require(len(panels)==192,'2 cohorts x2 groups x3 truth classes x16 pairs')
    zero_keys=('baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong',
        'baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
    for panel in panels:
        expected=2 if panel['truth_class']=='ALL_DE' else 1
        require(panel['N']==panel['both_correct']==expected and all(panel[k]==0 for k in zero_keys),'Every transplanted transition panel denominator/count')
    require(sum(panel['N'] for panel in panels)==256 and len(calls)==512,'Full invented transition membership')
    require(len({(q['cohort'],q['group'],q['truth_class'],q['baseline'],q['new_model']) for q in panels})==192,'All unique transition panels')
    checks['transplanted_transitions192_panels_all_N_and_bothcorrect_exact']=True
    require(all(checks.values()),'Generated checks')
    return dict(completed=True,PASS=True,invented_only=True,checks=checks,number_checks=len(checks),transition_regression=dict(rows=4,panels=192,total_panel_memberships=256,correct_call_count=512,all_N_and_both_correct_exact=True),current_label_score_RGB_or_annotation_body_or_hash_reads=0)
def prepare():
    require(sha(POLICY)==POLICY_SHA and sha(PRED)==PRED_SHA,'Frozen policy/completed prediction seal')
    old=read(OLD_PROTOCOL);seal=read(PRED)
    require(seal['completed'] and seal['PASS'] and tuple(seal['conditions'])==CONDITIONS,'All four saved conditions')
    for b in old['operator_codes']+[old['prior_code']]:check(b)
    checks=generated();write(H/'GENERATED_INVENTED_EVALUATION_CHECKS.json',checks)
    quality=read(H/'ROOT_EXISTING_FROZEN_QUALITY_LEDGER_BINDING.json')
    # Descriptor supplied by root; no current CSV/hash/body read at preparation.
    require(Path(quality['path']).name=='ALL1784_POINT_QUALITY_AUTHORITY_AND_TRACEBACK.csv','Frozen old provenance ledger only')
    keys=('operator_codes','C16_scores','C16_raw','historical_C16_metrics','prior','truth','manual','historical_original','historical_three_tables','annotation_authority')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),prediction_seal=bind(PRED),prior_code=bind(OLD_CODE),prior_protocol=bind(OLD_PROTOCOL),models=list(MODELS),conditions=list(CONDITIONS),output=str(OUT),generated=bind(H/'GENERATED_INVENTED_EVALUATION_CHECKS.json'),frozen_annotation_ledger=quality,**{k:old[k] for k in keys},
      fixed_contract='Original paired1237 primary/same-slot1248/manualnear376/all1784 and1739valid45U unchanged; humanS/OOF/unknown explicit; identical metric/mapping kernels; legacyCAL overlap and repeated development exposure disclosed.',
      new_adapter='Original disk rawFP32/probability preserved. Saved int8pred authoritative; temporary float64 decision-score aliases only within unchanged historical sign kernel, verified>= equivalence. Never call shifted sigmoid probability calibration.',
      annotation_scope='Copy only existing frozen V156 provenance ledger fields; no raw annotation files, no new traceback, no new quality gold. Primary visibility never quality/error gated.',
      current_label_score_RGB_or_annotation_body_hash_reads_during_prepare=0,selected_model=None)
    write(PROTOCOL,p)
    template=dict(approved_saved_evaluation=False,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],prediction_seal=p['prediction_seal'],selected_model=None,private_fit=False,threshold_or_model_choice=False,RGB_read=False,encoder_forward=False,V112=False,PnP=False,q_L_correction=False,new_raw_annotation_traceback=False)
    write(H/'ROOT_SAVED_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    receipt=dict(completed=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),generated=p['generated'],prediction_seal=p['prediction_seal'],entry_template=bind(H/'ROOT_SAVED_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json'),current_label_score_RGB_or_annotation_body_hash_reads=0,command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).resolve())+' --entry '+str(V/'ROOT_APPROVED_ACTUAL_SAVED_EVALUATION_ENTRY.json'))
    write(H/'PREPARATION_COMPLETE.json',receipt);print(json.dumps(receipt))

def run(entry):
    e=read(entry);require(e['approved_saved_evaluation'] is True and e['selected_model'] is None,'Separate root saved evaluation entry')
    require(all(e[k] is False for k in ('private_fit','threshold_or_model_choice','RGB_read','encoder_forward','V112','PnP','q_L_correction','new_raw_annotation_traceback')),'Scope')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['policy']==bind(POLICY) and sha(POLICY)==POLICY_SHA,'Frozen evaluator')
    p=read(PROTOCOL);require(e['prediction_seal']==p['prediction_seal']==bind(PRED),'Completed prediction binding');seal=read(PRED)
    require(seal['completed'] and seal['PASS'] and tuple(seal['conditions'])==CONDITIONS and seal['selected_model'] is None,'All conditions')
    require(read(check(seal['C16_parity']))['PASS'],'C16 parity first')
    for b in p['operator_codes']+[p['prior_code'],p['prior_protocol'],p['generated']]:check(b)
    require(read(check(p['generated']))['PASS'] and not OUT.exists(),'Invented guards and single output')
    OUT.mkdir();write(OUT/'BEFORE_LABEL_READ.json',dict(completed=True,entry=bind(entry),prediction_seal=p['prediction_seal'],label_bodies_read_so_far=False))
    try:
        core,base,extra,helpers,transitions=pure()
        truth=core.readcsv(check(p['truth']));manual=core.readcsv(check(p['manual']));prior=[json.loads(s) for s in check(p['prior']).read_text().splitlines() if s]
        require(len(truth)==len(prior)==1784 and len({r['id'] for r in truth})==223,'All1784')
        reference=helpers.load_packet(check(p['C16_raw']));require(int(reference['valid'].sum())==1739,'Original1739/45U')
        rows={};packs={};rows['C16'],packs['C16']=helpers.records(core,prior,p['C16_scores'],p['C16_raw'],reference)
        for name in CONDITIONS:rows[name],packs[name]=condition_records(core,helpers,prior,seal['conditions'][name],reference)
        indices={name:core.score_index(value,'raw_query_aux_hidden_logit') for name,value in rows.items()}
        lookup={name:{core.key(r):int(packs[name]['prediction_D0_E1_Uminus1'][i]) for i,r in enumerate(truth)} for name in CONDITIONS}
        original_call=base.call
        def call(r,model,task,probabilities,threshold,rules=None):
            if model in lookup:return decision_state(lookup[model][core.key(r)])
            if model=='C16':return core.response(indices[model][core.key(r)],False)
            return original_call(r,model,task,probabilities,threshold,rules)
        base.call=call
        oldrows=[dict(r,raw_physical_external_logit=r['raw_query_aux_hidden_logit']) for r in rows[CONDITIONS[1]]]
        tables=core.calculate_all(base,truth,prior,rows[CONDITIONS[0]],oldrows,manual)
        tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,rows[CONDITIONS[0]],oldrows)
        tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
        model_call=lambda r,m:call(r,m,'D_vs_E',{},.5)
        tables['paired_recovery_and_damage']=transitions(core,base,truth,model_call)
        replay=core.historical_replay(tables['metrics'],core.readcsv(check(p['historical_original'])),{'v61','legacy'})
        replay16=core.historical_replay([dict(r,model='V124_C16_RAW') for r in tables['metrics'] if r['model']=='C16'],core.readcsv(check(p['historical_C16_metrics'])),{'V124_C16_RAW'})
        require(sum(base.yes(r,'paired_same_query_RGB_D_vs_E') and r['human_state'] in (base.D,base.E) for r in truth)==1237,'Paired1237')
        require(sum(base.yes(r,'same_slot_pipeline_D_vs_E') and r['human_state'] in (base.D,base.E) for r in truth)==1248,'Same-slot1248')
        require({r['N'] for r in tables['manual_near_D']}=={153,223},'Manualnear376 unchanged')
        frozen_quality=core.readcsv(check(p['frozen_annotation_ledger']));require(len(frozen_quality)==1784,'Original frozen provenance ledger1784')
        ledger=[]
        for i,r in enumerate(truth):
            k=core.key(r);qr=frozen_quality[i];require(core.key(qr)==k,'Frozen provenance order')
            d=dict(global_index=i,id=k[0],native_corner=int(k[1]),group=r['group'],human_state=r['human_state'],mapping_status=r['mapping_status'],query_x=float(reference['original_query_xy'][i,0]),query_y=float(reference['original_query_xy'][i,1]),roi_side=float(reference['original_side_L'][i]),query_available=bool(reference['valid'][i]),human_reference_x=r['human_reference_x'],human_reference_y=r['human_reference_y'],paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],initial_reference_error_px=r['initial_reference_error_px'],original_query_valid=r['query_valid'],legacy_original_state=r['legacy_stored_state'])
            for m in MODELS:
                response='OTHER_UNRESOLVED' if m=='v61' and (r['query_valid']!='True' or r.get('present_at_fixed_0_5') not in ('True','False')) else model_call(r,m)
                d[m+'_original_response']=response;d[m+'_D_E_U']='D' if response=='VISIBLE' else 'E' if response=='ABSENT_NONVISIBLE' else 'U'
            d['C16_raw_visibility_logit']=float(packs['C16']['raw_head_logit_before_U'][i])
            for m in CONDITIONS:
                d[m+'_raw_visibility_logit']=float(packs[m]['raw_head_logit_before_U'][i]);d[m+'_threshold_float64']=float(packs[m]['threshold_logit_float64']);d[m+'_decision_score64']=float(packs[m]['decision_score64'][i])
            d['v61_saved_p_present']=r.get('presence_probability','');d['legacy_saved_p_DIRECT']=r.get('legacy_p_DIRECT',r.get('legacy_direct_probability',''))
            for key in ('annotation_coordinate_source','raw_annotation_traceback','new_primary_quality_gold_target','secondary_error_px','secondary_fixed_reference_target'):d[key]=qr.get(key,'UNKNOWN_UNCERTIFIED')
            d['quality_reference_scope']='PREEXISTING_FROZEN_REFERENCE_PROXY_NOT_PHYSICAL_NATIVE_GOLD';ledger.append(d)
        inventory=[]
        for group in ('ALL',*base.GROUPS):
            rr=ledger if group=='ALL' else [r for r in ledger if r['group']==group]
            for m in MODELS:
                cc=Counter(r[m+'_D_E_U'] for r in rr);responses=Counter(r[m+'_original_response'] for r in rr)
                inventory.append(dict(group=group,model=m,all_registered_slots=len(rr),D=cc['D'],E_binary_response=cc['E'],U=cc['U'],U_rate=cc['U']/len(rr),legacy_abstain=responses['ABSTAIN'],out_of_task_geometry_S=responses['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=responses['OTHER_UNRESOLVED'],human_state_inventory=json.dumps(dict(Counter(r['human_state'] for r in rr)),sort_keys=True)))
        tables['all1784_model_inventory']=inventory;historical={}
        for name,b in p['historical_three_tables'].items():
            hist=core.readcsv(check(b))
            if name=='ALL1784_NATIVE_LEDGER':
                fields=[k for k in hist[0] if not any(k.startswith(m+'_') for m in ('C29','C32','C33','C34'))]
                current=[{k:r[k] for k in fields} for r in ledger];hist=[{k:r[k] for k in fields} for r in hist]
            else:
                key=next(k for k in tables if k.upper()==name);current=[r for r in tables[key] if r.get('model') in BASELINES];hist=[r for r in hist if r.get('model') in BASELINES]
            historical[name]=helpers.exact_csv_rows(current,hist)
        for name in ('metrics','confusion','human_self_binary_response','manual_near_D','availability_by_group','coordinate_error_bins','D_vs_external_E_2x3','paired_recovery_and_damage','all1784_model_inventory'):core.writecsv(OUT/(name.upper()+'.csv'),tables[name])
        core.writecsv(OUT/'ALL1784_NATIVE_LEDGER.csv',ledger)
        result=dict(completed=True,PASS=True,models=list(MODELS),tables=tables,original_slots=1784,WOOD_frames=105,SQUARE_frames=118,original_input_valid=1739,original_input_U=45,additional_U=0,cohort_denominators=dict(paired_same_query_RGB=1237,same_slot_pipeline=1248),historical_v61_legacy_rows_exact=replay,historical_C16_rows_exact=replay16,historical_three_all_table_fields_exact=historical,frozen_annotation_ledger=p['frozen_annotation_ledger'],annotation_authority=p['annotation_authority'],new_primary_real_quality_gold_known=0,primary_visibility_never_quality_error_or_auxiliary_gated=True,self='HumanS binary response only; learned external score does not identify semantic self. Existing geometry route stays separate.',coordinate_reference='Frozen reference/manual<=3px proxy only, not physical native gold.',legacy_CAL_overlap_and_development_exposed_not_blind=True,prediction_seal=p['prediction_seal'],policy=p['policy'],new_forward_fit_RGB_encoder_PnP_qL_or_threshold_choice=0,selected_model=None,adopted=False,fully_solved=False)
        write(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json',result)
        outputs=[bind(x) for x in OUT.iterdir() if x.is_file()]
        write(OUT/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),policy=p['policy'],entry=bind(entry),prediction_seal=p['prediction_seal'],outputs=outputs,selected_model=None,adopted=False,source_admission_changed=False,new_forward_fit_RGB_PnP=0))
        print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'EVALUATION_SEAL.json'))))
    except BaseException:
        write(OUT/'FAILED_EVALUATION.json',dict(completed=False,traceback=__import__('traceback').format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args()
    prepare() if args.prepare else run(args.entry)
