"""V178 saved five-model diagnostic. Preparation reads code/small metadata + invented records only."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from collections import Counter
import argparse
import ast
import csv
import hashlib
import json
import math
import traceback

import numpy as np

HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_CURRENT_FEATURE_CHECKPOINT_SCORE_OR_LABEL_BODIES.json'
SCOPE_SHA='83f065b096035acff3dec30de8b941199b00934d12d580ecff718927f0a89228'
CAUSE=V/'ROOT_CAUSE_SEPARATION_READOUT_BEFORE_CURRENT_LABEL_AND_SCORE_BODIES.json'
CAUSE_SHA='ecf869269c883c1d6495dec6659b1108d32443131aa32cc33b96a4e91d7cd54b'
PRED=V/'frozen_all3_direct_owner_predictions_v1/PREDICTION_SEAL.json'
PRED_SHA='0d23482007351b027deb10668bc6d0f08fa13f1fec4f5a0308f900530ee01401'
PROTOCOL=HERE/'PROTOCOL_BEFORE_SAVED_CURRENT_SCORE_OR_LABEL_BODIES.json'
CHECKS=HERE/'GENERATED_INVENTED_EVALUATION_CHECKS.json'
PREP=HERE/'PREPARATION_COMPLETE.json'
OUT=V/'saved_five_model_evaluation_v1'
OLD=P/'query_visibility_frozen_public_threshold_current_benchmark_20261024_v169'
CORE=P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py'
DECODER=CORE.parent/'infer_same_query_once.py'
METRIC=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
REPORT=P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/evaluate_current_saved.py'
HELPER=P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149/saved_evaluation_preparation/evaluate_saved_benchmark.py'
FORWARD=V/'prediction_preparation/infer_frozen_direct_owner_all3.py'
BASELINES=('v61','legacy');NEW_MODELS=('C47','C48','C49');MODELS=BASELINES+NEW_MODELS
ARMS=('C47_PUBLIC_WEAK253_ONLY','C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC','C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314')
CLASSES=['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
META=('query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside',
      'query_xy','side_L','original_query_xy','original_side_L','id','native_corner')
PAIRS=tuple((a,b) for b in NEW_MODELS for a in BASELINES)+(('C47','C48'),('C48','C49'),('C47','C49'))
FIXED19=dict(path=str(OLD/'fixed_current_crop_review_v1/CASE_IDS_BEFORE_ANY_RGB.json'),sha256='db1f1303e9faebf675971696d2f60e54caf97a52beddd64157d11d2029a8735f')
FIXED19_POLICY=dict(path=str(OLD/'crop_review_preparation/SELECTION_RULE_BEFORE_CURRENT_BODIES.json'),sha256='cf64a0c947b713fc040aae37920b92a712a77c95569dfa96aafc8b57bddfe345')
FIXED19_RENDER=dict(path=str(OLD/'fixed_current_crop_review_v1/CROP_RENDER_COMPLETE.json'),sha256='9f47236d47af91e3667cd18589106deed2e324072cf29291b96c6f1286b219cd')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    require('_v112' not in str(path).lower(),'Forbidden path')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):require(bind(b['path'])==b,'Changed binding');return Path(b['path'])
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def functions(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Pure function coverage')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure','exec'),ns)


def pure():
    ns=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(require=require),Path=Path,csv=csv,require=require,
            NEW='C47',OLD='C48',NEW_GEO='UNREPORTED_GEOMETRY0',OLD_GEO='UNREPORTED_GEOMETRY1',MODELS=MODELS)
    functions(CORE,{'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'},ns)
    functions(DECODER,{'decode_rows'},ns);core=SimpleNamespace(**ns)
    base=ModuleType('original_five_model_metric');base.__dict__.update(math=math,Counter=Counter,D='DIRECT_VISIBLE',E='EXTERNAL_OCCLUDED',S='SELF_OCCLUDED',
        TASKS={'D_vs_E':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED'],'D_vs_S':['DIRECT_VISIBLE','SELF_OCCLUDED'],'D_vs_ES':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED']},GROUPS=['WOOD_PRIMARY105','SQUARE118'])
    functions(METRIC,{'yes','ratio','call','calculate'},base.__dict__)
    ex=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(NEW='C47',OLD='C48',NEW_GEO='UNREPORTED_GEOMETRY0',OLD_GEO='UNREPORTED_GEOMETRY1',MODELS=MODELS))
    functions(REPORT,{'error_bins','matrices'},ex)
    helper=dict(require=require);functions(HELPER,{'exact_csv_rows'},helper)
    decoder=dict(np=np,require=require);functions(FORWARD,{'decode'},decoder)
    return core,base,SimpleNamespace(**ex),helper['exact_csv_rows'],decoder['decode']


def decision_state(pred):return 'OTHER_UNRESOLVED' if pred<0 else 'VISIBLE' if pred==0 else 'ABSENT_NONVISIBLE'


def assert_row_order(core,truth,prior):
    require(len(truth)==len(prior)==1784,'Original1784 row-order census')
    truth_keys=[core.key(r) for r in truth];prior_keys=[core.key(r) for r in prior]
    require(len(set(truth_keys))==len(set(prior_keys))==1784,'Truth/prior unique native keys')
    require(truth_keys==prior_keys,'Truth/prior original key sequence changed')


def decision_rows(core,prior,pred):
    # In-memory sign sentinels for the unchanged historical scalar kernel only.
    # They are not raw owner logits/probabilities and are never exported as such.
    require(pred.dtype==np.int8 and pred.shape==(len(prior),) and np.isin(pred,[-1,0,1]).all(),'Saved binary schema')
    sign=np.where(pred==0,-1.,1.).astype(np.float32)
    display=np.full(len(pred),.5,np.float32)
    pp=[dict(r,prior_reason=r.get('prior_reason',r.get('reason',''))) for r in prior]
    rows=core.decode_rows(pp,sign,display,pred>=0)
    index=core.score_index(rows,'raw_query_aux_hidden_logit')
    require(all(core.response(index[core.key(r)])==decision_state(int(pred[i])) for i,r in enumerate(rows)),'Stored prediction/kernel equivalence')
    return rows


def load_owner(binding,decode,reference=None):
    with np.load(check(binding),allow_pickle=False) as f:
        a={k:f[k].copy() for k in META}
        raw=f['raw_owner_three_logits'].copy();owner=f['owner_prediction'].copy();pred=f['prediction_D0_E1_Uminus1'].copy()
        require(f['owner_class_order'].astype(str).tolist()==CLASSES and np.array_equal(f['global_index'],np.arange(1784)),'Frozen owner order/indices')
    require(raw.dtype==np.float32 and raw.shape==(1784,3) and owner.dtype==pred.dtype==np.int8,'Frozen raw schema')
    require(all(a[k].shape==(1784,) and a[k].dtype==bool for k in META[:6]),'Original input flag schemas')
    require(np.array_equal(a['query_available'],a['valid']) and int(a['valid'].sum())==1739,'Original1739/U45')
    oo,bb=decode(raw,a['valid']);require(eq(oo,owner) and eq(bb,pred),'Original argmax/U/binary parity')
    if reference is not None:require(all(eq(a[k],reference[k]) for k in META),'All12 original metadata bytes')
    return a,raw,owner,pred


def paired_transitions(base,truth,call):
    out=[]
    for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
        for group in base.GROUPS:
            for cls in ('ALL_DE',base.D,base.E):
                rr=[r for r in truth if r['group']==group and base.yes(r,cohort+'_D_vs_E') and r['human_state'] in (base.D,base.E) and (cls=='ALL_DE' or r['human_state']==cls)]
                for old,new in PAIRS:
                    cc=Counter()
                    for r in rr:
                        a,b=call(r,old),call(r,new);gt='VISIBLE' if r['human_state']==base.D else 'ABSENT_NONVISIBLE'
                        if b not in ('VISIBLE','ABSENT_NONVISIBLE'):cc['new_unresolved']+=1
                        elif a not in ('VISIBLE','ABSENT_NONVISIBLE'):cc['baseline_unresolved_new_correct' if b==gt else 'baseline_unresolved_new_wrong']+=1
                        else:cc['both_correct' if a==gt and b==gt else 'baseline_correct_new_wrong' if a==gt else 'baseline_wrong_new_correct' if b==gt else 'both_wrong']+=1
                    keys=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
                    require(sum(cc.values())==len(rr),'Transition denominator')
                    out.append(dict(cohort=cohort,group=group,task='D_vs_E',truth_class=cls,baseline=old,new_model=new,N=len(rr),**{k:cc[k] for k in keys}))
    return out


def error_bucket(row):
    try:v=float(row.get('initial_reference_error_px',''))
    except (TypeError,ValueError):v=float('nan')
    return 'MISSING_REFERENCE_ERROR' if not math.isfinite(v) else 'LE3' if v<=3 else 'GT3'


def owner_native_cross(truth,owners,available,base):
    out=[]
    for group in ('ALL',*base.GROUPS):
        for bucket in ('ALL','LE3','GT3','MISSING_REFERENCE_ERROR'):
            ix=[i for i,r in enumerate(truth) if (group=='ALL' or r['group']==group) and (bucket=='ALL' or error_bucket(r)==bucket)]
            for tag,op in owners.items():
                for state in (base.D,base.E,base.S,'UNKNOWN'):
                    jj=[i for i in ix if (truth[i]['human_state'] if truth[i]['human_state'] in (base.D,base.E,base.S) else 'UNKNOWN')==state]
                    cc=Counter(int(op[i]) for i in jj)
                    out.append(dict(group=group,error_bin=bucket,model=tag,original_native_visibility=state,N=len(jj),
                        owner_T=cc[0],owner_F=cc[1],owner_BG=cc[2],owner_U=cc[-1],original_input_U=sum(not bool(available[i]) for i in jj),
                        raw_original_state_inventory=json.dumps(dict(Counter(truth[i]['human_state'] for i in jj)),sort_keys=True),
                        scope='Association only: predicted F/BG is not certified physical native external occlusion; q-reference error is an existing proxy.'))
    return out


def generated():
    core,base,_,_,decode=pure();checks=[]
    raw=np.array([[0,0,0],[1,2,2],[0,-1,1],[np.nan,np.nan,np.nan]],np.float32)
    owner,pred=decode(raw,np.array([True,True,True,False]));require(owner.tolist()==[0,1,2,-1] and pred.tolist()==[0,1,1,-1],'Argmax/binary')
    checks.append('Original_forward_decoder_first_tie_and_U_first_T_D_F_BG_E')
    prior=[dict(id=f'I{i//8}',native_corner=i%8,valid=i<1739,geometry_self_hidden=(i%2==0),reason='INPUT_U' if i>=1739 else '') for i in range(1784)]
    original_truth=[dict(r) for r in prior];assert_row_order(core,original_truth,prior)
    swapped=[dict(r) for r in original_truth];swapped[0],swapped[1]=swapped[1],swapped[0]
    duplicated=[dict(r) for r in original_truth];duplicated[-1]=dict(duplicated[0])
    rejected=0
    for tt,pp in ((swapped,prior),(duplicated,prior),(original_truth,duplicated)):
        try:assert_row_order(core,tt,pp)
        except ValueError:rejected+=1
    require(rejected==3,'Swapped or duplicate original native keys accepted')
    checks.append('Full1784_truth_prior_original_key_sequence_and_both_sides_uniqueness_failclosed')
    pp=np.tile(np.array([0,1],np.int8),892);pp[1739:]=-1
    rr=decision_rows(core,prior,pp);require(sum(not r['valid'] for r in rr)==45,'Original45U adapter')
    require([r['authoritative_decision'] for r in rr[:2]]==['D','E'],'Sign adapter')
    checks.append('Actual_decision_row_adapter_full1784_kernel_equivalence_and45U_no_fake_disk_raw')
    require(core.response(dict(valid=True,z=-1,geo=True),False)=='VISIBLE' and core.response(dict(valid=True,z=-1,geo=True),True)=='ABSENT_NONVISIBLE','Separate geometry gate')
    checks.append('Original_geometry_route_separate_not_new_owner_gate')
    legacy=[dict(id='I',human_state=base.D,present_at_fixed_0_5='True',legacy_stored_state='U'),dict(id='I',human_state=base.E,present_at_fixed_0_5='False',legacy_stored_state='S')]
    z=base.calculate(legacy,'legacy','D_vs_E',{},.5);require(z['N']==2 and z['U_visible']==1 and z['out_of_task_S_occluded']==1,'Legacy abstention')
    checks.append('Legacy_U_and_out_of_task_self_denominators_preserved')
    toy=[dict(id=group+state,group=group,human_state=state,paired_same_query_RGB_D_vs_E='True',same_slot_pipeline_D_vs_E='True',initial_reference_error_px='3') for group in base.GROUPS for state in (base.D,base.E)]
    panels=paired_transitions(base,toy,lambda r,m:'VISIBLE' if r['human_state']==base.D else 'ABSENT_NONVISIBLE')
    require(len(panels)==108 and sum(r['N'] for r in panels)==144,'All108 fixed transition panels')
    require(all(r['both_correct']==r['N']==(2 if r['truth_class']=='ALL_DE' else 1) for r in panels),'Panel exact counts')
    checks.append('All108_fixed_paired_transition_panels_N_and_bothcorrect_exact')
    own={'C47':np.array([0,1,2,-1],np.int8)}
    xt=owner_native_cross(toy,own,np.array([True,True,True,False]),base)
    require(sum(r['N'] for r in xt if r['group']=='ALL' and r['error_bin']=='ALL')==4,'Cross full denominator')
    require(sum(r['owner_T']+r['owner_F']+r['owner_BG']+r['owner_U'] for r in xt if r['group']=='ALL' and r['error_bin']=='LE3')==4,'Cross owner4 columns')
    require(error_bucket(dict(initial_reference_error_px='3'))=='LE3' and error_bucket(dict(initial_reference_error_px='3.001'))=='GT3' and error_bucket({})=='MISSING_REFERENCE_ERROR','Existing coordinate bins')
    allstates=[dict(r,human_state=state) for r,state in zip(toy,(base.D,base.E,base.S,'UNASSOCIATED'))]
    states=owner_native_cross(allstates,own,np.array([True,True,True,False]),base)
    selected=[r for r in states if r['group']=='ALL' and r['error_bin']=='ALL']
    require([r['N'] for r in selected]==[1,1,1,1] and sum(r['original_input_U'] for r in selected)==1,'Native D/E/S/unknown inventory')
    checks.append('Owner_T_F_BG_U_vs_native_states_and_exact_existing3px_strata_counts')
    require(all(v==0 for k,v in panels[0].items() if k in ('baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','new_unresolved')),'Transitions partition')
    return dict(completed=True,PASS=True,invented_only=True,checks=checks,number_checks=len(checks),
                current_score_label_RGB_annotation_or_feature_checkpoint_body_or_hash_reads=0,new_forward_fit_threshold_model_data_selection=0)


def prepare():
    require(sha(SCOPE)==SCOPE_SHA and sha(CAUSE)==CAUSE_SHA and sha(PRED)==PRED_SHA,'Frozen scope/addendum/completed seal')
    scope=read(SCOPE);old=read(check(scope['original_current_evaluation_protocol']));seal=read(PRED)
    require(seal['completed'] and seal['PASS'] and seal['all3_predictions_completed_before_any_label_reads'] and seal['no_label_or_baseline_score_reads'],'All3 prior sealed')
    require(tuple(seal['models'])==ARMS and seal['valid']==1739 and seal['U']==45 and seal['total_candidate_head_calls']==6,'Fixed all3 execution')
    for b in old['operator_codes']+[old['code']]:check(b)
    check(seal['code']);check(seal['protocol'])
    helper=bind(HELPER)
    write(CHECKS,generated())
    inherited=('operator_codes','prior','truth','manual','historical_original','historical_three_tables','annotation_authority','frozen_annotation_ledger')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),cause_addendum=bind(CAUSE),prediction_seal=bind(PRED),
           prior_evaluation_protocol=scope['original_current_evaluation_protocol'],prior_evaluation_code=old['code'],helper_code=helper,
           original_decoder_code=seal['code'],models=list(MODELS),arms=list(ARMS),output=str(OUT),checks=bind(CHECKS),**{k:old[k] for k in inherited},
           original_metadata_whitelist=list(META),metadata_arrays=12,feature_plus_input_arrays=13,
           row_order_guard='Truth/prior each1784 unique id/native keys and complete original key sequence must match before indexed prediction association; invented swapped/duplicate rows rejected.',
           fixed_contract='Original all1784,paired1237 D1147/E90,same-slot1248,manualnear376(153WOOD+223SQUARE),1739valid/U45,WOOD105/SQUARE118,self separate,original coordinate bins',
           new_adapter='Original rawowner3 and stored binary prediction unchanged. Historical sign kernel receives in-memory +/-1 decision sentinels withconstant0.5display only, never persisted as raw visibility scores/probabilities. Direct stored pred and kernel responses checked exact.',
           cause_readout='All1784 nativeD/E/S/UNKNOWN x frozen ownerT/F/BG/U; ALL/WOOD/SQUARE and ALL/LE3/GT3/MISSING existing reference-error strata; not owner gold or cause proof.',
           fixed_crop_case_ids_inherited_not_read=FIXED19,fixed_crop_policy_inherited_not_read=FIXED19_POLICY,fixed_crop_render_receipt_inherited_not_read=FIXED19_RENDER,
           fixed_crop_gallery=str(OLD/'fixed_current_crop_review_v1/caption_aligned_v2/index.html'),new_crop_selection=False,
           current_score_label_RGB_or_annotation_payload_body_hash_reads_during_prepare=0,
           semantic_limit='Current native semantic D/E/S is different from exactq-owner; F/BG=>E diagnostic does not certify native external/self cause. No model/data/threshold/architecture selection or adoption.',
           selected_model=None,adopted=False,fully_solved=False,automatic_retry=False)
    write(PROTOCOL,p)
    template=HERE/'ROOT_SAVED_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_saved_five_model_evaluation=False,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),
                        cause_addendum=bind(CAUSE),prediction_seal=bind(PRED),selected_model=None,
                        private_fit=False,threshold_or_model_or_data_choice=False,RGB_read=False,encoder_forward=False,V112=False,PnP=False,q_L_correction=False,new_raw_annotation_traceback=False))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),
                    prediction_seal=bind(PRED),entry_template=bind(template),current_score_label_body_or_hash_reads=0))
    print(json.dumps(dict(prepared_only=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entry):
    e=read(entry);require(e['approved_saved_five_model_evaluation'] is True and e['selected_model'] is None,'Separate saved evaluation entry')
    require(all(e[k] is False for k in ('private_fit','threshold_or_model_or_data_choice','RGB_read','encoder_forward','V112','PnP','q_L_correction','new_raw_annotation_traceback')),'Evaluation-only scope')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['scope']==bind(SCOPE) and e['cause_addendum']==bind(CAUSE),'Frozen evaluator')
    p=read(PROTOCOL);require(e['prediction_seal']==p['prediction_seal']==bind(PRED),'Completed all3 prediction seal')
    require(sha(SCOPE)==SCOPE_SHA and sha(CAUSE)==CAUSE_SHA and sha(PRED)==PRED_SHA,'Frozen root bindings')
    seal=read(PRED);require(seal['completed'] and seal['PASS'] and seal['all3_predictions_completed_before_any_label_reads'] and tuple(seal['models'])==ARMS,'All3 complete')
    for b in p['operator_codes']+[p['prior_evaluation_code'],p['helper_code'],p['original_decoder_code'],p['checks']]:check(b)
    require(read(check(p['checks']))['PASS'] and not OUT.exists(),'Invented checks/single new output')
    OUT.mkdir();write(OUT/'BEFORE_LABEL_READ.json',dict(completed=True,entry=bind(entry),prediction_seal=p['prediction_seal'],label_or_score_bodies_read_so_far=False))
    try:
        core,base,extra,exact_csv_rows,decode=pure()
        truth=core.readcsv(check(p['truth']));manual=core.readcsv(check(p['manual']));prior=[json.loads(s) for s in check(p['prior']).read_text().splitlines() if s]
        require(len(truth)==len(prior)==1784 and len({r['id'] for r in truth})==223,'All1784/223')
        assert_row_order(core,truth,prior)
        packs={};raws={};owners={};preds={};rows={};reference=None
        for tag,arm in zip(NEW_MODELS,ARMS):
            a,raw,owner,pred=load_owner(seal['models'][arm]['outputs']['raw_packet'],decode,reference)
            if reference is None:reference=a
            for key in META:
                require(seal['original_input_array_fingerprints'][key]==dict(dtype=str(a[key].dtype),shape=list(a[key].shape),bytes_sha256=hashlib.sha256(a[key].tobytes()).hexdigest()),'Original12 metadata fingerprint')
            for i,r in enumerate(prior):
                require(core.key(r)==(str(a['id'][i]),str(int(a['native_corner'][i]))),'Original native order')
                require(core.numeric_same(r['query_x'],a['original_query_xy'][i,0]) and core.numeric_same(r['query_y'],a['original_query_xy'][i,1]) and core.numeric_same(r['roi_side'],a['original_side_L'][i]),'Original prior qL')
            packs[tag],raws[tag],owners[tag],preds[tag]=a,raw,owner,pred;rows[tag]=decision_rows(core,prior,pred)
        lookup={m:{core.key(r):int(preds[m][i]) for i,r in enumerate(truth)} for m in NEW_MODELS}
        original_call=base.call
        def call(r,model,task,probabilities,threshold,rules=None):
            return decision_state(lookup[model][core.key(r)]) if model in lookup else original_call(r,model,task,probabilities,threshold,rules)
        base.call=call
        oldrows=[dict(r,raw_physical_external_logit=r['raw_query_aux_hidden_logit']) for r in rows['C48']]
        tables=core.calculate_all(base,truth,prior,rows['C47'],oldrows,manual)
        tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,rows['C47'],oldrows)
        tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
        for av in [tables['availability'],*tables['availability_by_group']]:
            av['C47_total_U']=av.pop('new_total_U');av['C48_total_U']=av.pop('V91_total_U')
        model_call=lambda r,m:call(r,m,'D_vs_E',{},.5)
        tables['paired_recovery_and_damage']=paired_transitions(base,truth,model_call)
        tables['owner_vs_native_visibility_and_reference_error']=owner_native_cross(truth,owners,reference['valid'],base)
        require(sum(base.yes(r,'paired_same_query_RGB_D_vs_E') and r['human_state'] in (base.D,base.E) for r in truth)==1237,'Primary1237')
        require(sum(base.yes(r,'same_slot_pipeline_D_vs_E') and r['human_state'] in (base.D,base.E) for r in truth)==1248,'Same-slot1248')
        require({r['N'] for r in tables['manual_near_D']}=={153,223},'Near376')
        replay=core.historical_replay(tables['metrics'],core.readcsv(check(p['historical_original'])),set(BASELINES))
        quality=core.readcsv(check(p['frozen_annotation_ledger']));require(len(quality)==1784,'Frozen provenance ledger')
        ledger=[]
        for i,r in enumerate(truth):
            k=core.key(r);qr=quality[i];require(core.key(qr)==k,'Frozen provenance order')
            d=dict(global_index=i,id=k[0],native_corner=int(k[1]),group=r['group'],human_state=r['human_state'],mapping_status=r['mapping_status'],
                   query_x=float(reference['original_query_xy'][i,0]),query_y=float(reference['original_query_xy'][i,1]),roi_side=float(reference['original_side_L'][i]),
                   query_available=bool(reference['valid'][i]),human_reference_x=r['human_reference_x'],human_reference_y=r['human_reference_y'],
                   paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],
                   initial_reference_error_px=r['initial_reference_error_px'],original_query_valid=r['query_valid'],legacy_original_state=r['legacy_stored_state'])
            for m in MODELS:
                response='OTHER_UNRESOLVED' if m=='v61' and (r['query_valid']!='True' or r.get('present_at_fixed_0_5') not in ('True','False')) else model_call(r,m)
                d[m+'_original_response']=response;d[m+'_D_E_U']='D' if response=='VISIBLE' else 'E' if response=='ABSENT_NONVISIBLE' else 'U'
            for m in NEW_MODELS:
                d[m+'_owner_prediction']=int(owners[m][i]);d[m+'_owner_T_F_BG_U']='U' if owners[m][i]<0 else ('T','F','BG')[int(owners[m][i])]
                for j,cls in enumerate(('T','F','BG')):d[m+'_raw_owner_'+cls+'_logit']=float(raws[m][i,j])
            d['v61_saved_p_present']=r.get('presence_probability','');d['legacy_saved_p_DIRECT']=r.get('legacy_p_DIRECT',r.get('legacy_direct_probability',''))
            for key in ('annotation_coordinate_source','raw_annotation_traceback','new_primary_quality_gold_target','secondary_error_px','secondary_fixed_reference_target'):d[key]=qr.get(key,'UNKNOWN_UNCERTIFIED')
            d['quality_reference_scope']='PREEXISTING_FROZEN_REFERENCE_PROXY_NOT_PHYSICAL_NATIVE_GOLD';ledger.append(d)
        inventory=[]
        for group in ('ALL',*base.GROUPS):
            rr=ledger if group=='ALL' else [r for r in ledger if r['group']==group]
            for m in MODELS:
                cc=Counter(r[m+'_D_E_U'] for r in rr);responses=Counter(r[m+'_original_response'] for r in rr)
                inventory.append(dict(group=group,model=m,all_registered_slots=len(rr),D=cc['D'],E_binary_response=cc['E'],U=cc['U'],U_rate=cc['U']/len(rr),
                    legacy_abstain=responses['ABSTAIN'],out_of_task_geometry_S=responses['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=responses['OTHER_UNRESOLVED'],
                    human_state_inventory=json.dumps(dict(Counter(r['human_state'] for r in rr)),sort_keys=True)))
        tables['all1784_model_inventory']=inventory;historical={}
        for name in ('METRICS','CONFUSION','HUMAN_SELF_BINARY_RESPONSE','MANUAL_NEAR_D','COORDINATE_ERROR_BINS','D_VS_EXTERNAL_E_2X3','ALL1784_MODEL_INVENTORY'):
            hist=core.readcsv(check(p['historical_three_tables'][name]));key=next(k for k in tables if k.upper()==name)
            historical[name]=exact_csv_rows([r for r in tables[key] if r.get('model') in BASELINES],[r for r in hist if r.get('model') in BASELINES])
        oldledger=core.readcsv(check(p['historical_three_tables']['ALL1784_NATIVE_LEDGER']))
        fields=('id','native_corner','v61_original_response','v61_D_E_U','legacy_original_response','legacy_D_E_U')
        historical['ALL1784_NATIVE_LEDGER_BASELINE_RESPONSES']=exact_csv_rows([{k:r[k] for k in fields} for r in ledger],[{k:r[k] for k in fields} for r in oldledger])
        for tag in NEW_MODELS:
            cross=[r for r in tables['owner_vs_native_visibility_and_reference_error'] if r['model']==tag and r['group']=='ALL']
            require(sum(r['N'] for r in cross if r['error_bin']=='ALL')==1784 and sum(r['N'] for r in cross if r['error_bin']!='ALL')==1784,'Owner-native all1784 conservation')
        for name in ('metrics','confusion','human_self_binary_response','manual_near_D','availability_by_group','coordinate_error_bins','D_vs_external_E_2x3','paired_recovery_and_damage','all1784_model_inventory','owner_vs_native_visibility_and_reference_error'):
            core.writecsv(OUT/(name.upper()+'.csv'),tables[name])
        core.writecsv(OUT/'ALL1784_NATIVE_LEDGER.csv',ledger)
        result=dict(completed=True,PASS=True,models=list(MODELS),tables=tables,original_slots=1784,WOOD_frames=105,SQUARE_frames=118,original_input_valid=1739,original_input_U=45,
                    cohort_denominators=dict(paired_same_query_RGB=1237,same_slot_pipeline=1248),historical_v61_legacy_metric_rows_exact=replay,historical_baseline_tables_exact=historical,
                    prediction_seal=p['prediction_seal'],scope=p['scope'],cause_addendum=p['cause_addendum'],frozen_annotation_ledger=p['frozen_annotation_ledger'],annotation_authority=p['annotation_authority'],
                    fixed_crop_case_ids=p['fixed_crop_case_ids_inherited_not_read'],fixed_crop_gallery=p['fixed_crop_gallery'],new_crop_selection=False,
                    new_primary_quality_gold_known=0,primary_visibility_never_quality_error_or_owner_gate=True,
                    self='HumanS binary response only; ownerF/BG is not physical external/self cause; legacy geometry route stays separate.',
                    coordinate_reference='Existing frozen reference/manual<=3px proxy only, not physical native gold or certified query-miss truth.',
                    legacy_CAL_overlap_and_repeated_current_development_exposure=True,semantic_native_visibility_vs_literal_owner_targets_differ=True,
                    source_qualification_not_overridden=True,new_forward_fit_RGB_encoder_PnP_qL_or_threshold_model_data_choice=0,selected_model=None,adopted=False,fully_solved=False)
        write(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json',result)
        outputs=[bind(x) for x in OUT.iterdir() if x.is_file()]
        write(OUT/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),scope=p['scope'],entry=bind(entry),prediction_seal=p['prediction_seal'],
            outputs=outputs,selected_model=None,adopted=False,fully_solved=False,source_admission_changed=False,new_forward_fit_RGB_PnP=0))
        print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'EVALUATION_SEAL.json'))))
    except BaseException:
        write(OUT/'FAILED_EVALUATION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path)
    args=parser.parse_args();prepare() if args.prepare else run(args.entry)
