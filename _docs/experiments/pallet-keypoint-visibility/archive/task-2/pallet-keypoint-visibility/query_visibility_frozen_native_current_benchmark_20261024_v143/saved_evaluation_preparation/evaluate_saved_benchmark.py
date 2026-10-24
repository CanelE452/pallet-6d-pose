"""Thin saved-only V143 adapter over unchanged historical current metric math."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from collections import Counter
import argparse,ast,csv,hashlib,json,math,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_FROZEN_BENCHMARK_POLICY_BEFORE_PRIVATE_BODY_READS.json'
POLICY_SHA='33856e89ab3a563e7ee80d2d2daf846d64199122340c02e7299f66fc9f51cd1b'
OLD=P/'query_visibility_training_duration_control_20261023_v124'
ADAPTER=OLD/'current_evaluation_preparation/frozen_head_current.py'
ADAPTER_SHA='d0319bb9e9d69bf963a43e77c9b4b5c0ccf1f582d1e5d2ef5e8afb6d73122cee'
CORE=P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py'
CORE_SHA='0045abb266c4e3341aa2bf4b02f387f24ec5f04b5c19b74b200000e5e55113b5'
DECODER=CORE.parent/'infer_same_query_once.py';DECODER_SHA='53fe3d7031aa9a610638465d12d8c17c0778ba0ed289a6a03e4f3540c0d372bd'
REPORT=P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/evaluate_current_saved.py'
REPORT_SHA='83c7cf31ab29154b447e98db7614fec6604307b15c9522508b8b8719740f8478'
METRIC=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
METRIC_SHA='d079650b65aef777675af552da61335a28b73f978580413d74d143433019bef6'
PRED=V/'frozen_current_head_benchmark_v1/PREDICTION_SEAL.json'
PRED_SHA='74f403323879e42f08b0561e483c2939df91321922365c1c4960838f754a27b9'
PROTOCOL=H/'PROTOCOL.json';GENERATED=H/'GENERATED_CHECKS.json';OUT=V/'saved_current_benchmark_evaluation_v1'
MODELS=('v61','legacy','C16','C29','C32')
FIELDS=('id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision')
META=('query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside','query_xy','side_L','original_query_xy','original_side_L','id','native_corner')
def require(x,why):
    if not x:raise ValueError(why)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    require('_v112'not in b['path'].lower(),'Forbidden phase path');require(sha(b['path'])==b['sha256'],'Changed binding');return Path(b['path'])
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def functions(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==set(names),'Pure function coverage')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure', 'exec'),ns)
def pure():
    for path,digest in((ADAPTER,ADAPTER_SHA),(CORE,CORE_SHA),(DECODER,DECODER_SHA),(REPORT,REPORT_SHA),(METRIC,METRIC_SHA)):require(sha(path)==digest,'Pinned math changed')
    ns=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(require=require),Path=Path,csv=csv,require=require,
        NEW='C29',NEW_GEO='UNREPORTED_C29_GEOMETRY',OLD='C32',OLD_GEO='UNREPORTED_C32_GEOMETRY',MODELS=MODELS,MINIMAL_SCORE_FIELDS=FIELDS)
    functions(CORE,{'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'},ns)
    functions(DECODER,{'decode_rows'},ns);functions(ADAPTER,{'enrich_minimal_baseline'},ns)
    core=SimpleNamespace(**ns)
    base=ModuleType('v143_original_metric_math');base.__dict__.update(math=math,Counter=Counter,D='DIRECT_VISIBLE',E='EXTERNAL_OCCLUDED',S='SELF_OCCLUDED',
        TASKS={'D_vs_E':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED'],'D_vs_S':['DIRECT_VISIBLE','SELF_OCCLUDED'],'D_vs_ES':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED']},GROUPS=['WOOD_PRIMARY105','SQUARE118'])
    functions(METRIC,{'yes','ratio','call','calculate'},base.__dict__)
    ex=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(NEW='C29',NEW_GEO='UNREPORTED_C29_GEOMETRY',OLD='C32',OLD_GEO='UNREPORTED_C32_GEOMETRY',MODELS=MODELS))
    functions(REPORT,{'error_bins','matrices'},ex)
    return core,base,SimpleNamespace(**ex)
def byte_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def generated():
    core,base,_=pure()
    rows=[dict(id='INVENTED',human_state=base.D,present_at_fixed_0_5='True',legacy_stored_state='U'),dict(id='INVENTED',human_state=base.E,present_at_fixed_0_5='False',legacy_stored_state='S')]
    result=base.calculate(rows,'legacy','D_vs_E',{},.5)
    require(result['N']==2 and result['U_visible']==1 and result['out_of_task_S_occluded']==1 and result['TN']==0,'LegacyU/S denominator')
    require(core.response(dict(valid=False,z=-100,geo=True))=='OTHER_UNRESOLVED','U precedence')
    require(core.response(dict(valid=True,z=0.,geo=False))=='ABSENT_NONVISIBLE','Fixed zero rule')
    require(not byte_equal(np.array([0.],np.float32),np.array([-0.],np.float32)),'Byte comparison')
    write(GENERATED,dict(completed=True,PASS=True,invented_only=True,code=bind(__file__),legacyU_and_S_not_externalE=True,raw0_and_U_precedence=True,current_body_reads_or_hashes=0))
def prepare():
    require(sha(POLICY)==POLICY_SHA and sha(PRED)==PRED_SHA,'Frozen phase/seal')
    generated();entry=read(OLD/'ROOT_ACTUAL_C16_SAVED_CURRENT_EVALUATION_ENTRY.json')
    c16seal=read(check(entry['inference_seal']))
    oldeval=read(check(dict(path=str(OLD/'private_evaluation/C16_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json'),sha256='ddda975c152f2b773c16ed323aaebe74b5aa850dd0a27e769f1e60df99d27199')))
    outputs={Path(b['path']).name:b for b in c16seal['outputs']}
    historical=next(b for b in oldeval['files']if Path(b['path']).name=='METRICS.csv')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),prediction_seal=bind(PRED),
        operator_codes=[bind(x)for x in(ADAPTER,CORE,DECODER,REPORT,METRIC)],C16_inference_seal=entry['inference_seal'],
        C16_scores=outputs['HEAD_ONLY_SCORES.csv'],C16_raw=outputs['RAW_HEAD_OUTPUTS.npz'],historical_C16_metrics=historical,
        prior=entry['bindings']['prior_input_rows'],truth=entry['bindings']['evaluation_manifest'],manual=entry['bindings']['manual_provenance'],
        historical_original=entry['bindings']['original_baseline_metrics'],models=list(MODELS),output=str(OUT),
        contract='Unchanged historical core math; original1237paired/1248same-slot/1784inventory. ALL45U preserved; S/unknown outside binaryE. GTerror bands/manual376 diagnostic only. AllC29/C32 fixed reported; sourcefail unchanged; selectionnull.',
        payload_body_or_hash_reads_during_prepare=0)
    write(PROTOCOL,p);template=dict(approved_saved_evaluation=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),prediction_seal=p['prediction_seal'],
        private_fit=False,threshold_or_model_choice=False,RGB_read=False,encoder_forward=False,V112=False,PnP=False,q_L_correction=False)
    write(H/'ROOT_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_ENTRY_TEMPLATE.json'))))
def load_packet(path):
    with np.load(path,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def records(core,prior,score_binding,packet_binding,reference):
    minimal=core.readcsv(check(score_binding));a=load_packet(check(packet_binding))
    require(len(minimal)==1784 and all(tuple(r)==FIELDS for r in minimal),'Original9field score schema')
    require(all(byte_equal(a[k],reference[k])for k in META),'Original ID/qL/mask byte join')
    require(a['query_logit'].shape==(1784,)and a['query_logit'].dtype==np.float32 and np.isfinite(a['raw_head_logit_before_U']).all(),'F32 scalar schema')
    require(np.array_equal(a['query_logit'],np.where(a['valid'],a['raw_head_logit_before_U'],0).astype(np.float32)),'Raw/safe mismatch')
    for i,r in enumerate(minimal):
        require(core.key(r)==(str(a['id'][i]),str(int(a['native_corner'][i]))),'Saved order')
        require(core.boolean(r['valid'])==bool(a['valid'][i]),'CSV/packet validity mismatch')
        require(float(r['raw_query_aux_hidden_logit'])==float(a['query_logit'][i])and float(r['raw_p_visible_FP32'])==float(a['p_visible_FP32'][i]),'CSV/packet mismatch')
    if 'prediction_D0_E1_Uminus1'in a:require(np.array_equal(a['prediction_D0_E1_Uminus1'],np.where(a['valid'],(a['query_logit']>=0).astype(np.int8),-1)),'Fixed prediction decode')
    return core.enrich_minimal_baseline(core,prior,minimal),a
def transitions(core,base,truth,model_call):
    out=[]
    for cohort in('paired_same_query_RGB','same_slot_pipeline'):
        for group in base.GROUPS:
            rr=[r for r in truth if r['group']==group and base.yes(r,cohort+'_D_vs_E')and r['human_state']in(base.D,base.E)]
            for new in('C29','C32'):
                for old in('C16','v61','legacy'):
                    c=Counter()
                    for r in rr:
                        a,b=model_call(r,old),model_call(r,new);gt='VISIBLE'if r['human_state']==base.D else'ABSENT_NONVISIBLE'
                        if b not in('VISIBLE','ABSENT_NONVISIBLE'):c['new_unresolved']+=1
                        elif a not in('VISIBLE','ABSENT_NONVISIBLE'):c['baseline_unresolved_new_correct'if b==gt else'baseline_unresolved_new_wrong']+=1
                        else:c['both_correct'if a==gt and b==gt else'baseline_correct_new_wrong'if a==gt else'baseline_wrong_new_correct'if b==gt else'both_wrong']+=1
                    keys=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
                    require(sum(c.values())==len(rr),'Transition denominator')
                    out.append(dict(cohort=cohort,group=group,task='D_vs_E',baseline=old,new_model=new,N=len(rr),**{k:c[k]for k in keys}))
    return out
def run(path):
    e=read(path);require(e['approved_saved_evaluation']is True,'Root entry')
    require(all(e[k]is False for k in('private_fit','threshold_or_model_choice','RGB_read','encoder_forward','V112','PnP','q_L_correction')),'Scope')
    require(sha(POLICY)==POLICY_SHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL),'Frozen code')
    p=read(PROTOCOL);require(e['prediction_seal']==p['prediction_seal']and Path(e['prediction_seal']['path'])==PRED,'Prediction binding')
    seal=read(check(e['prediction_seal']));require(seal['completed']and seal['PASS']and seal['GT_read']is False and seal['selected_model']is None and seal['source_admission_changed']is False,'Complete prediction seal')
    parity=read(check(seal['C16_parity']));require(parity['PASS'],'C16 exact parity before candidates')
    require(set(seal['models'])=={'C29','C32'},'Both fixed candidates')
    for name,m in seal['models'].items():require(m['fixed_update']==8000 and m['threshold_logit']==0 and m['source_eligible']is False and m['benchmark_only_not_admitted']is True and m['state_before']==m['state_after'],'Frozen candidate')
    for b in p['operator_codes']:check(b)
    require(read(check(p['generated']))['PASS']and not OUT.exists(),'Generated/exclusive output')
    OUT.mkdir();write(OUT/'BEFORE_LABEL_READ.json',dict(completed=True,entry=bind(path),prediction_seal=e['prediction_seal'],C16_parity=seal['C16_parity'],label_read_so_far=False))
    core,base,extra=pure();truth=core.readcsv(check(p['truth']));manual=core.readcsv(check(p['manual']));prior=[json.loads(s)for s in check(p['prior']).read_text().splitlines()if s]
    require(len(truth)==len(prior)==1784 and len({r['id']for r in truth})==223,'Full native inventory')
    reference=load_packet(check(p['C16_raw']));require(int((~reference['valid']).sum())==45 and int(reference['valid'].sum())==1739,'Original U45')
    rr={};pack={}
    rr['C16'],pack['C16']=records(core,prior,p['C16_scores'],p['C16_raw'],reference)
    for name in('C29','C32'):rr[name],pack[name]=records(core,prior,seal['models'][name]['scores_csv'],seal['models'][name]['raw_packet'],reference)
    # The unchanged core names its second input field physical_external. This
    # lossless in-memory alias is only a schema bridge; source bytes unchanged.
    oldrows=[dict(r,raw_physical_external_logit=r['raw_query_aux_hidden_logit'])for r in rr['C32']]
    indices={k:core.score_index(v,'raw_query_aux_hidden_logit')for k,v in rr.items()}
    orig=base.call
    def call(r,model,task,probabilities,threshold,rules=None):
        if model in indices:return core.response(indices[model][core.key(r)],False)
        return orig(r,model,task,probabilities,threshold,rules)
    base.call=call
    tables=core.calculate_all(base,truth,prior,rr['C29'],oldrows,manual)
    tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,rr['C29'],oldrows)
    tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
    def model_call(r,model):return call(r,model,'D_vs_E',{},.5)
    tables['paired_recovery_and_damage']=transitions(core,base,truth,model_call)
    replay=core.historical_replay(tables['metrics'],core.readcsv(check(p['historical_original'])),{'v61','legacy'})
    cmapped=[dict(r,model='V124_C16_RAW')for r in tables['metrics']if r['model']=='C16']
    replay16=core.historical_replay(cmapped,core.readcsv(check(p['historical_C16_metrics'])),{'V124_C16_RAW'})
    cohortN={}
    for cohort,expected in(('paired_same_query_RGB',1237),('same_slot_pipeline',1248)):
        n=sum(1 for r in truth if base.yes(r,cohort+'_D_vs_E')and r['human_state']in(base.D,base.E));require(n==expected,'Original cohort denominator');cohortN[cohort]=n
    require({r['N']for r in tables['manual_near_D']}=={153,223},'Original manual376')
    # Full inventory is a separate denominator. v61 invalid/missing must stay U;
    # legacy S is retained separately and is not changed into external E.
    ledger=[];inventory=[]
    for r in truth:
        k=core.key(r);d=dict(id=k[0],native_corner=int(k[1]),group=r['group'],human_state=r['human_state'],mapping_status=r['mapping_status'],
             paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],
             initial_reference_error_px=r['initial_reference_error_px'],original_query_valid=r['query_valid'],legacy_original_state=r['legacy_stored_state'])
        for model in MODELS:
            if model=='v61'and(r['query_valid']!='True'or r.get('present_at_fixed_0_5')not in('True','False')):response='OTHER_UNRESOLVED'
            else:response=model_call(r,model)
            d[model+'_original_response']=response;d[model+'_D_E_U']='D'if response=='VISIBLE'else'E'if response=='ABSENT_NONVISIBLE'else'U'
        ledger.append(d)
    for group in('ALL',*base.GROUPS):
        rows=ledger if group=='ALL'else[r for r in ledger if r['group']==group]
        for model in MODELS:
            c=Counter(r[model+'_D_E_U']for r in rows);responses=Counter(r[model+'_original_response']for r in rows)
            inventory.append(dict(group=group,model=model,all_registered_slots=len(rows),D=c['D'],E_binary_response=c['E'],U=c['U'],U_rate=c['U']/len(rows),
                legacy_abstain=responses['ABSTAIN'],out_of_task_geometry_S=responses['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=responses['OTHER_UNRESOLVED'],
                human_state_inventory=json.dumps(dict(Counter(r['human_state']for r in rows)),sort_keys=True)))
    tables['all1784_model_inventory']=inventory
    for name in('metrics','confusion','human_self_binary_response','manual_near_D','availability_by_group','coordinate_error_bins','D_vs_external_E_2x3','paired_recovery_and_damage','all1784_model_inventory'):
        core.writecsv(OUT/(name.upper()+'.csv'),tables[name])
    core.writecsv(OUT/'ALL1784_NATIVE_LEDGER.csv',ledger)
    summary=dict(completed=True,PASS=True,models=list(MODELS),tables=tables,original_slots=1784,WOOD_frames=105,SQUARE_frames=118,cohort_denominators=cohortN,
        original_input_valid=1739,original_input_U=45,additional_U=0,historical_v61_legacy_rows_exact=replay,historical_C16_rows_exact=replay16,
        raw_C32_internal_field_alias_only=True,source_model_admission_unchanged=True,source_eligible=dict(C29=False,C32=False),selected_model=None,adopted_model=None,
        development_exposed_not_blind=True,coordinate_error_diagnostic_only=True,S_and_unknown_outside_external_binary=True,
        no_new_forward_fit_threshold_choice_RGB_encoder_PnP=True,policy=p['policy'],prediction_seal=e['prediction_seal'],
        limits_KO=['현재223은 개발 노출 자료이며 독립 blind test가 아닙니다.','사람 S는 별도 응답 진단이며 binary hidden이 학습된 self/외부 원인 인증을 뜻하지 않습니다.','원래 reference 오차와 manualD376은 진단이며 primary 필터·좌표 수정·모델 선택에 사용하지 않았습니다.','C29/C32 source qualification 실패와 current_allowedfalse는 불변이며 두 모델을 모두 보고하고 채택하지 않았습니다.'])
    write(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json',summary)
    files=[bind(f)for f in OUT.iterdir()if f.is_file()]
    write(OUT/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),policy=p['policy'],protocol=bind(PROTOCOL),entry=bind(path),prediction_seal=e['prediction_seal'],outputs=files,selected_model=None,source_admission_changed=False,new_forward_fit_RGB_PnP=0))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'EVALUATION_SEAL.json'),results=bind(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use metadata/invented-only --prepare or explicit root saved evaluation --entry.')
