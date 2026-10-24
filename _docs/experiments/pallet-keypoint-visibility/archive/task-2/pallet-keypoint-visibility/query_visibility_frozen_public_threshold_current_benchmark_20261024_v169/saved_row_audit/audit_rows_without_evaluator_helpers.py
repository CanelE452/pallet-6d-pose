"""Direct saved-row arithmetic; never import/call the original evaluator helpers."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,traceback
import numpy as np
V=Path(__file__).resolve().parent.parent;O=V/'saved_seven_condition_evaluation_v1';H=Path(__file__).resolve().parent
BASE=('v61','legacy','C16');NEW=('C45_ORIGINAL0','C45_PUBLICCAL_FULL','C46_ORIGINAL0','C46_PUBLICCAL_FULL');MODELS=BASE+NEW
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def key(r):return str(r['id']),int(r['native_corner'])
def yes(v):return str(v).lower()=='true'
def ratio(a,b):return a/b if b else None
def numeric(v):
    try:return float(v)
    except (ValueError,TypeError):return float('nan')
def compare(actual,expected,fields):
    for f in fields:
        a=actual[f];b=expected[f]
        if a is None:assert b in ('',None), (f,a,b)
        elif isinstance(a,(int,float)):assert float(b)==a,(f,a,b)
        else:assert str(a)==str(b),(f,a,b)
def main():
    assert not (H/'COMPLETE.json').exists()
    p=read(V/'evaluation_preparation/PROTOCOL_BEFORE_SAVED_CURRENT_EVALUATION_BODIES.json')
    eval_binding=bind(O/'EVALUATION_SEAL.json');assert eval_binding['sha256']=='6c6945d34b9eb372c72058cab30201f0b755dc310c0b40bdec86a50f04e1e62c'
    seal=read(O/'EVALUATION_SEAL.json');assert seal['completed'] and seal['PASS']
    for b in seal['outputs']:assert sha(b['path'])==b['sha256']
    led=rows(O/'ALL1784_NATIVE_LEDGER.csv');truth=rows(p['truth']['path']);manual=rows(p['manual']['path'])
    assert len(led)==len(truth)==1784 and len({r['id'] for r in led})==223
    assert [key(r) for r in led]==[key(r) for r in truth] and len({key(r) for r in led})==1784
    assert all(r['global_index']==str(i) and r['human_state']==truth[i]['human_state'] and r['group']==truth[i]['group'] for i,r in enumerate(led))
    tm={key(r):r for r in truth};lm={key(r):r for r in led};mm={key(r):r for r in manual}
    # This calculation uses stored baseline decisions/metadata. Legacy S has the
    # original task-specific treatment, independent of external learned scores.
    def response(r,model,task='D_vs_E'):
        t=tm[key(r)]
        if model=='v61':return 'VISIBLE' if yes(t['present_at_fixed_0_5']) else 'ABSENT_NONVISIBLE'
        if model=='legacy':
            s=r['legacy_original_state']
            return 'VISIBLE' if s in ('B','I') else 'ABSENT_NONVISIBLE' if s=='O' or (s=='S' and task!='D_vs_E') else 'OUT_OF_TASK_GEOMETRY_S' if s=='S' else 'ABSTAIN' if s=='U' else 'OTHER_UNRESOLVED'
        return r[model+'_original_response']
    def selection(cohort,group,task):
        states={'D_vs_E':(D,E),'D_vs_S':(D,S),'D_vs_ES':(D,E,S)}[task]
        flag=cohort+'_'+('D_vs_ES' if task=='D_vs_S' else task)
        return [r for r in led if (group=='ALL' or r['group']==group) and yes(tm[key(r)][flag]) and r['human_state'] in states]
    def counts(rr,model,task):
        c=Counter((r['human_state']==D,response(r,model,task)) for r in rr)
        pos=sum(r['human_state']==D for r in rr);neg=len(rr)-pos
        tp=c[True,'VISIBLE'];fn=c[True,'ABSENT_NONVISIBLE'];fp=c[False,'VISIBLE'];tn=c[False,'ABSENT_NONVISIBLE']
        up=c[True,'ABSTAIN'];un=c[False,'ABSTAIN'];sp=c[True,'OUT_OF_TASK_GEOMETRY_S'];sn=c[False,'OUT_OF_TASK_GEOMETRY_S'];op=c[True,'OTHER_UNRESOLVED'];on=c[False,'OTHER_UNRESOLVED']
        assert tp+fn+up+sp+op==pos and fp+tn+un+sn+on==neg
        return dict(N=len(rr),frames=len({r['id'] for r in rr}),visible_support=pos,occluded_support=neg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=op,other_unresolved_occluded=on,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),abstention_rate=ratio(up+un,len(rr)),out_of_task_S_rate=ratio(sp+sn,len(rr)),unresolved_rate=ratio(up+un+sp+sn+op+on,len(rr)),decision_coverage=ratio(tp+fn+fp+tn,len(rr)),correct_fraction_all_GT=ratio(tp+tn,len(rr)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))
    checked=Counter()
    metrics=rows(O/'METRICS.csv')
    for m in metrics:
        q=counts(selection(m['cohort'],m['group'],m['task']),m['model'],m['task']);compare(q,m,q);checked['metric_rows']+=1
    for m in rows(O/'CONFUSION.csv'):
        rr=[r for r in selection(m['cohort'],m['group'],m['task']) if r['human_state']==m['human_state']];c=Counter(response(r,m['model'],m['task']) for r in rr)
        q=dict(N=len(rr),pred_visible=c['VISIBLE'],pred_binary_hidden=c['ABSENT_NONVISIBLE'],legacy_U=c['ABSTAIN'],out_of_task_geometry_S=c['OUT_OF_TASK_GEOMETRY_S'],input_or_operator_U=c['OTHER_UNRESOLVED']);compare(q,m,q);checked['confusion_rows']+=1
    for m in rows(O/'HUMAN_SELF_BINARY_RESPONSE.csv'):
        rr=[r for r in selection(m['cohort'],m['group'],'D_vs_S') if r['human_state']==S];c=Counter(response(r,m['model'],'D_vs_S') for r in rr)
        q=dict(human_S_support=len(rr),binary_hidden_response=c['ABSENT_NONVISIBLE'],visible_response=c['VISIBLE'],unresolved=len(rr)-c['ABSENT_NONVISIBLE']-c['VISIBLE'],human_S_binary_hidden_response_rate=ratio(c['ABSENT_NONVISIBLE'],len(rr)));compare(q,m,q);checked['human_self_rows']+=1
    near=[lm[key(r)] for r in manual if r['annotation_coordinate_source']=='manual_click' and r['human_state']==D and numeric(r['initial_reference_error_px'])<=3 and yes(r['query_valid'])]
    assert len(near)==376 and len({key(r) for r in near})==376
    for m in rows(O/'MANUAL_NEAR_D.csv'):
        rr=[r for r in near if r['group']==m['group']];c=Counter(response(r,m['model'],'D_vs_ES') for r in rr)
        q=dict(N=len(rr),pred_visible=c['VISIBLE'],pred_binary_hidden=c['ABSENT_NONVISIBLE'],unresolved=len(rr)-c['VISIBLE']-c['ABSENT_NONVISIBLE'],semantic_D_recall=ratio(c['VISIBLE'],len(rr)));compare(q,m,q);checked['manual_near_rows']+=1
    def error_bin(r):
        x=numeric(r['initial_reference_error_px']);return 'MISSING_REFERENCE_ERROR' if not math.isfinite(x) else 'LE3' if x<=3 else 'GT3'
    for m in rows(O/'COORDINATE_ERROR_BINS.csv'):
        rr=[r for r in selection(m['cohort'],m['group'],'D_vs_E') if error_bin(r)==m['error_bin']];q=counts(rr,m['model'],'D_vs_E')
        q['manual_click_reference_N']=sum(mm.get(key(r),{}).get('annotation_coordinate_source')=='manual_click' for r in rr);q['other_or_unspecified_reference_N']=len(rr)-q['manual_click_reference_N'];compare(q,m,q);checked['coordinate_bin_rows']+=1
    transition_fields=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
    for m in rows(O/'PAIRED_RECOVERY_AND_DAMAGE.csv'):
        rr=[r for r in selection(m['cohort'],m['group'],'D_vs_E') if m['truth_class']=='ALL_DE' or r['human_state']==m['truth_class']];c=Counter()
        for r in rr:
            old=response(r,m['baseline']);new=response(r,m['new_model']);gt='VISIBLE' if r['human_state']==D else 'ABSENT_NONVISIBLE'
            if new not in ('VISIBLE','ABSENT_NONVISIBLE'):name='new_unresolved'
            elif old not in ('VISIBLE','ABSENT_NONVISIBLE'):name='baseline_unresolved_new_correct' if new==gt else 'baseline_unresolved_new_wrong'
            elif old==gt:name='both_correct' if new==gt else 'baseline_correct_new_wrong'
            else:name='baseline_wrong_new_correct' if new==gt else 'both_wrong'
            c[name]+=1
        q={k:c[k] for k in transition_fields};q['N']=len(rr);assert sum(c.values())==len(rr);compare(q,m,q);checked['paired_transition_rows']+=1
    for m in rows(O/'D_VS_EXTERNAL_E_2X3.csv'):
        q=counts(selection(m['cohort'],m['group'],'D_vs_E'),m['model'],'D_vs_E')
        mapped=dict(human_D_N=q['visible_support'],human_external_E_N=q['occluded_support'],D_pred_D=q['TP'],D_pred_hidden=q['FN_decided'],D_pred_U=q['U_visible']+q['out_of_task_S_visible']+q['other_unresolved_visible'],E_pred_D=q['FP'],E_pred_hidden=q['TN'],E_pred_U=q['U_occluded']+q['out_of_task_S_occluded']+q['other_unresolved_occluded'],D_precision=q['visible_precision'],D_recall=q['visible_recall_all_GT'],external_E_response_precision=q['occluded_precision'],external_E_response_recall=q['occluded_recall_all_GT'],unresolved_rate=q['unresolved_rate']);compare(mapped,m,mapped);checked['2x3_rows']+=1
    for m in rows(O/'ALL1784_MODEL_INVENTORY.csv'):
        rr=[r for r in led if m['group']=='ALL' or r['group']==m['group']];c=Counter(r[m['model']+'_D_E_U'] for r in rr);q=dict(all_registered_slots=len(rr),D=c['D'],E_binary_response=c['E'],U=c['U'],U_rate=ratio(c['U'],len(rr)));compare(q,m,q);checked['inventory_rows']+=1
    # Compare stored old baseline table rows exactly as CSV strings, using no
    # original replay helpers or recomputation to choose any model/threshold.
    baseline_replay={}
    for name,binding in p['historical_three_tables'].items():
        hist=rows(binding['path'])
        if name=='ALL1784_NATIVE_LEDGER':
            fields=[k for k in hist[0] if not any(k.startswith(tag+'_') for tag in ('C29','C32','C33','C34'))]
            aa=[{k:r[k] for k in fields} for r in led];bb=[{k:r[k] for k in fields} for r in hist]
        else:
            aa=[r for r in rows(O/(name+'.csv')) if r.get('model') in BASE];bb=[r for r in hist if r.get('model') in BASE]
        assert aa==bb,('historical baseline rows',name);baseline_replay[name]=len(bb)
    pred=read(V/'frozen_two_head_four_condition_predictions_v1/PREDICTION_SEAL.json');packet_checks={}
    for name in NEW:
        b=pred['conditions'][name]['raw_packet'];assert sha(b['path'])==b['sha256']
        with np.load(b['path'],allow_pickle=False) as f:a={k:f[k].copy() for k in ('raw_head_logit_before_U','threshold_logit_float64','decision_score64','prediction_D0_E1_Uminus1','valid','id','native_corner','original_query_xy','original_side_L')}
        t=float(a['threshold_logit_float64']);q=np.where(a['valid'],(a['raw_head_logit_before_U'].astype(np.float64)>=np.float64(t)).astype(np.int8),-1)
        assert np.array_equal(q,a['prediction_D0_E1_Uminus1']) and q.shape==(1784,) and int((q<0).sum())==45
        assert np.array_equal(a['decision_score64'],a['raw_head_logit_before_U'].astype(np.float64)-np.float64(t))
        for i,r in enumerate(led):
            assert key(r)==(str(a['id'][i]),int(a['native_corner'][i]))
            assert r[name+'_D_E_U']==('U' if q[i]<0 else 'D' if q[i]==0 else 'E')
            assert float(r[name+'_raw_visibility_logit'])==float(a['raw_head_logit_before_U'][i]) and float(r[name+'_threshold_float64'])==t and float(r[name+'_decision_score64'])==a['decision_score64'][i]
            assert yes(r['query_available'])==bool(a['valid'][i])
            assert float(r['query_x'])==a['original_query_xy'][i,0] and float(r['query_y'])==a['original_query_xy'][i,1] and float(r['roi_side'])==a['original_side_L'][i]
        packet_checks[name]=dict(rows=1784,valid=1739,U=45,threshold_float64=t,rawFP32_ge_float64_and_ledger_exact=True,packet=b)
    summaries=[]
    for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
        for group in ('ALL','WOOD_PRIMARY105','SQUARE118'):
            rr=selection(cohort,group,'D_vs_E')
            for model in MODELS:
                q=counts(rr,model,'D_vs_E');ud=q['U_visible']+q['out_of_task_S_visible']+q['other_unresolved_visible'];ue=q['U_occluded']+q['out_of_task_S_occluded']+q['other_unresolved_occluded']
                summaries.append(dict(cohort=cohort,group=group,model=model,CM_rows_D_E_columns_D_E_U=[[q['TP'],q['FN_decided'],ud],[q['FP'],q['TN'],ue]],D_N=q['visible_support'],E_N=q['occluded_support'],D_precision=q['visible_precision'],D_recall=q['visible_recall_all_GT'],E_precision=q['occluded_precision'],E_recall=q['occluded_recall_all_GT'],U=ud+ue,N=q['N']))
    main=[r for r in summaries if r['cohort']=='paired_same_query_RGB' and r['group']=='ALL']
    assert {(r['D_N'],r['E_N'],r['N']) for r in main}=={(1147,90,1237)}
    assert [(r['CM_rows_D_E_columns_D_E_U'][0][0],r['CM_rows_D_E_columns_D_E_U'][1][1]) for r in main if r['model'] in NEW]==[(995,55),(359,88),(843,80),(287,89)]
    assert sum(not yes(r['query_available']) for r in led)==45
    out=dict(completed=True,PASS=True,audit_method='Direct CSV row filters, integer counters and arithmetic plus direct saved FP32>=float64 packet comparisons; no original evaluator helpers imported/called',counts_checked=dict(checked),rows=1784,frames=223,primary_D1147_E90_N1237=True,all1784_inputU45_preserved=True,summary_tables=summaries,packet_checks=packet_checks,baseline_historical_all_tables_exact=baseline_replay,near376_preserved=True,coordinate_reference_proxy_not_physical_gold=True,self_not_learned_cause_identification=True,new_fit_or_forward_or_threshold_selection=0,new_RGB_or_feature_or_checkpoint_reads=0,bindings=dict(code=bind(__file__),evaluation_seal=eval_binding,ledger=bind(O/'ALL1784_NATIVE_LEDGER.csv'),truth=p['truth'],manual=p['manual']),selected_model=None,adopted=False,fully_solved=False)
    write(H/'COMPLETE.json',out);print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'COMPLETE.json'),counts_checked=dict(checked))))
if __name__=='__main__':
    try:main()
    except BaseException:
        write(H/'FAILED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
