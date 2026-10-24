"""Evaluate frozen saved classifications, without importing any inference code."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREP = HERE.parent/'evaluation_preparation_20261022'
MANIFEST = PREP/'PREPARED_QUERY_MANIFEST.csv'
D, E, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
GROUPS = ['WOOD_PRIMARY105', 'SQUARE118']
TASKS = {'D_vs_E':[D,E], 'D_vs_S':[D,S], 'D_vs_ES':[D,E,S]}
MODELS = ['v61', 'legacy']


def readcsv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def writecsv(name, rows):
    if not rows: return
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ratio(a,b): return a/b if b else None
def yes(r,k): return r[k]=='True'


rows=readcsv(MANIFEST)
assert len(rows)==1784
for r in rows:
    assert yes(r,'present_at_fixed_0_5') == (float(r['presence_probability'])>=.5)
    if yes(r,'legacy_RGB_evaluated'):
        p=float(r['legacy_p_DIRECT'])
        state='VISIBLE' if p>=.98 else 'NEGATIVE' if p<.96 else 'ABSTAIN'
        assert state==('VISIBLE' if r['legacy_stored_state'] in ('B','I') else 'NEGATIVE' if r['legacy_stored_state']=='O' else 'ABSTAIN')


def prediction(r,model,task):
    if model=='v61': return 'VISIBLE' if yes(r,'present_at_fixed_0_5') else 'NEGATIVE'
    state=r['legacy_stored_state']
    if state in ('B','I'): return 'VISIBLE'
    if state=='O': return 'NEGATIVE'
    if state=='S': return 'OUT_OF_TASK_S' if task=='D_vs_E' else 'NEGATIVE'
    if state=='U': return 'ABSTAIN'
    return 'OTHER_UNRESOLVED'


def cohort(kind,task):
    # D/S uses the already-fixed D/(E+S) cohort, then removes E labels.
    return kind+'_'+('D_vs_ES' if task=='D_vs_S' else task)


def calculate(rs,model,task):
    cm=Counter((r['human_state']==D,prediction(r,model,task)) for r in rs)
    npos=sum(r['human_state']==D for r in rs);nneg=len(rs)-npos
    tp,fn,fp,tn=(cm[(True,'VISIBLE')],cm[(True,'NEGATIVE')],cm[(False,'VISIBLE')],cm[(False,'NEGATIVE')])
    u_pos,u_neg=cm[(True,'ABSTAIN')],cm[(False,'ABSTAIN')]
    s_pos,s_neg=cm[(True,'OUT_OF_TASK_S')],cm[(False,'OUT_OF_TASK_S')]
    o_pos,o_neg=cm[(True,'OTHER_UNRESOLVED')],cm[(False,'OTHER_UNRESOLVED')]
    assert tp+fn+u_pos+s_pos+o_pos==npos and fp+tn+u_neg+s_neg+o_neg==nneg
    return dict(N=len(rs),frames=len({r['id'] for r in rs}),visible_support=npos,occluded_support=nneg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=u_pos,U_occluded=u_neg,out_of_task_S_visible=s_pos,out_of_task_S_occluded=s_neg,other_unresolved_visible=o_pos,other_unresolved_occluded=o_neg,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,npos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,nneg),abstention_rate=ratio(u_pos+u_neg,len(rs)),out_of_task_S_rate=ratio(s_pos+s_neg,len(rs)),unresolved_rate=ratio(u_pos+u_neg+s_pos+s_neg+o_pos+o_neg,len(rs)),decision_coverage=ratio(tp+fn+fp+tn,len(rs)),correct_fraction_all_GT=ratio(tp+tn,len(rs)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))


metrics=[];confusion=[];sessions=[];pairs=[];error_metrics=[];error_distribution=[];calls=[]
for kind in ('paired_same_query_RGB','same_slot_pipeline'):
    for group in GROUPS:
        for task,states in TASKS.items():
            rs=[r for r in rows if r['group']==group and yes(r,cohort(kind,task)) and r['human_state'] in states]
            if not any(r['human_state']!=D for r in rs): continue
            for model in MODELS:
                metrics.append(dict(cohort=kind,group=group,task=task,model=model,**calculate(rs,model,task)))
                for state in states:
                    z=[r for r in rs if r['human_state']==state]
                    counts=Counter(prediction(r,model,task) for r in z)
                    confusion.append(dict(cohort=kind,group=group,task=task,model=model,human_state=state,N=len(z),pred_visible=counts['VISIBLE'],pred_negative=counts['NEGATIVE'],pred_U=counts['ABSTAIN'],pred_out_of_task_S=counts['OUT_OF_TASK_S'],pred_other_unresolved=counts['OTHER_UNRESOLVED']))
                for rec in sorted({r['recording'] for r in rs}):
                    z=[r for r in rs if r['recording']==rec]
                    sessions.append(dict(cohort=kind,group=group,task=task,model=model,recording=rec,**calculate(z,model,task)))
            change=Counter()
            for r in rs:
                a,b=prediction(r,'legacy',task),prediction(r,'v61',task)
                pos=r['human_state']==D
                vb=(b=='VISIBLE')==pos
                if a in ('VISIBLE','NEGATIVE'):
                    va=(a=='VISIBLE')==pos
                    change[('both_correct' if va and vb else 'legacy_correct_v61_wrong' if va else 'legacy_wrong_v61_correct' if vb else 'both_wrong')]+=1
                else: change['legacy_unresolved_v61_correct' if vb else 'legacy_unresolved_v61_wrong']+=1
                calls.append(dict(cohort=kind,group=group,task=task,id=r['id'],native_corner=r['native_corner'],human_state=r['human_state'],v61_call=b,legacy_call=a,legacy_stored_state=r['legacy_stored_state'],initial_reference_error_px=r['initial_reference_error_px'],error_over_roi_side=float(r['initial_reference_error_px'])/float(r['v61_roi_side_raw_px']),human_reference_inside_nominal_crop=r['human_reference_inside_nominal_crop']))
            keys=['both_correct','legacy_correct_v61_wrong','legacy_wrong_v61_correct','both_wrong','legacy_unresolved_v61_correct','legacy_unresolved_v61_wrong']
            pairs.append(dict(cohort=kind,group=group,task=task,N=len(rs),**{k:change[k] for k in keys}))
        base=[r for r in rows if r['group']==group and yes(r,kind+'_D_vs_ES')]
        for human_state in (D,E,S):
            z=[r for r in base if r['human_state']==human_state]
            if not z: continue
            vals=sorted(float(r['initial_reference_error_px']) for r in z)
            error_distribution.append(dict(cohort=kind,group=group,human_state=human_state,N=len(z),mean_error_px=statistics.mean(vals),median_error_px=statistics.median(vals),p95_nearest_rank_px=vals[max(0,__import__('math').ceil(.95*len(vals))-1)],reference_outside_nominal_crop=sum(not yes(r,'human_reference_inside_nominal_crop') for r in z)))
            bins=[('px_le3',lambda r:float(r['initial_reference_error_px'])<=3),('px_3_to10',lambda r:3<float(r['initial_reference_error_px'])<=10),('px_10_to30',lambda r:10<float(r['initial_reference_error_px'])<=30),('px_gt30',lambda r:float(r['initial_reference_error_px'])>30),('reference_inside_crop',lambda r:yes(r,'human_reference_inside_nominal_crop')),('reference_outside_crop',lambda r:not yes(r,'human_reference_inside_nominal_crop'))]
            for label,accept in bins:
                zz=[r for r in z if accept(r)]
                for model in MODELS:
                    cnt=Counter(prediction(r,model,'D_vs_ES') for r in zz)
                    error_metrics.append(dict(cohort=kind,group=group,human_state=human_state,error_bin=label,model=model,N=len(zz),pred_visible=cnt['VISIBLE'],pred_negative=cnt['NEGATIVE'],pred_U=cnt['ABSTAIN'],class_recall_all_GT=ratio(cnt['VISIBLE' if human_state==D else 'NEGATIVE'],len(zz))))

writecsv('CLASSIFICATION_METRICS.csv',metrics)
writecsv('CONFUSION_BY_HUMAN_STATE.csv',confusion)
writecsv('PER_RECORDING_METRICS.csv',sessions)
writecsv('PAIRED_OUTCOME_COUNTS.csv',pairs)
writecsv('INITIAL_ERROR_STRATA.csv',error_metrics)
writecsv('INITIAL_ERROR_DISTRIBUTION.csv',error_distribution)
writecsv('EVALUATED_CALLS.csv',calls)
inventory=[]
for group in GROUPS:
    z=[r for r in rows if r['group']==group]
    for state in sorted({r['human_state'] for r in z}):
        zz=[r for r in z if r['human_state']==state]
        inventory.append(dict(group=group,human_state=state,N=len(zz),query_inside=sum(yes(r,'query_valid') for r in zz),query_OOB=sum(not yes(r,'query_valid') for r in zz),same_coordinate_RGB=sum(yes(r,'legacy_direct_score_comparable_location') for r in zz)))
writecsv('DENOMINATOR_INVENTORY.csv',inventory)
summary={'date_Asia_Seoul':'2026-10-22','new_fits':0,'new_forward_calls':0,'new_PnP_calls':0,'new_threshold_or_crop_changes':0,'classification_metrics_computed':True,'positive_label':D,'recall_denominator':'all human support in the declared cohort, including abstentions/unresolved; never selective-only','precision_denominator':'all predictions of the respective class; unresolved outcomes remain in coverage and recall reporting','square_self_visibility':'N/A: no human SELF_OCCLUDED labels','same_coordinate_vs_pipeline':'Paired RGB excludes geometry-before-RGB S slots. Pipeline preserves S and can use different geometric gate location. Model input crops differ.','interpretation':'Semantic transfer evaluation of query-pixel presence, not real pixel-presence accuracy. No real segmentation GT. Repeated development data, not independent final validation.','initial_error_diagnostics':'Reference pixel error and fixed 3/10/30 px diagnostic bins are descriptive, not model/threshold selection or causal intervention.','source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST)},'source_contract':{'path':str(PREP/'EVALUATION_CONTRACT.json'),'sha256':sha(PREP/'EVALUATION_CONTRACT.json')},'script_sha256':sha(Path(__file__)),'metrics':metrics,'paired_changes':pairs}
(HERE/'EVALUATION_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for m in metrics:
    if m['task']=='D_vs_E' and m['cohort']=='paired_same_query_RGB' or m['task']=='D_vs_S' and m['cohort']=='same_slot_pipeline':
        print(json.dumps(m,ensure_ascii=False))
