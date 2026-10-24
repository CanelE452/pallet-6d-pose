"""Independent sealed saved-current/source metric replay; no model/image imports."""
import ast
import csv
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
CONTRACT=ROOT/'contract'
PRIVATE=ROOT/'private_evaluation/CUE_ATTENTION_V3'
D,E,S='DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED'
TASKS={'D_vs_E':[D,E],'D_vs_S':[D,S],'D_vs_ES':[D,E,S]}


def csvread(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def yes(x):return str(x).lower()=='true'
def div(a,b):return a/b if b else None


def main():
    manifest=PARENT/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'
    nearfile=PARENT/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv'
    rows=csvread(manifest);scores=csvread(PRIVATE/'REAL_QUERY_SCORES.csv')
    inputs=[json.loads(x) for x in (PRIVATE/'ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
    inference=read(PRIVATE/'INFERENCE_SEAL.json');assert inference['completed'] and inference['model_state_unchanged']
    for b in inference['outputs']:
        assert bind(b['path'])['sha256']==b['sha256']
    assert len(rows)==len(scores)==len(inputs)==1784 and len({r['id'] for r in rows})==223
    key=lambda r:(r['id'],str(r['native_corner']))
    saved={key(r):r for r in scores};original={key(r):r for r in rows}
    assert len(saved)==len(original)==1784 and set(saved)==set(original)
    with np.load(PRIVATE/'RAW_NETWORK_OUTPUTS.npz',allow_pickle=False) as z:
        assert set(z.files)=={'logit_D_given_DE','cue_heatmap_logits','cue_attention_weights','valid','network_callable'}
        rawz=z['logit_D_given_DE'];rawvalid=z['valid'];callable_=z['network_callable']
    assert rawz.shape==rawvalid.shape==callable_.shape==(1784,)
    identity_deltas={'query_x':0.,'query_y':0.,'roi_side':0.}
    for i,(m,s,inp) in enumerate(zip(rows,scores,inputs)):
        assert key(m)==key(s)==key(inp)
        assert m['image_sha256']==s['image_sha256']==inp['image_sha256']
        for name in ['query_x','query_y']:
            delta=abs(float(m[name])-float(s[name]));identity_deltas[name]=max(identity_deltas[name],delta)
            assert delta<=1e-5 and float(s[name])==float(inp[name])
        delta=abs(float(m['v61_roi_side_raw_px'])-float(s['roi_side']));identity_deltas['roi_side']=max(identity_deltas['roi_side'],delta)
        assert delta<=1e-5 and float(s['roi_side'])==float(inp['roi_side'])
        assert int(s['pixel_x'])==math.floor(float(m['query_x'])+.5)==inp['pixel_x']
        assert int(s['pixel_y'])==math.floor(float(m['query_y'])+.5)==inp['pixel_y']
        assert yes(s['valid'])==yes(m['query_valid'])==bool(inp['valid'])==bool(rawvalid[i])
        assert yes(s['network_callable'])==bool(inp['network_callable'])==bool(callable_[i])
        assert yes(s['saved_roi_pixel_valid'])==bool(inp['saved_roi_pixel_valid'])==yes(m['roi_has_valid_sample'])
        assert s['geometry_self_hidden'].lower()==str(inp['geometry_self_hidden']).lower()
        assert not yes(s['geometry_overrides_learned_probability']) and not yes(s['semantic_U_detector']) and not yes(s['self_occlusion_learned'])
        if yes(s['valid']):
            p=1/(1+math.exp(-float(rawz[i])))
            assert float(s['logit'])==float(rawz[i]) and abs(float(s['p_visible'])-p)<1e-12
            assert s['p_visible']==s['p_D_given_DE'] and abs(float(s['p_E_given_DE'])-(1-p))<1e-12
            assert yes(s['present_at_frozen_threshold'])==(float(rawz[i])>=0)
        else:
            assert s['status']=='UNKNOWN' and s['predicted_D_or_E']=='UNKNOWN'
            assert not s['logit'] and not s['p_visible'] and not s['p_D_given_DE'] and not s['p_E_given_DE']
    assert int(rawvalid.sum())==inference['valid']==1739
    models=['v61','legacy','V72_CUE_ATTENTION','V72_CUE_ATTENTION+GEOMETRY']
    def call(r,model,task):
        if model=='v61':return 'V' if yes(r['present_at_fixed_0_5']) else 'A'
        if model=='legacy':
            old=r['legacy_stored_state']
            return 'V' if old in ['B','I'] else 'A' if old=='O' or (old=='S' and task!='D_vs_E') else 'S' if old=='S' else 'U' if old=='U' else 'X'
        s=saved[key(r)]
        if model.endswith('+GEOMETRY') and yes(s['geometry_self_hidden']):return 'A'
        return 'V' if s['p_visible'] and float(s['p_visible'])>=.5 else 'A' if s['p_visible'] else 'X'
    def metrics(rr,model,task):
        tab=Counter((r['human_state']==D,call(r,model,task)) for r in rr)
        pos=sum(r['human_state']==D for r in rr);neg=len(rr)-pos
        tp,fn,fp,tn=tab[True,'V'],tab[True,'A'],tab[False,'V'],tab[False,'A']
        up,un,sp,sn,xp,xn=tab[True,'U'],tab[False,'U'],tab[True,'S'],tab[False,'S'],tab[True,'X'],tab[False,'X']
        return dict(N=len(rr),frames=len({r['id'] for r in rr}),visible_support=pos,occluded_support=neg,
                    TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,
                    out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=xp,other_unresolved_occluded=xn,
                    visible_precision=div(tp,tp+fp),visible_recall_all_GT=div(tp,pos),occluded_precision=div(tn,tn+fn),occluded_recall_all_GT=div(tn,neg),
                    abstention_rate=div(up+un,len(rr)),out_of_task_S_rate=div(sp+sn,len(rr)),unresolved_rate=div(up+un+sp+sn+xp+xn,len(rr)),
                    decision_coverage=div(tp+fn+fp+tn,len(rr)),correct_fraction_all_GT=div(tp+tn,len(rr)),selective_accuracy=div(tp+tn,tp+fn+fp+tn))
    def cohort(c,g,t):
        flag=c+'_'+('D_vs_ES' if t=='D_vs_S' else t)
        return [r for r in rows if r['group']==g and yes(r[flag]) and r['human_state'] in TASKS[t]]
    comparisons=0;maxdelta=0.;mismatches=[]
    def compare(actual,expected,context):
        nonlocal comparisons,maxdelta
        for name,a in actual.items():
            b=expected[name];comparisons+=1
            if a is None:
                if b not in [None,'']:mismatches.append((context,name,a,b))
            elif isinstance(a,(int,float)):
                delta=abs(float(a)-float(b));maxdelta=max(maxdelta,delta)
                if delta>1e-11:mismatches.append((context,name,a,b))
            elif a!=b:mismatches.append((context,name,a,b))
    table=csvread(CONTRACT/'V72_CURRENT_CLASSIFICATION_METRICS.csv')
    recomputed=[]
    for reported in table:
        rr=cohort(reported['cohort'],reported['group'],reported['task'])
        out=metrics(rr,reported['model'],reported['task']);compare(out,reported,reported['model']+str((reported['cohort'],reported['group'],reported['task'])))
        recomputed.append({k:reported[k] for k in ['cohort','group','task','model']}|out)
    # Every historical baseline cell must also match the unchanged original table.
    oldrows=csvread(PARENT/'saved_output_evaluation_20261022/CLASSIFICATION_METRICS.csv')
    oldmap={(r['cohort'],r['group'],r['task'],r['model']):r for r in oldrows}
    for r in recomputed:
        if r['model'] in ['v61','legacy']:
            old=oldmap[tuple(r[k] for k in ['cohort','group','task','model'])]
            compare({k:v for k,v in r.items() if k not in ['cohort','group','task','model']},old,'unchanged_old_baseline')
    for reported in csvread(CONTRACT/'V72_CURRENT_CONFUSION_BY_HUMAN_STATE.csv'):
        rr=[r for r in cohort(reported['cohort'],reported['group'],reported['task']) if r['human_state']==reported['human_state']]
        c=Counter(call(r,reported['model'],reported['task']) for r in rr)
        compare(dict(N=len(rr),pred_visible=c['V'],pred_absent_nonvisible=c['A'],pred_U=c['U'],pred_out_of_task_geometry_S=c['S'],pred_other_unresolved=c['X']),reported,'state_confusion')
    for reported in csvread(CONTRACT/'V72_CURRENT_PER_RECORDING_METRICS.csv'):
        rr=[r for r in cohort(reported['cohort'],reported['group'],reported['task']) if r['recording']==reported['recording']]
        compare(metrics(rr,reported['model'],reported['task']),reported,'recording_metrics')
    for reported in csvread(CONTRACT/'V72_CURRENT_ERROR_STRATA.csv'):
        rr=[r for r in rows if r['group']==reported['group'] and yes(r[reported['cohort']+'_D_vs_ES']) and r['human_state']==reported['human_state']]
        label=reported['error_bin']
        def keep(r):
            err=float(r['initial_reference_error_px'])
            return err<=3 if label=='px_le3' else 3<err<=10 if label=='px_3_to10' else 10<err<=30 if label=='px_10_to30' else err>30 if label=='px_gt30' else yes(r['human_reference_inside_nominal_crop']) if label=='reference_inside_crop' else not yes(r['human_reference_inside_nominal_crop'])
        rr=[r for r in rr if keep(r)];c=Counter(call(r,reported['model'],'D_vs_ES') for r in rr)
        compare(dict(N=len(rr),pred_visible=c['V'],pred_absent_nonvisible=c['A'],pred_U=c['U'],pred_other_unresolved=c['X'],class_recall_all_GT=div(c['V' if reported['human_state']==D else 'A'],len(rr))),reported,'error_strata')
    for reported in csvread(CONTRACT/'V72_CURRENT_GEOMETRY_TRADEOFF.csv'):
        rr=[r for r in cohort(reported['cohort'],reported['group'],reported['task']) if r['human_state']==reported['human_state']]
        a=Counter(call(r,'V72_CUE_ATTENTION',reported['task']) for r in rr);b=Counter(call(r,'V72_CUE_ATTENTION+GEOMETRY',reported['task']) for r in rr)
        correct='V' if reported['human_state']==D else 'A'
        compare(dict(N=len(rr),geometry_true=sum(yes(saved[key(r)]['geometry_self_hidden']) for r in rr),geometry_unknown=sum(saved[key(r)]['geometry_self_hidden'].lower() in ['', 'none'] for r in rr),raw_pred_visible=a['V'],hybrid_pred_visible=b['V'],suppressed_raw_visible=a['V']-b['V'],raw_class_recall=div(a[correct],len(rr)),hybrid_class_recall=div(b[correct],len(rr))),reported,'geometry_tradeoff')
    for reported in csvread(CONTRACT/'V72_CURRENT_PAIRED_CHANGE.csv'):
        rr=cohort(reported['cohort'],reported['group'],reported['task']);c=Counter()
        for r in rr:
            a,b=call(r,reported['baseline'],reported['task']),call(r,reported['new_model'],reported['task']);truth='V' if r['human_state']==D else 'A'
            if b not in ['V','A']:c['new_unresolved']+=1
            elif a not in ['V','A']:c['baseline_unresolved_new_correct' if b==truth else 'baseline_unresolved_new_wrong']+=1
            elif a==truth and b==truth:c['both_correct']+=1
            elif a==truth:c['baseline_correct_new_wrong']+=1
            elif b==truth:c['baseline_wrong_new_correct']+=1
            else:c['both_wrong']+=1
        fields=['both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved']
        compare(dict(N=len(rr),**{k:c[k] for k in fields}),reported,'paired_changes')
    near=csvread(nearfile);manual=[r for r in near if r['annotation_coordinate_source']=='manual_click']
    assert Counter(r['group'] for r in manual)=={'WOOD_PRIMARY105':153,'SQUARE118':223}
    assert all(r['human_state']==D and float(r['initial_reference_error_px'])<=3 and yes(r['query_valid']) for r in manual)
    for r in manual:
        before=original[key(r)]
        assert before['image_sha256']==r['image_sha256'] and before['query_x']==r['query_x'] and before['query_y']==r['query_y']
    manual_report=[]
    for reported in csvread(CONTRACT/'V72_CURRENT_MANUAL_NEAR_D_LE3.csv'):
        rr=[r for r in manual if r['group']==reported['group']];c=Counter(call(r,reported['model'],'D_vs_ES') for r in rr)
        actual=dict(N=len(rr),pred_visible=c['V'],pred_absent_nonvisible=c['A'],pred_U=c['U'],pred_other_unresolved=c['X'],semantic_D_recall=div(c['V'],len(rr)))
        compare(actual,reported,'manual_click_only_near_D')
        manual_report.append(dict(group=reported['group'],model=reported['model'],**actual))
    # Cross-reconcile root source tables with the previously sealed independent source audit.
    source=read(ROOT/'source_evaluation/SOURCE_RESULTS.json')
    independent=read(ROOT/'source_audit/INDEPENDENT_SAVED_SOURCE_METRIC_RECONCILIATION.json')
    renames={'truth_D':'D','truth_E':'E','E_precision_conditional_knownDE':'E_precision','unweighted_NLL':'known_NLL'}
    for model,tag in [('V72','V72_CUE_ATTENTION_V3'),('v61','V61_PIXEL_HEAD_ON_SAME_INPUT')]:
        mine=independent['metrics'][tag];theirs=source['models'][model]
        for group in ['by_source_split','by_source_split_cue']:
            for k,a in mine[group].items():
                name='|'.join(ast.literal_eval(k));b=theirs['cohorts'][name]
                compare({renames.get(kk,kk):vv for kk,vv in a.items()},b,'source_cohort_'+model)
        for k,a in mine['paired'].items():
            name='|'.join(ast.literal_eval(k))
            if name not in theirs['paired']:continue
            b=theirs['paired'][name]
            compare(dict(N=a['eligible_known_pair'],both_correct=a['both_correct'],original_D=a['original_D_pred_D'],intervention_correct=a['intervention_correct']),b,'source_pairs_'+model)
        u=defaultdict(Counter)
        for a in mine['unknown_output_distributions']:
            name='|'.join([a['source'],a['split'],a['ureason_recorded'],a['availability']])
            for k in ['N','raw_D','raw_E']:u[name][k]+=a[k]
        assert set(u)==set(theirs['unknown_output_distribution'])
        for k,a in u.items():compare(dict(a),theirs['unknown_output_distribution'][k],'source_U_'+model)
        for k,a in mine['by_GT_offset'].items():
            name='|'.join(ast.literal_eval(k));b=theirs['by_reference_offset'][name]
            compare({renames.get(kk,kk):vv for kk,vv in a.items()},b,'source_offset_'+model)
        compare(dict(null_reference_U_rows_excluded_from_offset=52),theirs,'source_nullref_'+model)
    assert source['source_scores_sha256']==next(b['sha256'] for b in independent['bindings'] if b['path'].endswith('/SOURCE_SCORES.npz'))
    assert source['v61_scores_sha256']==next(b['sha256'] for b in independent['bindings'] if b['path'].endswith('/V61_SOURCE_SCORES.npz'))
    assert source['GT_sha256']==next(b['sha256'] for b in independent['bindings'] if b['path'].endswith('/GT_ONLY.jsonl'))
    assert not mismatches,mismatches[:10]
    # Availability/exclusion and raw-output distribution for all slots, not invented accuracy on unknown GT.
    all_states=[]
    for group in ['WOOD_PRIMARY105','SQUARE118']:
        for state in sorted({r['human_state'] for r in rows}):
            rr=[r for r in rows if r['group']==group and r['human_state']==state]
            all_states.append(dict(group=group,human_state=state,N=len(rr),input_valid=sum(yes(r['query_valid']) for r in rr),
                                   raw_calls=dict(Counter(call(r,'V72_CUE_ATTENTION','D_vs_ES') for r in rr)),
                                   geometry_calls=dict(Counter(call(r,'V72_CUE_ATTENTION+GEOMETRY','D_vs_ES') for r in rr))))
    excluded={}
    for kind in ['paired_same_query_RGB','same_slot_pipeline']:
        for task in ['D_vs_E','D_vs_ES']:
            for group in ['WOOD_PRIMARY105','SQUARE118']:
                rr=[r for r in rows if r['group']==group]
                excluded[str((kind,task,group))]=dict(all_slots=len(rr),eligible_flag=sum(yes(r[kind+'_'+task]) for r in rr),
                    excluded_flag=sum(not yes(r[kind+'_'+task]) for r in rr),
                    excluded_human_states=dict(Counter(r['human_state'] for r in rr if not yes(r[kind+'_'+task]))),
                    excluded_status_reasons=dict(Counter(r['exclusion_or_status_reasons'] for r in rr if not yes(r[kind+'_'+task]))))
    report=dict(schema='V72_INDEPENDENT_SEALED_CURRENT_AND_SOURCE_METRIC_AUDIT_V1',completed=True,status='PASS',
      bindings=[bind(p) for p in [manifest,nearfile,PRIVATE/'INFERENCE_SEAL.json',PRIVATE/'REAL_QUERY_SCORES.csv',PRIVATE/'RAW_NETWORK_OUTPUTS.npz',PRIVATE/'ROWS_INPUT_ONLY.jsonl',ROOT/'source_evaluation/SOURCE_RESULTS.json',ROOT/'source_audit/INDEPENDENT_SAVED_SOURCE_METRIC_RECONCILIATION.json',Path(__file__)]]+[bind(p) for p in sorted(CONTRACT.glob('V72_CURRENT*'))],
      slots=1784,frames=223,input_valid=1739,input_invalid=45,network_callable=int(callable_.sum()),input_identity_max_abs_deltas=identity_deltas,
      validity_scope_note='The preserved saved_roi_pixel_valid flag denotes a nonempty saved ROI (1769), not an in-canvas query center. Thirty out-of-frame queries still have partial ROI content; all30 stay validFalse and scores UNKNOWN. Query-valid1739 matches manifest and RAW outputs. Initial auditor alias assumption was corrected; no producer data changed.',
      all_keys_q_side_pixel_SHA_valid_flags_and_saved_raw_logit_parity=True,comparisons=comparisons,max_abs_metric_difference=maxdelta,mismatches=[],
      classification_table_rows=len(table),all_classification_metrics=recomputed,all_human_states_raw_geometry=all_states,cohort_exclusions=excluded,
      manual_click_near_D=dict(N=376,WOOD_PRIMARY105=153,SQUARE118=223,rows=manual_report,
        general_near_D_counts=dict(Counter(r['group'] for r in near)),excluded_nonmanual_wood=85,
        warning='General wood<=3 has238, includes85 nonmanual reference coordinates; it is not the manual153 cohort. No exact-pixel occupancy GT is inferred.'),
      historical_v61_legacy_original_metrics_preserved=True,source_root_independent_cohort_pair_U_offset_comparison=True,
      interpretation='Current negative is absent/nonvisible score, not a learned external/self cause. Saved geometry suppression is separate. Historical current exposure and generic-source/native8 mismatch remain; no fresh holdout/full-goal success claim.',
      evaluator_or_model_imports=0,new_forward_training_feature_RGB_weights_PnP=0,old_files_modified=False,reselection_refit_threshold_changes=False)
    dest=ROOT/'source_audit/INDEPENDENT_SEALED_CURRENT_METRIC_RECONCILIATION.json';assert not dest.exists()
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=bind(dest),comparisons=comparisons,max_abs_difference=maxdelta,slots=1784,valid=1739,manual=manual_report,
      primary=[r for r in recomputed if r['model']=='V72_CUE_ATTENTION' and r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E']),indent=2))


if __name__=='__main__':main()
