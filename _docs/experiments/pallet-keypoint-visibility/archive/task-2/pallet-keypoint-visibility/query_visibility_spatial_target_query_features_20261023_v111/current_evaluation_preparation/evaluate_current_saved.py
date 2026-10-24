"""Inert C11 saved-current evaluator. Actual I/O requires a root evaluation entry.

The pinned V94 cohort/annotation/legacy definitions are reused without changing
their denominators. A raw binary hidden response is not a learned self class.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import numpy as np
import current_adapter as c


def validate_raw(core, records, raw):
    """Pure CSV/NPZ consistency; no data loader or model is called."""
    n=1784
    c.require(len(records)==n,'All original1784 CSV rows required')
    for field,dtype,shape in [('query_logit',np.float32,(n,)),('valid',np.bool_,(n,)),
                              ('query_available',np.bool_,(n,)),('prior_valid',np.bool_,(n,)),
                              ('context_available',np.bool_,(n,)),('local_central4_available',np.bool_,(n,)),
                              ('context_Q_inside',np.bool_,(n,))]:
        c.require(raw[field].dtype==dtype and raw[field].shape==shape,'Raw schema: '+field)
    c.require(np.isfinite(raw['query_logit']).all(),'Nonfinite raw logits')
    c.require(raw['original_query_xy'].shape==(n,2)and raw['original_side_L'].shape==(n,),'Original q/L schema')
    c.require(np.array_equal(raw['valid'],raw['prior_valid']&raw['context_available']&raw['local_central4_available']),
              'Factual/context availability changed')
    c.require(np.array_equal(raw['query_available'],raw['valid']),'Operator availability changed')
    c.require(int((~raw['prior_valid']).sum())==45,'Prior45U changed')
    for i,r in enumerate(records):
        c.require(float(raw['query_logit'][i])==float(r['raw_query_aux_hidden_logit']),'Raw/CSV scalar mismatch')
        for field in ('valid','prior_valid','context_available','local_central4_available','context_Q_inside'):
            c.require(bool(raw[field][i])==core.boolean(r[field]),'Raw/CSV '+field+' mismatch')
        c.require(str(raw['id'][i])==r['id']and int(raw['native_corner'][i])==int(r['native_corner']),'Raw/CSV order mismatch')
        for axis,field in enumerate(('query_x','query_y')):
            c.require(core.numeric_same(raw['original_query_xy'][i,axis],r[field]),'Original '+field+' changed')
        c.require(core.numeric_same(raw['original_side_L'][i],r['roi_side']),'Original L changed')
        c.require(r['context_box_mode']==c.BOX_MODE and r['context_bbox_origin']=='INITIAL_Q8_DERIVED_NOT_SELECTED_DETECTION',
                  'Context provenance changed')
    return dict(rows=n,original_q_L_ID_order_exact=True,original_input_U=45,
                total_U=int((~raw['valid']).sum()),extra_U=int((raw['prior_valid']&~raw['valid']).sum()),
                Q_outside_does_not_add_U=True,context_Q_inside=int(raw['context_Q_inside'].sum()))


def error_bins(core,base,truth,manual,newrows,oldrows):
    """General existing q-to-reference error, separate from manual-near-D table."""
    new=core.score_index(newrows,'raw_query_aux_hidden_logit')
    old=core.score_index(oldrows,'raw_physical_external_logit')
    provenance={core.key(r):r for r in manual};original=base.call;out=[]
    def call(r,model,task,probabilities,threshold,rules=None):
        if model in(c.NEW,c.NEW_GEO):return core.response(new[core.key(r)],model==c.NEW_GEO)
        if model in(c.OLD,c.OLD_GEO):return core.response(old[core.key(r)],model==c.OLD_GEO)
        return original(r,model,task,probabilities,threshold,rules)
    base.call=call
    try:
        for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
            for group in base.GROUPS:
                for name in ('LE3','GT3','MISSING_REFERENCE_ERROR'):
                    selected=[]
                    for r in truth:
                        if r['group']!=group or not base.yes(r,cohort+'_D_vs_E')or r['human_state']not in(base.D,base.E):continue
                        try:v=float(r.get('initial_reference_error_px',''))
                        except(TypeError,ValueError):v=float('nan')
                        bucket='MISSING_REFERENCE_ERROR'if not math.isfinite(v)else'LE3'if v<=3 else'GT3'
                        if bucket==name:selected.append(r)
                    sources=Counter(provenance.get(core.key(r),{}).get('annotation_coordinate_source','UNSPECIFIED')for r in selected)
                    for model in c.MODELS:
                        out.append(dict(cohort=cohort,group=group,task='D_vs_E',error_bin=name,model=model,
                            manual_click_reference_N=sources['manual_click'],other_or_unspecified_reference_N=len(selected)-sources['manual_click'],
                            scope='Existing general q-reference error; not coordinate correction or verified query-miss truth',
                            **base.calculate(selected,model,'D_vs_E',{},.5)))
    finally:base.call=original
    return out


def matrices(metrics):
    """Two human rows D/E; three prediction columns D/binary-hidden/U."""
    result=[]
    for r in metrics:
        if r['task']!='D_vs_E':continue
        ud=r['U_visible']+r['out_of_task_S_visible']+r['other_unresolved_visible']
        ue=r['U_occluded']+r['out_of_task_S_occluded']+r['other_unresolved_occluded']
        result.append(dict(cohort=r['cohort'],group=r['group'],model=r['model'],
            human_D_N=r['visible_support'],human_external_E_N=r['occluded_support'],
            D_pred_D=r['TP'],D_pred_hidden=r['FN_decided'],D_pred_U=ud,
            E_pred_D=r['FP'],E_pred_hidden=r['TN'],E_pred_U=ue,
            D_precision=r['visible_precision'],D_recall=r['visible_recall_all_GT'],
            external_E_response_precision=r['occluded_precision'],external_E_response_recall=r['occluded_recall_all_GT'],
            unresolved_rate=r['unresolved_rate'],
            hidden_response_not_learned_S=True))
    return result


def run(entry_path):
    e,b=c.validate_entry(entry_path,'EVALUATE_SAVED',code_path=__file__)
    for field in ('inference_seal','scores','raw_outputs','preparation'):c.check(e[field])
    seal=c.read(e['inference_seal']['path'])
    for field,value in dict(completed=True,frames=223,slots=1784,model_unchanged=True,current_GT_read=False,
                            current_used_for_selection=False,new_initializer=0,optimizer_updates=0,PnP=0).items():
        c.require(seal.get(field)==value,'Complete frozen inference required: '+field)
    c.require(c.same(seal['code'],e['common_code'])and c.same(seal['source_model_freeze'],e['source_freeze'])
              and c.same(seal['checkpoint'],e['checkpoint']),'Inference implementation/source mismatch')
    c.require(seal['D_rule']=='z<0'and seal['raw_E_threshold_logit']==0,'Frozen sign decoder changed')
    c.require(all(c.contains(seal['outputs'],e[k])for k in ('scores','raw_outputs'))
              and c.same(seal['preparation'],e['preparation']),'Unsealed evaluation inputs')
    # Current GT/baseline body and SHA reads begin only after the complete seal.
    for field in c.EVAL_KEYS|{'prior_input_rows'}:c.check(b[field])
    core=c.pure_core();prior=c.prior_reader().rows(b['prior_input_rows']['path'])
    truth=core.readcsv(b['evaluation_manifest']['path']);manual=core.readcsv(b['manual_provenance']['path'])
    records=core.readcsv(e['scores']['path']);old=core.readcsv(b['baseline_v91_scores']['path'])
    oldseal=c.read(b['baseline_v91_inference_seal']['path'])
    c.require(oldseal['completed']and c.contains(oldseal['outputs'],b['baseline_v91_scores']),'Historical V91 scores not sealed')
    with np.load(e['raw_outputs']['path'],allow_pickle=False)as raw:raw_review=validate_raw(core,records,raw)
    base=c.metric_base();tables=core.calculate_all(base,truth,prior,records,old,manual)
    checked=core.historical_replay(tables['metrics'],core.readcsv(b['original_baseline_metrics']['path']),{'v61','legacy'})
    checked+=core.historical_replay(tables['metrics'],core.readcsv(b['historical_v91_metrics']['path']),{c.OLD,c.OLD_GEO})
    tables['coordinate_error_bins']=error_bins(core,base,truth,manual,records,old)
    tables['D_vs_external_E_2x3']=matrices(tables['metrics'])
    states=Counter(r.get('human_state','')or'MISSING_HUMAN_STATE'for r in truth)
    out=Path(e['output_directory']);out.mkdir(parents=True,exist_ok=False)
    for name in ('metrics','confusion','human_self_binary_response','manual_near_D','availability_by_group','coordinate_error_bins','D_vs_external_E_2x3'):
        core.writecsv(out/(name.upper()+'.csv'),tables[name])
    c.write_new(out/'CURRENT_PRIMARY_RESULTS.json',dict(completed=True,tables=tables,raw_consistency=raw_review,
        all_original_native_rows=1784,annotation_state_inventory=dict(states),historical_baseline_metric_rows_exact=checked,
        primary_model=c.NEW,geometry_routes_separate=[c.OLD_GEO,c.NEW_GEO],
        rawsign='D iff z<0; binary hidden response iff z>=0; factual/context U first; probability display only',
        self='Human S is reported separately. Binary hidden is not a learned S or external-cause identification. Existing initial-geometry route is separate and valid-only.',
        input_scope='Frozen initial q8-derived unclipped/unexpanded target box, original q/L/RGB96/native order. No GT-selected box, permutation, recenter or denominator cut.',
        interpretation='Conditional source-cue transfer to existing human D/E annotations; native association remains UNKNOWN. Current223 has historical development exposure; no fresh holdout claim.',
        no_forward_fit_threshold_checkpoint_selection_or_PnP=True,bindings=e))
    c.write_new(out/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,entry=c.bind(entry_path),source_freeze=e['source_freeze'],
        inference=e['inference_seal'],files=[c.bind(p)for p in sorted(out.iterdir())if p.is_file()],new_forward_fit_current_selection_PnP=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);run(parser.parse_args().entry)
