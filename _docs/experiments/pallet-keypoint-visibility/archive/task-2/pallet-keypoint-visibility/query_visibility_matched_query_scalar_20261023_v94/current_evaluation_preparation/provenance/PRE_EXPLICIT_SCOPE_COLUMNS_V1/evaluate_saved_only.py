"""GT-only evaluator after source gates + complete one-shot inference seal.

No model/Torch/checkpoint deserialization or RGB/P3 input loader is used here.
Raw hidden is a binary AUX response, never a learned external/self classifier.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import math

import numpy as np

import current_common as c

NEW='V94_C1_RAW';NEW_GEO='V94_C1_PLUS_OLD_GEOMETRY'
OLD='V91_RAW';OLD_GEO='V91_PLUS_OLD_GEOMETRY'
MODELS=('v61','legacy',OLD,OLD_GEO,NEW,NEW_GEO)


def readcsv(path):
    with Path(path).open(newline='')as f:return list(csv.DictReader(f))


def writecsv(path,rows):
    if not rows:return
    with Path(path).open('x',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def key(r):return r['id'],str(r['native_corner'])


def boolean(value,nullable=False):
    if value is True or str(value).lower()=='true':return True
    if value is False or str(value).lower()=='false':return False
    if nullable and(value is None or str(value).lower()in('','none','null')):return None
    raise ValueError('Malformed boolean')


def numeric_same(a,b):
    missing=lambda x:x is None or str(x).lower()in('','none','nan')
    return (missing(a)and missing(b))or(not missing(a)and not missing(b)and float(a)==float(b))


def score_index(rows,raw_field):
    c.require(len(rows)==1784,'All1784 scores required');result={}
    for r in rows:
        k=key(r);c.require(k not in result,'Duplicate native key')
        valid=boolean(r['valid']);z=float(r[raw_field]);c.require(math.isfinite(z),'Finite raw scalar required')
        geo=boolean(r['geometry_self_hidden'],True);guarded=boolean(r['geometry_self_hidden_if_valid'],True)
        c.require(guarded==(geo if valid else None),'Invalid U/geometry precedence changed')
        c.require(r['authoritative_decision']==(('D'if z<0 else'E')if valid else'UNKNOWN'),'Raw sign/export mismatch')
        c.require(boolean(r['decision_visible'],True)==((z<0)if valid else None),'Raw sign/visible mismatch')
        c.require(bool(r.get('p_visible',''))==valid,'Invalid row gained finite main score')
        if valid:c.require(math.isfinite(float(r['p_visible']))and 0<=float(r['p_visible'])<=1,'Display probability malformed')
        result[k]={'row':r,'valid':valid,'z':z,'geo':guarded}
    return result


def response(record,geometry=False):
    if not record['valid']:return 'OTHER_UNRESOLVED'
    if geometry and record['geo']is True:return 'ABSENT_NONVISIBLE'
    return 'VISIBLE'if record['z']<0 else'ABSENT_NONVISIBLE'


def join_all(truth,prior,new,old):
    tm={key(r):r for r in truth};pm={key(r):r for r in prior}
    c.require(len(tm)==len(pm)==1784 and set(new)==set(old)==set(tm)==set(pm),'Exact1784 key coverage required')
    c.require(list(new)==list(old)==list(pm),'Original native output order changed')
    for k,value in new.items():
        n=value['row'];o=old[k]['row'];p=pm[k];t=tm[k]
        c.require(n['image_sha256']==o['image_sha256']==p['image_sha256']==t['image_sha256'],'Image join changed')
        for field in ('query_x','query_y','roi_side','pixel_x','pixel_y'):
            c.require(numeric_same(n[field],o[field])and numeric_same(n[field],p[field]),'Original '+field+' changed')
        for field in ('geometry_self_hidden','saved_roi_pixel_valid'):
            c.require(boolean(n[field],True)==boolean(o[field],True)==p[field],'Original factual/geometry flag changed')
        c.require(not value['valid']or boolean(p['valid']),'Original inputU resolved')
        c.require(not old[k]['valid']or boolean(p['valid']),'Historical inputU resolved')
        for field in ('query_x','query_y'):
            if n[field]and t[field]:c.require(abs(float(n[field])-float(t[field]))<=1e-5,'Registered query changed')
    c.require(sum(not boolean(r['valid'])for r in prior)==45,'Original45U changed')
    return tm,pm


def calculate_all(base,truth,prior,new_rows,old_rows,manual,expected_manual=True):
    """Pure supplied-record metrics; imports only the frozen metric code core."""
    new=score_index(new_rows,'raw_query_aux_hidden_logit');old=score_index(old_rows,'raw_physical_external_logit')
    tm,pm=join_all(truth,prior,new,old);original_call=base.call
    def call(row,model,task,probabilities,threshold,rules=None):
        if model in(NEW,NEW_GEO):return response(new[key(row)],model==NEW_GEO)
        if model in(OLD,OLD_GEO):return response(old[key(row)],model==OLD_GEO)
        return original_call(row,model,task,probabilities,threshold,rules)
    base.call=call
    metrics=[];confusion=[];self_response=[];near=[]
    try:
        for cohort in('paired_same_query_RGB','same_slot_pipeline'):
            for group in base.GROUPS:
                for task,states in base.TASKS.items():
                    flag=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
                    eligible=[r for r in truth if r['group']==group and base.yes(r,flag)and r['human_state']in states]
                    if not any(r['human_state']!=base.D for r in eligible):continue
                    for model in MODELS:
                        metrics.append(dict(cohort=cohort,group=group,task=task,model=model,**base.calculate(eligible,model,task,{},.5)))
                        for state in states:
                            rr=[r for r in eligible if r['human_state']==state];counts=Counter(call(r,model,task,{},.5)for r in rr)
                            confusion.append(dict(cohort=cohort,group=group,task=task,model=model,human_state=state,N=len(rr),
                                pred_visible=counts['VISIBLE'],pred_binary_hidden=counts['ABSENT_NONVISIBLE'],legacy_U=counts['ABSTAIN'],
                                out_of_task_geometry_S=counts['OUT_OF_TASK_GEOMETRY_S'],input_or_operator_U=counts['OTHER_UNRESOLVED']))
                rr=[r for r in truth if r['group']==group and base.yes(r,cohort+'_D_vs_ES')and r['human_state']==base.S]
                for model in MODELS:
                    counts=Counter(call(r,model,'D_vs_S',{},.5)for r in rr)
                    self_response.append(dict(cohort=cohort,group=group,model=model,human_S_support=len(rr),
                        binary_hidden_response=counts['ABSENT_NONVISIBLE'],visible_response=counts['VISIBLE'],
                        unresolved=len(rr)-counts['ABSENT_NONVISIBLE']-counts['VISIBLE'],
                        human_S_binary_hidden_response_rate=base.ratio(counts['ABSENT_NONVISIBLE'],len(rr)),
                        learned_self_prediction=False,external_vs_self_identified=False,
                        scope='HumanS subset apparentvisibility binary response; oldgeometry route separately named. ZeroS support is no gold evidence.'))
        selected=[r for r in manual if r['annotation_coordinate_source']=='manual_click'and r['human_state']==base.D
                  and float(r['initial_reference_error_px'])<=3 and base.yes(r,'query_valid')]
        for r in selected:
            t=tm[key(r)];c.require(all(r[f]==t[f]for f in('image_sha256','query_x','query_y')),'Manualnear join changed')
        expected={'WOOD_PRIMARY105':153,'SQUARE118':223}
        for group in base.GROUPS:
            rr=[r for r in selected if r['group']==group]
            if expected_manual:c.require(len(rr)==expected[group],'Historical manualnear denominator changed')
            for model in MODELS:
                counts=Counter(call(r,model,'D_vs_ES',{},.5)for r in rr)
                near.append(dict(group=group,model=model,scope='MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3',N=len(rr),
                    pred_visible=counts['VISIBLE'],pred_binary_hidden=counts['ABSENT_NONVISIBLE'],
                    unresolved=len(rr)-counts['VISIBLE']-counts['ABSENT_NONVISIBLE'],semantic_D_recall=base.ratio(counts['VISIBLE'],len(rr))))
    finally:base.call=original_call
    priorU=sum(not boolean(r['valid'])for r in prior);newU=sum(not v['valid']for v in new.values())
    availability=dict(rows=1784,prior_input_U=priorU,new_total_U=newU,extra_operator_U=newU-priorU,
                      V91_total_U=sum(not v['valid']for v in old.values()),confidence_abstention_trained=False)
    return dict(metrics=metrics,confusion=confusion,human_self_binary_response=self_response,manual_near_D=near,availability=availability)


def historical_replay(metrics,historical,models):
    old={(r['cohort'],r['group'],r['task'],r['model']):r for r in historical if r['model']in models}
    checked=0
    for row in metrics:
        if row['model']not in models:continue
        original=old[(row['cohort'],row['group'],row['task'],row['model'])]
        for field in('N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded'):
            c.require(int(row[field])==int(original[field]),'Historical denominator/count drift')
        checked+=1
    c.require(checked==len(old),'Historical row coverage changed')
    return checked


def run(entry_path):
    e,policy,b=c.validate_entry(entry_path,__file__,'EVALUATE_SAVED')
    for field in('inference_seal','scores','raw_outputs','preparation'):c.check(e[field])
    seal=c.read(e['inference_seal']['path'])
    for field,value in dict(completed=True,frames=223,slots=1784,model_unchanged=True,current_used_for_selection=False,current_GT_read=False,new_backbone=0,new_initializer=0,optimizer_updates=0,PnP=0).items():
        c.require(seal.get(field)==value,'Inference scope changed: '+field)
    c.require(c.same(seal['source_model_freeze'],b['source_model_freeze'])and c.same(seal['checkpoint'],b['checkpoint']),'Source model changed')
    c.require(seal['D_rule']=='z<0'and seal['raw_E_threshold_logit']==0,'Raw decoder changed')
    for field in('scores','raw_outputs'):c.require(any(c.same(d,e[field])for d in seal['outputs']),'Unsealed output')
    c.require(c.same(seal['preparation'],e['preparation']),'Input preparation mismatch')
    # GT/baseline bodies and SHA reads start only here, after complete inference.
    for field in c.EVAL_KEYS|{'prior_input_rows'}:c.check(b[field])
    prior=c.prior_reader().rows(b['prior_input_rows']['path'])
    records=readcsv(e['scores']['path']);old=readcsv(b['baseline_v91_scores']['path'])
    oldseal=c.read(b['baseline_v91_inference_seal']['path'])
    c.require(oldseal['completed']and any(c.same(d,b['baseline_v91_scores'])for d in oldseal['outputs']),'HistoricalV91 not sealed')
    with np.load(e['raw_outputs']['path'],allow_pickle=False)as raw:
        c.require(raw['query_logit'].shape==(1784,)and raw['query_logit'].dtype==np.float32,'Raw scalar schema changed')
        for i,row in enumerate(records):
            c.require(float(raw['query_logit'][i])==float(row['raw_query_aux_hidden_logit']),'CSV/raw scalar mismatch')
            c.require(bool(raw['valid'][i])==boolean(row['valid']),'CSV/raw valid mismatch')
            c.require(str(raw['id'][i])==row['id']and int(raw['native_corner'][i])==int(row['native_corner']),'CSV/raw order mismatch')
    base=c.metric_base()
    tables=calculate_all(base,readcsv(b['evaluation_manifest']['path']),prior,records,old,readcsv(b['manual_provenance']['path']))
    verified=historical_replay(tables['metrics'],readcsv(b['original_baseline_metrics']['path']),{'v61','legacy'})
    verified+=historical_replay(tables['metrics'],readcsv(b['historical_v91_metrics']['path']),{OLD,OLD_GEO})
    out=Path(e['output_directory']);out.mkdir(parents=True,exist_ok=False)
    for name in('metrics','confusion','human_self_binary_response','manual_near_D'):writecsv(out/(name.upper()+'.csv'),tables[name])
    c.write_new(out/'CURRENT_PRIMARY_RESULTS.json',dict(completed=True,tables=tables,historical_baseline_metric_rows_exact=verified,
        primary_model=NEW,oldgeometry_routes_separate=[OLD_GEO,NEW_GEO],rawsign='D iff z<0; hiddenE auxiliary iff z>=0; factualU first',
        scope='Public conditional AUX cue D/hiddenE -> human D/E transfer; nativeassociation UNKNOWN. Current223 is historically development-exposed; no exactpixel semanticgold equivalence.',
        self='HumanS tables are binaryhidden responses/apparentvisibility CM, not learnedself predictions. E-versus-S not identified. Oldgeometry is a separately named existing proxy route.',
        no_model_forward_fit_threshold_or_checkpoint_selection_or_PnP=True,bindings=e))
    c.write_new(out/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,entry=c.bind(entry_path),source_freeze=b['source_model_freeze'],inference=e['inference_seal'],files=[c.bind(p)for p in sorted(out.iterdir())if p.is_file()],new_forward_fit_current_selection_PnP=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
