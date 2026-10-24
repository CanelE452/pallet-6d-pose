"""Independent saved-score identity/sign/conditional-metric replay; no torch."""
from pathlib import Path
import collections
import hashlib
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent;V=HERE.parent;BASE=V/'source_pair_intake'


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def check(d):assert bind(d['path'])['sha256']==d['sha256']
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def scores(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}


def audit_identity(z):
    rows=lines(BASE/'ALL960_NORMALIZED_INPUT_ONLY.jsonl');lineage=lines(BASE/'SOURCE960_DRIVER_LINEAGE.jsonl')
    gt=lines(BASE/'image_only_v2_exclusive_alpha/OCCURRENCES2880_LINEAGE.jsonl')
    inputs=lines(V/'actual_source_pair_input_pack/ALL2880_INPUT_LINEAGE.jsonl')
    assert z['query_logit'].shape==(2880,)and z['query_logit'].dtype==np.float32 and np.isfinite(z['query_logit']).all()
    assert z['query_available'].dtype==bool and z['prediction'].dtype==np.int8
    assert np.array_equal(z['occurrence_index'],np.arange(2880))and np.array_equal(z['base_row_index'],np.repeat(np.arange(960),3))
    for j,(g,a)in enumerate(zip(gt,inputs)):
        i=j//3;r=rows[i];s=lineage[i]
        for k,val in {'occurrence_id':g['occurrence_id'],'variant':g['variant'],'input_id':r['input_id'],
                      'native_slot_logical_only':r['native_slot_logical_only'],'original_image_sha256':r['image_sha256'],
                      'image_sha256':a['image_sha256'],'source':s['source'],'original_split':r['split']}.items():assert z[k][j]==val,(j,k)
        partition='TRAIN'if r['split']in('TRAIN','TRAIN_LAB')else'VALIDATION'
        assert z['split'][j]==partition
        if r['query_xy']is None:assert np.isnan(z['query_xy_raw'][j]).all()and np.isnan(z['L_raw'][j])
        else:assert np.array_equal(z['query_xy_raw'][j],r['query_xy'])and z['L_raw'][j]==r['L']
        assert z['scalar_target_D0_E1_Uminus1'][j]==g['candidate_aux_hidden_label_GT_only']
        assert bool(z['candidate_known'][j])==(g['candidate_aux_hidden_label_GT_only']>=0)
        assert bool(z['query_available'][j])==a['input_facts']['operator_query_available']
        assert bool(z['accepted_pair'][i])==g['pair_ready']
    prediction=np.where(z['query_available'],(z['query_logit']>=0).astype(np.int8),-1)
    assert np.array_equal(z['prediction'],prediction)
    assert np.isfinite(z['NLL_FP32']).all()
    assert collections.Counter(map(int,z['scalar_target_D0_E1_Uminus1']))=={-1:2687,0:158,1:35}
    return dict(all2880_identity_rawsplit_qL_labels_and_sign_exact=True,all960_acceptedpair_flags_exact=True,
                available=int(z['query_available'].sum()),U=int((~z['query_available']).sum()),known=int(z['candidate_known'].sum()),
                unknown=int((~z['candidate_known']).sum()),valid_exact_zero_logits=int(((z['query_logit']==0)&z['query_available']).sum()),
                unknown_NLL_placeholders_excluded_from_metrics=True)


def metric(z,mask):
    y=z['scalar_target_D0_E1_Uminus1'];pred=z['prediction'];known=mask&(y>=0);unk=mask&(y<0)
    cm=np.array([[int(np.sum(mask&(y==c)&(pred==p)))for p in (0,1,-1)]for c in (0,1)],np.int64)
    div=lambda a,b:None if not b else float(a/b)
    return dict(matrix=cm.tolist(),rows=['D','E'],columns=['D','E','U'],
       D_recall=div(cm[0,0],cm[0].sum()),E_recall=div(cm[1,1],cm[1].sum()),
       D_precision=div(cm[0,0],cm[:,0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),
       known_U_rate=div(cm[:,2].sum(),known.sum()),all_N=int(mask.sum()),known_N=int(known.sum()),unknown_GT_N=int(unk.sum()),
       unknown_GT_predictions={str(p):int(np.sum(unk&(pred==p)))for p in (0,1,-1)})


def all_metrics(z):
    result={}
    for split in ['TRAIN','VALIDATION']:
        mask=z['split']==split
        pair=z['accepted_pair']&(z['split'][::3]==split)
        correct=np.all(z['prediction'].reshape(960,3)==np.array([0,1,0])[None],axis=1)
        result[split]=dict(all_known=metric(z,mask),by_variant={v:metric(z,mask&(z['variant']==v))for v in ['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']},
                         accepted_pairs=int(pair.sum()),all3_correct=int((correct&pair).sum()),all3_correct_rate=None if not pair.any()else float(correct[pair].mean()))
    return result


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        for k,val in expected.items():compare(actual[k],val,path+'/'+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,val in enumerate(expected):compare(actual[i],val,path+'/'+str(i))
    elif isinstance(expected,float):assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)


def main():
    d=V/'source_baselines/C1_FROZEN_ON_V95';complete=read(d/'BASELINE_COMPLETE.json');assert complete['completed']and complete['PASS_execution']
    for k in ['entry','code','model_freeze','checkpoint','input_pack','results','scores']:check(complete[k])
    freeze=read(complete['model_freeze']['path']);assert freeze['checkpoint']['path']==complete['checkpoint']['path']and freeze['checkpoint']['sha256']==complete['checkpoint']['sha256']
    z=scores(complete['scores']['path']);identity=audit_identity(z);metrics=all_metrics(z);published=read(complete['results']['path'])
    compare(metrics,published['metrics']);assert complete['model_unchanged']and published['diagnostic_only']and not published['model_selection_used']
    assert published['selected_update']==freeze['selected_update']==100 and published['threshold_rawlogit']==0 and published['all2880_preserved']
    summary={s:dict(all_known=metrics[s]['all_known']['matrix'],accepted=metrics[s]['accepted_pairs'],cover_E_recall=metrics[s]['by_variant']['OPAQUE_COVER']['E_recall'],sham_D_recall=metrics[s]['by_variant']['CUE_PRESERVING_SHAM']['D_recall'],all3=metrics[s]['all3_correct'])for s in metrics}
    report=dict(completed=True,PASS=True,blockers=[],baseline_complete=bind(d/'BASELINE_COMPLETE.json'),model_freeze=complete['model_freeze'],checkpoint=complete['checkpoint'],
                scores=complete['scores'],published=complete['results'],identity=identity,independent_metrics=metrics,summary=summary,
                all_reported_metric_cells_and_rates_reconciled=True,current_or_model_optimizer_forward_fit_PnP=0,
                unknown_GT_not_negative=True,source_semantics='Conditional recorded-cue-set AUX and controlled edits; native UNKNOWN/human goldfalse. This baseline is diagnostic and not a new selector.')
    write(HERE/'REVIEW.json',report);write(HERE/'SEAL.json',dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),code=bind(__file__),baseline=complete,
          no_model_optimizer_current_fit_PnP=True))
    print(json.dumps(dict(review=bind(HERE/'REVIEW.json'),summary=summary),indent=2))


if __name__=='__main__':main()
