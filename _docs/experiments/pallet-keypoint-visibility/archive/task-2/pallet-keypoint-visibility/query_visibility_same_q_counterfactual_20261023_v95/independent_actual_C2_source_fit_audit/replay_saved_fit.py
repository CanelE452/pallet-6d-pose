"""Independent saved-only C2 audit; no trainer, model or optimizer import."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;V=HERE.parent
RUN=V/'runs/C2_SAME_Q_COUNTERFACTUAL_SOURCE_V1'
VARIANTS=('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def verify(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def canonical(source,s):return {('LOCO','TRAIN'):'TRAIN',('LOCO','VALIDATION'):'VALIDATION',('PALLOX','TRAIN_LAB'):'TRAIN',('PALLOX','VAL_HALL'):'VALIDATION'}[source,s]
def decode(z,a):return np.where(a,(z>=0).astype(np.int8),-1)
def stratum_masks(label,split,variant):
    return {name:(split=='VALIDATION')&(variant==v)&(label==y)for name,v,y in
        [('BEFORE_D','BEFORE',0),('BEFORE_E','BEFORE',1),('SHAM_D',VARIANTS[2],0),('COVER_E',VARIANTS[1],1)]}
def bce(z,label):return np.logaddexp(0.,np.asarray(z,np.float64))-np.asarray(label)*np.asarray(z,np.float64)
def metric(pred,label,mask):
    known=label>=0;cm=np.array([[int(np.sum(mask&(label==y)&(pred==p)))for p in [0,1,-1]]for y in [0,1]])
    ratio=lambda a,b:float(a/b)if b else None
    return dict(matrix=cm.tolist(),rows=['D','E'],columns=['D','E','U'],D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),
        D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),known_U_rate=ratio(cm[:,2].sum(),cm.sum()),
        all_N=int(mask.sum()),known_N=int(cm.sum()),unknown_GT_N=int(np.sum(mask&~known)),unknown_GT_predictions={str(p):int(np.sum(mask&~known&(pred==p)))for p in [0,1,-1]})
def metrics_and_gates(z,a,label,split,variant,accepted):
    pred=decode(z,a);all3=(pred.reshape(960,3)==[0,1,0]).all(1);out={}
    for s in ['TRAIN','VALIDATION']:
        m=split==s;pm=accepted&m[::3];n=int(pm.sum())
        out[s]=dict(all_known=metric(pred,label,m),by_variant={v:metric(pred,label,m&(variant==v))for v in VARIANTS},accepted_pairs=n,
            all3_correct=int(np.sum(all3&pm)),all3_correct_rate=float(np.sum(all3&pm)/n)if n else None)
    val=out['VALIDATION'];before=val['by_variant']['BEFORE']
    specs=[('natural_before_D_recall',before['D_recall'],.85,False),('natural_before_E_recall',before['E_recall'],.75,False),
        ('sham_D_recall',val['by_variant'][VARIANTS[2]]['D_recall'],.85,False),('cover_E_recall',val['by_variant'][VARIANTS[1]]['E_recall'],.75,False),
        ('accepted_VAL_pair_all3_correct_rate',val['all3_correct_rate'],.75,False),('VAL_all_known_U_rate',val['all_known']['known_U_rate'],.05,True)]
    gates={name:dict(value=value,**({'maximum':limit}if upper else{'minimum':limit}),PASS=value is not None and(value<=limit if upper else value>=limit))for name,value,limit,upper in specs}
    return out,gates

def main(fit_sha):
    # Exact root completion authorization precedes every run score/GT packet read.
    assert len(fit_sha)==64 and sha(RUN/'FIT_COMPLETE.json')==fit_sha
    complete=read(RUN/'FIT_COMPLETE.json');assert complete['completed']and complete['PASS_execution']and complete['optimizer_updates']==2000
    for name in ['MODEL_FREEZE','source_results','source_gates','scores','history','train_updates']:verify(complete[name])
    freeze=read(complete['MODEL_FREEZE']['path']);entry=read(verify(freeze['entry']))
    assert entry['approved_actual_source_fit']is True and entry['source_only']is True and entry['current_allowed']is False and entry['PnP_allowed']is False
    assert entry['allow_restart_after_optimizer_update']is False and freeze['current_inference_authorized']is False
    for key in ['trainer','protocol','model_code','candidate_definition','input_pack_manifest','input_pack_seal','base_input_lineage','GT_ledger','generated_tests','independent_preparation_review','independent_input_audit']:verify(entry[key])
    protocol=read(entry['protocol']['path']);manifest=read(entry['input_pack_manifest']['path'])
    assert manifest['base_rows']==960 and manifest['occurrences']==2880
    lineage=rows(verify(manifest['input_lineage']));gt=rows(entry['GT_ledger']['path'])
    assert len(lineage)==len(gt)==2880
    label=np.array([x['candidate_aux_hidden_label_GT_only']for x in gt],np.int8)
    known=label>=0;source=np.array([x['source']for x in lineage]);rawsplit=np.array([x['split']for x in lineage])
    split=np.array([canonical(x['source'],x['split'])for x in lineage]);variant=np.array([x['variant']for x in lineage])
    factual=np.array([x['input_facts']['operator_query_available']for x in lineage],bool)
    ready=np.array([x['pair_ready']for x in gt],bool).reshape(960,3)
    assert (ready==ready[:,:1]).all();accepted=ready[:,0]
    for i,(r,g)in enumerate(zip(lineage,gt)):
        assert r['occurrence_index']==g['occurrence_index']==i and r['slot_index']==g['base_row_index']==g['slot_index']==i//3
        assert r['occurrence_id']==g['occurrence_id'] and r['variant']==g['variant']==VARIANTS[i%3]
        assert g['known_candidate_GT_only']==known[i]
        if i%3==0:assert label[i]==g['original_natural_label_GT_only']
    for i in range(960):
        assert label[3*i:3*i+3].tolist()==([0,1,0]if accepted[i]else[gt[3*i]['original_natural_label_GT_only'],-1,-1])
    with np.load(complete['scores']['path'],allow_pickle=False)as packet:score={k:np.array(packet[k],copy=True)for k in packet.files}
    z=score['query_logit'];a=score['query_available'];assert z.shape==a.shape==(2880,)and z.dtype==np.float32 and a.dtype==bool and np.isfinite(z).all()
    assert np.array_equal(a,factual)and(z[~a]==0).all()
    for key,value in [('candidate_known',known),('scalar_target_D0_E1_Uminus1',label),('source',source),('split',split),('original_split',rawsplit),('variant',variant),
        ('occurrence_index',np.arange(2880)),('base_row_index',np.arange(2880)//3),('accepted_pair',accepted),('prediction',decode(z,a))]:assert np.array_equal(score[key],value),key
    for key in ['occurrence_id','input_id','native_slot_logical_only','image_sha256','original_image_sha256']:
        assert np.array_equal(score[key],[r[key]for r in lineage]),key
    q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage],np.float64)
    L=np.array([r['L']if r['L']is not None else np.nan for r in lineage],np.float64)
    assert np.array_equal(q,score['query_xy_raw'],equal_nan=True)and np.array_equal(L,score['L_raw'],equal_nan=True)
    strata=stratum_masks(label,split,variant);loss=bce(z,label)
    selected_means={k:dict(count=int(m.sum()),mean_BCE=float(loss[m].mean()))for k,m in strata.items()}
    selected_value=float(np.mean([x['mean_BCE']for x in selected_means.values()]))
    history=rows(complete['history']['path']);assert [h['update']for h in history]==list(range(0,2001,100))
    best=float('inf');selected=None;history_aggregation_error=0.;unavailable_count=int(np.sum(known&(split=='VALIDATION')&~a))
    for h in history:
        assert set(h['VAL_strata'])==set(strata) and h['known_unavailable']==unavailable_count
        for name,m in strata.items():
            assert h['VAL_strata'][name]['count']==int(m.sum())
            assert h['VAL_strata'][name]['mean_BCE']>=int(np.sum(m&~a))*math.log(2)/int(m.sum())-1e-12
        value=float(np.mean([h['VAL_strata'][k]['mean_BCE']for k in strata]))
        history_aggregation_error=max(history_aggregation_error,abs(value-h['criterion_four_VAL_strata_BCE']))
        assert abs(value-h['criterion_four_VAL_strata_BCE'])<=1e-12
        if h['criterion_four_VAL_strata_BCE']<best:best=h['criterion_four_VAL_strata_BCE'];selected=h['update']
    assert selected==complete['selected_update']==freeze['selected_update']
    selected_h=next(h for h in history if h['update']==selected)
    selected_history_error=abs(selected_value-best)
    assert selected_history_error<=1e-12
    for name in strata:assert abs(selected_means[name]['mean_BCE']-selected_h['VAL_strata'][name]['mean_BCE'])<=1e-12
    updates=rows(complete['train_updates']['path']);assert len(updates)==2000
    generator=torch.Generator(device='cpu').manual_seed(194)
    pools=[np.flatnonzero((split=='TRAIN')&known&(label==k))for k in [0,1]]
    for u,event in enumerate(updates,1):
        assert event['update']==u
        expected=torch.cat([torch.as_tensor(p)[torch.randint(len(p),(32,),generator=generator)]for p in pools])
        expected=expected[torch.randperm(64,generator=generator)].numpy()
        ix=np.array(event['occurrence_indices']);assert np.array_equal(ix,expected)
        assert ix.shape==(64,) and known[ix].all()and(split[ix]=='TRAIN').all()
        assert event['batch_D']==event['batch_E']==32
        assert event['available_known_count']==int(a[ix].sum())and event['known_unavailable_count']==int((~a[ix]).sum())
        assert np.isfinite(event['loss'])and np.isfinite(event['gradient_norm_before_clip'])
    own_metrics,own_gates=metrics_and_gates(z,a,label,split,variant,accepted)
    published=read(complete['source_results']['path']);gates=read(complete['source_gates']['path'])
    assert published['metrics']==own_metrics and gates['gates']==own_gates
    assert gates['all_gates_pass']==gates['PASS']==all(g['PASS']for g in own_gates.values()) and gates['current_inference_authorized']is False
    counts={}
    for s in ['TRAIN','VALIDATION']:
        m=split==s;b=m&(variant=='BEFORE')
        counts[s]=dict(before_D=int(np.sum(b&(label==0))),before_E=int(np.sum(b&(label==1))),before_U=int(np.sum(b&(label<0))),
            accepted_pairs=int(np.sum(m[::3]&accepted)),known_D=int(np.sum(m&(label==0))),known_E=int(np.sum(m&(label==1))),unknown=int(np.sum(m&(label<0))))
    assert counts==protocol['expected_counts']==published['counts']
    assert published['VAL_strata']==selected_means and published['selector_four_VAL_strata_BCE']==selected_value
    for key in ['checkpoint','initial_checkpoint','final_checkpoint']:verify(freeze[key])
    fp64_diagnostic=bce(z,np.maximum(label,0));nll_error=float(np.max(np.abs(score['NLL_FP32'].astype(np.float64)-fp64_diagnostic)))
    assert nll_error<=1e-6
    result=dict(completed=True,PASS=True,blockers=[],policy=bind(HERE/'AUDIT_POLICY_BEFORE_FIT_COMPLETE.json'),FIT_COMPLETE=bind(RUN/'FIT_COMPLETE.json'),
        trainer=entry['trainer'],model_code=entry['model_code'],protocol=entry['protocol'],all2880_ID_qL_image_order_GT_rawsplit_sign_U_exact=True,
        counts=counts,source_metrics=own_metrics,gates=own_gates,all_source_gates_pass=all(g['PASS']for g in own_gates.values()),
        selected_update=selected,selector_four_VAL_strata_BCE=selected_value,selected_strata=selected_means,
        history_rows=21,history_recorded_four_mean_aggregation_max_abs_difference=history_aggregation_error,
        selected_saved_scores_vs_history_BCE_abs_difference=selected_history_error,all2000x64_seed194_TRAIN_known_D32_E32_indices_exact=True,
        NLL_FP32_vs_independent_FP64_max_abs_difference=nll_error,NLL_unknown_rows_diagnostic_only_not_aggregated=True,
        saved_unavailable_count=int((~a).sum()),known_unavailable_count=int(np.sum(known&~a)),unknown_GT_count=int(np.sum(~known)),
        limitations=['Only selected per-query logits are stored. Other20 history stratum means are checked for count/aggregation/selection consistency, not recomputed from absent per-query logits.','Checkpoint bytes are hashed, not tensor-deserialized; no model or optimizer is instantiated. Saved logs cannot prove absence of an undisclosed external run.','Conditional AI cue-known-set/controlled augmentation; native/human semantic gold UNKNOWN.','No gate result automatically authorizes current evaluation.'],
        activity=dict(source_feature_RGB_mask_reads=0,checkpoint_tensor_load=0,model_optimizer_construction=0,new_forward_fit_current_PnP=0))
    with (HERE/'REVIEW.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with (HERE/'SEAL.json').open('x')as f:json.dump(dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),reader=bind(__file__),policy=bind(HERE/'AUDIT_POLICY_BEFORE_FIT_COMPLETE.json')),f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit-complete-sha',required=True)
    main(p.parse_args().fit_complete_sha)
