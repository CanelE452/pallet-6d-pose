"""Saved H1 action-rank diagnosis after full bank coverage scoring.

No model forward, fitting, pose production, or new reference decoding. Logits
are compared only within the same image/arm/seed, never as calibrated beliefs.
"""
from pathlib import Path
from collections import Counter
import argparse
import math
from . import s1_union as U
from .evaluation import sealed_predictions, POPULATIONS, EXPECTED

SEEDS=(20261018,20261019,20261020)


def diagnose(row,prediction,seed):
    good=set(row['good_eligible_ids']);actions=prediction['actions']
    flags=[a['candidate_id']in good or bool(good.intersection(a.get('source_aliases',[])))for a in actions]
    good_indices=[i for i,v in enumerate(flags)if v]
    if any(not math.isfinite(a['score'])for a in actions):raise ValueError('Saved finite action logits required')
    if actions:
        chosen=min(range(len(actions)),key=lambda i:(-actions[i]['score'],i))
        if prediction['selected_candidate_id']!=actions[chosen]['candidate_id']or abs(prediction['selected_score']-actions[chosen]['score'])>1e-6:
            raise ValueError('Saved selected action does not reproduce original finite-logit argmax/tie order')
    selected=row['selected'][str(seed)];success=selected['metric']['success']
    detail='SUCCESS'if success else'GENERATED_ABSENT'if not row['generated_good']else'ADMISSION_LOSS'if not row['eligible_good']else'ACTION_CONSTRUCTION_GAP'if not good_indices else'WRONG_ACTION_RANK'
    if not actions and row['eligible_good']:detail='NO_AVAILABLE_ACTION_WITH_ELIGIBLE_GOOD_BANK'
    best=None;rank=None;margin=None
    if good_indices:
        best=min(good_indices,key=lambda i:(-actions[i]['score'],i))
        rank=1+sum(a['score']>actions[best]['score']or (a['score']==actions[best]['score']and i<best)for i,a in enumerate(actions))
        if prediction.get('selected_score')is not None:margin=actions[best]['score']-prediction['selected_score']
    top=sorted(range(len(actions)),key=lambda i:(-actions[i]['score'],i))[:5]
    return dict(arm=row['arm'],id=row['id'],partition=row['partition'],session=row['session'],cluster_id=row['cluster_id'],seed=seed,
        success=success,failure_detail=detail,eligible_good=row['eligible_good'],good_actions=len(good_indices),action_count=len(actions),
        best_good_action_id=actions[best]['candidate_id']if best is not None else None,best_good_action_rank=rank,
        best_good_logit_minus_selected=margin,selected_candidate_id=prediction.get('selected_candidate_id'),
        selected_score=prediction.get('selected_score'),top_actions=[dict(actions[i],posthoc_actual_success=flags[i])for i in top],
        logit_is_calibrated_confidence=False,rank_scope='within same image/arm/seed; posthoc reference labels from previously sealed bank analysis')


def run(manifest,bank_analysis,fixed_sets,output):
    manifest,bank_analysis,fixed_sets,output=map(Path,(manifest,bank_analysis,fixed_sets,output))
    plan=U.read(manifest);seal=U.read(bank_analysis/'ANALYSIS_SEAL.json')
    if not seal['complete']or seal['manifest_sha256']!=U.sha(manifest):raise ValueError('Completed matching bank analysis required')
    for name,digest in seal['artifacts'].items():
        if U.sha(bank_analysis/name)!=digest:raise ValueError('Scored bank coverage changed')
    if U.sha(fixed_sets)!=seal['fixed_sets_sha256']:raise ValueError('Fixed failure-cohort definition changed')
    input_bindings=U.read(bank_analysis/'INPUT_BINDINGS.json')
    allrows=list(U.rows(bank_analysis/'BANK_COVERAGE_ROWS.jsonl.gz'));result=[];bindings={}
    for arm,spec in plan['arms'].items():
        pp,binding=sealed_predictions(spec['prediction_dir']);bindings[arm]=binding;byid={p['id']:p for p in pp}
        if binding!=input_bindings[arm]['prediction']:raise ValueError('Saved prediction is not the one used for bank coverage labels')
        rows=[r for r in allrows if r['arm']==arm]
        if len(rows)!=238 or {r['id']for r in rows}!=set(byid):raise ValueError('Full arm population required')
        for row in rows:
            if any(row[key]!=byid[row['id']][key]for key in ('partition','session','cluster_id')):raise ValueError('Scored/action metadata identity differs')
            for seed in SEEDS:result.append(diagnose(row,byid[row['id']]['predictions'][arm+'_s'+str(seed)],seed))
    sets=U.read(fixed_sets)['sets'];summary={};cohorts={}
    for arm in plan['arms']:
        summary[arm]={};cohorts[arm]={}
        for pop,parts in POPULATIONS.items():
            summary[arm][pop]={}
            for seed in SEEDS:
                rows=[r for r in result if r['arm']==arm and r['seed']==seed and r['partition']in parts]
                if len(rows)!=EXPECTED[pop]:raise ValueError('Full denominator lost')
                summary[arm][pop][str(seed)]=dict(n=len(rows),counts=dict(Counter(r['failure_detail']for r in rows)),
                    missed_good_action_ranks=[dict(id=r['id'],rank=r['best_good_action_rank'],margin=r['best_good_logit_minus_selected'])for r in rows if not r['success']and r['eligible_good']])
        for name,ids in sets.items():cohorts[arm][name]=[r for r in result if r['arm']==arm and r['id']in ids]
    output.mkdir(parents=True,exist_ok=True);U.save(output/'ACTION_RANK_ROWS.jsonl.gz',result)
    U.write(output/'SUMMARY.json',summary);U.write(output/'FIXED_COHORT_ACTIONS.json',cohorts)
    receipt=dict(complete=True,source_sha256=U.sha(__file__),bank_analysis_seal_sha256=U.sha(bank_analysis/'ANALYSIS_SEAL.json'),
        manifest_sha256=U.sha(manifest),fixed_sets_sha256=U.sha(fixed_sets),prediction_bindings=bindings,
        new_model_forwards=0,new_fits=0,new_pose_calls=0,new_reference_decodes=0,all_denominators_preserved=True,
        artifacts={n:U.sha(output/n)for n in ('ACTION_RANK_ROWS.jsonl.gz','SUMMARY.json','FIXED_COHORT_ACTIONS.json')})
    U.write(output/'SELECTION_DIAGNOSTIC_SEAL.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('manifest','bank-analysis','fixed-sets','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(run(a.manifest,a.bank_analysis,a.fixed_sets,a.output))
