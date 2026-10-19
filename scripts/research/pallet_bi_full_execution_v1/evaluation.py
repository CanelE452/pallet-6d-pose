"""Separate postseal saved-pose evaluation; never imported by producers.

All original recording clusters, missing outputs and algorithm fallbacks stay
in the binary denominator. Continuous errors are available-only, never zero
imputed. PRIMARY is repeatedly used development data, not an independent test.
"""
from pathlib import Path
from collections import Counter
import argparse
import time
import numpy as np
from scripts.research.pallet_statewise_end_to_end_recovery_v1.evaluator import _safe_metric, _status
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import aggregate, paired
from .s1_union import read, rows, sha, write, save

POPULATIONS={'PRIMARY192':('TEST','REMAINDER'),'TEST49':('TEST',),'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
EXPECTED={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
BASELINE_ARMS=['N0','E0']+['D1_H1_MLP_s'+str(s)for s in (20261018,20261019,20261020)]+['POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR']

def sealed_predictions(folder):
    folder=Path(folder);path=folder/'PREDICTIONS.jsonl.gz';sp=folder/'PREDICTION_SEAL.json';seal=read(sp)
    if seal.get('complete')is not True:raise ValueError('Incomplete prediction population')
    if sha(path)not in [seal.get('predictions_sha256'),seal.get('prediction_sha256')]+list(seal.get('artifacts',{}).values()):raise ValueError('Unsealed prediction bytes')
    for k in ('reference_reads','reference_inputs','real_reference_inputs','GT_inputs'):
        if k in seal and seal[k]not in (False,0):raise ValueError('Prediction used numeric references')
    ledger=list(rows(path))
    if len(ledger)!=238 or len({r['id']for r in ledger})!=238:raise ValueError('Full238 outputs required before scoring')
    return ledger,dict(path=str(path),sha256=sha(path),seal_sha256=sha(sp))

def evaluate(d1,reference_dir,output,additional=()):
    start=time.monotonic();d1,reference_dir,output=map(Path,(d1,reference_dir,output))
    base,binding=sealed_predictions(d1)
    merged={r['id']:dict(r,predictions={a:r['predictions'][a]for a in BASELINE_ARMS})for r in base}
    bindings=[binding];arms=list(BASELINE_ARMS)
    for folder in additional:
        new,b=sealed_predictions(folder);bindings.append(b)
        if {r['id']for r in new}!=set(merged):raise ValueError('New prediction IDs differ')
        names=list(new[0]['predictions'])
        if set(names)&set(arms):raise ValueError('Ambiguous arm name')
        arms+=names
        for r in new:
            if set(r['predictions'])!=set(names):raise ValueError('Partial arm population')
            if any(r.get(k)!=merged[r['id']].get(k)for k in ('partition','session','cluster_id')):raise ValueError('Identity metadata changed')
            merged[r['id']]['predictions'].update(r['predictions'])
    rp=reference_dir/'FIXED_REFERENCES.json';ss=read(reference_dir/'SCORE_SEAL.json')
    if sha(rp)!=ss['artifacts'][rp.name]:raise ValueError('Canonical reference bytes changed')
    output.mkdir(parents=True,exist_ok=True)
    write(output/'REFERENCE_ACCESS.json',dict(purpose='postseal saved-pose evaluation only',prediction_bindings=bindings,
        reference_sha256=sha(rp),scorer_sha256=sha(__file__),production_input=False))
    refs=read(rp)['references'];scored=[]
    for r in base:
        row=merged[r['id']];ref=refs[row['partition']][row['id']]
        if ref['order']!=2:raise ValueError('Original real C2 required')
        record={k:row[k]for k in ('id','partition','session','cluster_id')}
        record.update(poses={},metrics={},status={})
        for a in arms:
            p=row['predictions'][a];record['poses'][a]=p['actual_pose'];record['metrics'][a]=_safe_metric(p['actual_pose'],ref);record['status'][a]=_status(p)
        scored.append(record)
    subsets={p:[r for r in scored if r['partition']in parts]for p,parts in POPULATIONS.items()}
    if {p:len(v)for p,v in subsets.items()}!=EXPECTED:raise ValueError('Full reporting populations required')
    summary={p:aggregate(v,arms)for p,v in subsets.items()}
    contrasts=[]
    for a in arms:
        if a not in BASELINE_ARMS:
            seed=a.rsplit('_s',1)[-1]
            before='D1_H1_MLP_s'+seed
            contrasts.append((before if before in arms else 'D1_H1_MLP_s20261018',a))
    pp={p:{a+'_minus_'+b:paired(v,b,a)for b,a in contrasts}for p,v in subsets.items()}
    save(output/'SCORED_ROWS.jsonl.gz',scored);write(output/'SUMMARY.json',summary);write(output/'PAIRED.json',pp)
    seal=dict(complete=True,status='COMPLETE_SAVED_POSE_SCORING',populations=EXPECTED,arms=arms,reference_sha256=sha(rp),
        prediction_bindings=bindings,artifacts={name:sha(output/name)for name in ('SCORED_ROWS.jsonl.gz','SUMMARY.json','PAIRED.json')},
        no_pose='full-denominator binary failure, available-only continuous errors',tail_trimming=0,
        independent_real_test_n=0,wall_seconds=time.monotonic()-start)
    write(output/'EVALUATION_SEAL.json',seal)
    return summary

def fixed_sets(d1,reference_dir,output,baseline_evaluation):
    """Independent whole saved-bank scoring, not selector performance."""
    d1,reference_dir,output,baseline_evaluation=map(Path,(d1,reference_dir,output,baseline_evaluation))
    if not read(baseline_evaluation/'EVALUATION_SEAL.json')['complete']:raise ValueError('Baseline must be sealed/scored first')
    refs=read(reference_dir/'FIXED_REFERENCES.json')['references']
    base={r['id']:r for r in rows(baseline_evaluation/'SCORED_ROWS.jsonl.gz')}
    feats=list(rows(d1/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz'));ledger=[]
    for index,f in enumerate(feats):
        r=base[f['id']];ref=refs[r['partition']][r['id']]
        raw=next(rows(d1/'d1_real_candidates_v1/frames'/f'{index:04d}'/'E_RUN.jsonl.gz'))
        generated=any(_safe_metric(dict(available=True,R_physical=c['R_physical'],centroid=c['centroid']),ref)['success']for c in raw['candidates'])
        eligible=any(c.get('eligible')and _safe_metric(c['actual_pose'],ref)['success']for c in f['candidates'])
        pf=any(r['metrics'][a]['success']for a in ('POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR'))
        ledger.append(dict(id=r['id'],partition=r['partition'],cluster_id=r['cluster_id'],generated_good=generated,eligible_good=eligible,
            selected_good=r['metrics']['D1_H1_MLP_s20261018']['success'],PF_good=pf,union_good=generated or pf))
    primary=[r for r in ledger if r['partition']!='CAL']
    sets=dict(original31_selection_failure=[r['id']for r in primary if r['generated_good']and not r['selected_good']],
        original41_candidate_absent=[r['id']for r in primary if not r['generated_good']],
        PF_union38_candidate_absent=[r['id']for r in primary if not r['union_good']],
        PF_new_good=[r['id']for r in primary if r['PF_good']and not r['generated_good']],
        PF_good_old_bank_good_H1_fail=[r['id']for r in primary if r['PF_good']and r['generated_good']and not r['selected_good']])
    result=dict(schema='FULL_CAMPAIGN_FIXED_FAILURE_SETS_V1',sets=sets,counts={k:len(v)for k,v in sets.items()},
        generated_good=sum(r['generated_good']for r in primary),eligible_good=sum(r['eligible_good']for r in primary),
        unadmitted_PF_union_good=sum(r['union_good']for r in primary),denominator=192,reference_sha256=sha(reference_dir/'FIXED_REFERENCES.json'),
        scope='independently rescored all stored actual endpoints, old definition fixed before new selector outcomes; not automatic selector upper bound')
    save(output/'BANK_COVERAGE_ROWS.jsonl.gz',ledger);write(output/'FIXED_SETS.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['score','fixed-sets'])
    for n in ('d1','reference-dir','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--additional',type=Path,nargs='*',default=[]);p.add_argument('--baseline-evaluation',type=Path)
    a=p.parse_args()
    if a.stage=='score':
        result=evaluate(a.d1,a.reference_dir,a.output,a.additional);print({p:{arm:v['successes']for arm,v in s.items()}for p,s in result.items()})
    else:print(fixed_sets(a.d1,a.reference_dir,a.output,a.baseline_evaluation)['counts'])
