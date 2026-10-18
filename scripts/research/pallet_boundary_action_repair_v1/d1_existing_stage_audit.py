"""Prediction-only audit for exact reuse of the original fixed-e stage cache.

No scorer, reference, oracle, successful-candidate list, or accuracy metric is
loaded. Reuse is not performed here. Every mismatch remains explicit.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,time
from collections import Counter
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(original_run,typed_run,output):
    started,cpu=time.monotonic(),time.process_time();original=Path(original_run);typed=Path(typed_run)
    seal=read(original/'PREDICTION_SEAL.json');journal=original/'PREDICTIONS.jsonl.gz'
    if sha(journal)!=seal['predictions_sha256']:raise ValueError('Original prediction cache changed')
    records=[];totals=Counter();bindings={str(journal):sha(journal)}
    # slots are individual prediction-only payloads; no score journal is used.
    slotmap={}
    for p in sorted((typed/'slots').glob('*_TYPED_MULTI.json')):
        # This is the original predictor's own replay result, not an oracle.
        r=read(p);slotmap[r['id']]=(p,r)
    for row in rows(journal):
        ident=row['id'];path,replayed=slotmap[ident];replay=replayed['result']
        g10=row['old_global_outline'];g13=row['partial_global_outline']['arms']['FACE_AWARE_PARTIAL_BI_OLD_Q']
        matches={};differences=[]
        for stage,old in [('G10',g10),('G13',g13)]:
            new=replay['stage_results'][stage]
            matches[stage+'_family_selection']=old['selected_family']==new['selected_family']
            for of,nf in zip(old['families'],new['families'],strict=True):
                if of['name']!=nf['name']:differences.append(stage+'/family_order')
                os,ns=of['solution'],nf['solution']
                for key in ('accepted','R','t','R_physical','objective','feasible_pose_samples'):
                    if os.get(key)!=ns.get(key):differences.append(stage+'/'+of['name']+'/'+key)
                oc,nc=os.get('candidates',[]),ns.get('candidates',[])
                if len(oc)!=len(nc):differences.append(stage+'/'+of['name']+'/outcome_count')
                for i,(a,b)in enumerate(zip(oc,nc)):
                    for key in ('initialization','branch','joint_feasible','objective','minimum_constraint_slack','optimizer_success','iterations','error'):
                        if a.get(key)!=b.get(key):differences.append(stage+'/'+of['name']+'/outcome/'+str(i)+'/'+key)
        actual=row['arms']['FROZEN_WD_FACE_AWARE_PARTIAL_BI']['actual_pose'];rp=replay['actual_pose']
        matches['E0_R_t_exact']=all(actual.get(k)==rp.get(k)for k in ('R_cf','R_physical','centroid'))
        complete=not differences and all(matches.values())
        records.append(dict(id=ident,exact=complete,matches=matches,differences=differences,
            original_P1_outcome_count=len(row['full_prior_seed'].get('candidates',[])),
            original_WD0_outcome_count=sum(len(f['solution'].get('candidates',[]))for f in row['full_WD_search']['families']),
            replay_G10_G13_logged_endpoints=len(replay['generated_candidates']),typed_slot_sha256=sha(path)))
        totals['rows']+=1;totals['exact']+=complete;totals['different']+=not complete
        totals['original_P1_outcomes']+=records[-1]['original_P1_outcome_count'];totals['original_WD0_outcomes']+=records[-1]['original_WD0_outcome_count']
        totals['actual_replay_endpoints']+=len(replay['generated_candidates'])
    result=dict(schema='D1_EXISTING_FIXED_E_STAGE_CACHE_AUDIT_V1',status='EXACT'if totals['different']==0 else'NOT_EXACT',
        source_bindings=bindings,original_prediction_seal_sha256=sha(original/'PREDICTION_SEAL.json'),
        statistics=dict(totals),rows=records,reference_or_accuracy_reads=0,pose_reused=False,
        scope='original one fixed P1/WD0/G10/G13 run compared with its exact hierarchical replay endpoint archive; never a union of different research-method banks',
        source_sha256=sha(__file__),wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-run',required=True);p.add_argument('--typed-run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=run(a.original_run,a.typed_run,a.output);print(json.dumps(dict(status=r['status'],statistics=r['statistics'])))
