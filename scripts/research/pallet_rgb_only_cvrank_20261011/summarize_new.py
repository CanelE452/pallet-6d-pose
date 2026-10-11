"""Reuse verified statistics; new family of three, all fifteen common routes."""
import argparse, csv, gzip, hashlib, json, math, shutil, statistics, time
from collections import Counter
from pathlib import Path
import numpy as np
import rgb_model as R
import stats_report as S

METHODS=('BASE','SUBPIX','CVRANK_NATIVE','CVRANK_CAP1','CVRANK_THEN_SUBPIX',
 'OLD_P','OLD_P_SUBPIX','M1','M1_SUBPIX','RGB_ONLY','RGB_ONLY_SUBPIX','M3','M3_SUBPIX','POSEFIX','POSEFIX_SUBPIX')
PRIMARY=[('M3_SUBPIX','RGB_ONLY_SUBPIX'),('M3_SUBPIX','M1_SUBPIX'),('M3_SUBPIX','CVRANK_THEN_SUBPIX')]
DESCRIPTIVE=[('RGB_ONLY_SUBPIX','M1_SUBPIX'),('RGB_ONLY_SUBPIX','CVRANK_THEN_SUBPIX'),('M1_SUBPIX','CVRANK_THEN_SUBPIX'),
 ('RGB_ONLY','M1'),('M3','RGB_ONLY'),('M3','CVRANK_CAP1'),('RGB_ONLY','CVRANK_CAP1')]
WITHIN=[(m+'_SUBPIX',m) for m in ('OLD_P','M1','RGB_ONLY','M3','POSEFIX')]+[('CVRANK_THEN_SUBPIX','CVRANK_NATIVE')]
PAIRS=PRIMARY+DESCRIPTIVE+WITHIN

def run(args):
    start=time.monotonic();doc,old=Path(args.public_dir),Path(args.prior_public_dir)
    rows=S.load_rows([doc/'ROW_METRICS.jsonl.gz']);assert len(rows)==3300
    index={(r['method'],r['id'],r['seed']):r for r in rows};ids=sorted({r['id'] for r in rows})
    assert len(ids)==110 and {r['method'] for r in rows}==set(METHODS)
    metadata={fid:next(r for r in rows if r['id']==fid) for fid in ids}
    sessions=[metadata[fid]['session'] for fid in ids];names=sorted(set(sessions));assert len(names)==9
    prior_lock=S.read(old/'BOOTSTRAP_LOCK.json');assert prior_lock['sessions']==names
    prior_metrics=S.read(old/'METRICS.json');assert prior_metrics['frames']==110
    draw=np.load(old/'BOOTSTRAP_DRAWS.npy',allow_pickle=False)
    assert draw.shape==(10000,9) and np.array_equal(draw,np.random.default_rng(20260917).integers(0,9,size=(10000,9)))
    shutil.copyfile(old/'BOOTSTRAP_DRAWS.npy',doc/'BOOTSTRAP_DRAWS.npy')
    assert S.sha(doc/'BOOTSTRAP_DRAWS.npy')==prior_lock['sha256']
    boot=S.Bootstrap(ids,sessions,draw)
    lock=dict(**prior_lock,source='Exact prior draw reuse; same sorted IDs and session names',source_prior_sha256=S.sha(old/'BOOTSTRAP_LOCK.json'),
        family_size=3,family=PRIMARY,ID_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
    S.write(doc/'BOOTSTRAP_LOCK.json',lock)
    groups={m:dict(common_seed12=S.summarize(index,m,ids,(1,2),boot)) for m in METHODS}
    for packet in groups.values():
        axis=packet['common_seed12']['axis_wrong'];axis.update(reference_eligible_frames=109,reference_eligible_prediction_rows=218,
            excluded_reference_UNCLEAR_frames=1,missing_selected_hyp_rows=218-axis['denominator'])
    metrics=dict(status='COMPLETE',frames=110,sessions=9,grades=dict(Counter(metadata[fid]['grade'].lower() for fid in ids)),
        common_seeds=[1,2],prediction_rows=3300,groups=groups,bootstrap=lock,
        reference='manual 2D corner geometric reconstruction; exposed non-Severe DEV',
        prior_reference_routes='../pallet_nonsevere_precision_20261011/ROW_METRICS.jsonl.gz',
        old_N3_M0_M2_M4_preserved_in_prior=True,new_fits='RGB_ONLY two seeds only')
    primary=S.family(PRIMARY,index,ids,boot,{})
    for row in primary:row['exposure']='Existing observed M3−M1 contrast reused' if row['before']=='M1_SUBPIX' else 'New contrast on previously exposed DEV'
    paired=dict(status='COMPLETE',primary=primary,primary_declared_family_size=3,mechanism=[],
        bootstrap=lock,other_comparisons_descriptive=True,non_significance_is_not_equivalence=True)
    S.PRIMARY=PRIMARY;S.MECHANISM=DESCRIPTIVE+WITHIN
    transitions=S.all_transitions(index,ids);assert len(transitions)==len(PAIRS)*220
    for row in transitions:
        before,after=row['before_corner_mean_px'],row['after_corner_mean_px']
        row['corner_better_pose_worse']=bool(before is not None and after is not None and after<before-1e-8 and
            ((row['after_Tcm'] is not None and row['before_Tcm'] is not None and row['after_Tcm']>row['before_Tcm']+1e-8) or
             (row['after_Rdeg'] is not None and row['before_Rdeg'] is not None and row['after_Rdeg']>row['before_Rdeg']+1e-8)))
        row['availability_transition']=row['before_available']!=row['after_available']
    changed=[r for r in transitions if r['event']!='UNCHANGED' or r['corner_better_pose_worse'] or r['availability_transition']]
    S.csv_table(doc/'ALL_PAIRED_TRANSITIONS.csv',list(transitions[0]),transitions)
    S.csv_table(doc/'CHANGED_CASES.csv',list(transitions[0]),changed)
    pair_summary=[]
    for after,before in PAIRS:
        rr=[r for r in transitions if r['after']==after and r['before']==before];events=Counter(r['event'] for r in rr)
        a,b=groups[after]['common_seed12'],groups[before]['common_seed12']
        pair_summary.append(dict(after=after,before=before,rows=220,recovery=events['RECOVERY'],damage=events['DAMAGE'],
            unchanged=events['UNCHANGED'],corner_better_pose_worse=sum(r['corner_better_pose_worse'] for r in rr),
            corners_damaged=sum(r['corners_damaged'] for r in rr),corners_improved=sum(r['corners_improved'] for r in rr),
            net_success_difference=a['strict_success']['successes']-b['strict_success']['successes'],
            descriptive= (after,before) not in PRIMARY,
            per_seed={str(s):dict(Counter(r['event'] for r in rr if r['seed']==s)) for s in (1,2)}))
    S.write(doc/'PAIRED_TRANSITION_SUMMARY.json',dict(status='COMPLETE',pairs=pair_summary,all_pairs=PAIRS,
        all_rows=3520,changed_or_corner_pose_tradeoff_rows=len(changed),no_unfavorable_case_deleted=True))
    S.write(doc/'METRICS.json',metrics);S.write(doc/'PAIRED_RESULTS.json',paired)
    S.write(doc/'FAILURES.json',dict(status='COMPLETE',unavailable={m:{str(s):[fid for fid in ids if not index[m,fid,s]['available']] for s in (1,2)} for m in METHODS},
        excluded_new_frames=[],reference_axis_UNCLEAR_only=['eval_outside:1778653367706938112']))
    table=[]
    for m in METHODS:
        p=groups[m]['common_seed12'];r=dict(method=m,frames=110,seeds='1,2',success=p['strict_success']['successes'],denominator=220,
            success_percent=p['strict_success']['rate']*100,pose_unavailable=220-p['pose_availability']['available'],
            axis_wrong=p['axis_wrong']['errors'],axis_denominator=p['axis_wrong']['denominator'],legacy_WD_percent=p['legacy_WD_confusion']['rate']*100)
        for key in S.METRICS:
            for stat in ('mean','variance','SD','median','P90','P95'):r[key+'_'+stat]=p['metrics'][key][stat]
            r[key+'_mean_CI95']=json.dumps(p['metrics'][key]['mean_CI95'])
        for t in S.THRESHOLDS:r[f'PCK_{t}px_percent']=p['PCK'][str(t)]['rate']*100
        table.append(r)
    S.csv_table(doc/'ACCURACY_TABLE.csv',list(table[0]),table)
    S.write(doc/'STATISTICS_RECEIPT.json',dict(status='COMPLETE',frames=110,rows=3300,groups=15,all_pair_rows=3520,
        changed_success_rows=sum(r['event']!='UNCHANGED' for r in transitions),changed_or_tradeoff_rows=len(changed),
        source_sha256=S.sha(__file__),reused_statistical_source_sha256=S.sha(Path(S.__file__)),
        source_rows_sha256=S.sha(doc/'ROW_METRICS.jsonl.gz'),wall_seconds=time.monotonic()-start))
    print('RGB_ONLY_NEW_FAMILY_SUMMARIZED',primary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-dir',required=True);p.add_argument('--prior-public-dir',required=True);run(p.parse_args())
