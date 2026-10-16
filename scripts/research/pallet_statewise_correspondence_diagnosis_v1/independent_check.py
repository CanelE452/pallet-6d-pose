"""Postseal second aggregation using rotation vectors instead of trace/acos.

This module imports no primary scoring helper and performs no inference.
Reference coordinates are the hash-bound fixed references emitted by the
separate primary scorer, not fitted against these predictions.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .common import read, rows, sha, write, record_stage


def independent_error(pose, ref):
    if not pose['available']:
        return None,None,False
    distance=100*math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(pose['centroid'],ref['t'])))
    if ref['R'] is None:return distance,None,None
    observed=np.asarray(pose['R_physical'],float)
    target=np.asarray(ref['R'],float)
    # Explicit physical Y 180deg, no dimension exchange or owner reassignment.
    equivalent=(target,target@np.diag([-1.,1.,-1.]))
    angles=[float(np.linalg.norm(Rotation.from_matrix(r.T@observed).as_rotvec())*180/np.pi)
            for r in equivalent]
    angle=min(angles)
    return distance,angle,distance<=5. and angle<=5.


def run(folder,score_folder):
    root=Path(folder);scored=Path(score_folder);start=time.monotonic();cpu=time.process_time()
    seal=read(scored/'SCORE_SEAL.json')
    for name in ('BASELINE_PREDICTIONS.jsonl.gz','ROW_METRICS.csv','FIXED_REFERENCES.json'):
        if sha(scored/name)!=seal['artifacts'][name]:raise ValueError('Score artifact changed: '+name)
    references=read(scored/'FIXED_REFERENCES.json')['references']
    baseline={r['id']:r for r in rows(scored/'BASELINE_PREDICTIONS.jsonl.gz')}
    fresh={r['id']:r for r in rows(root/'REPLAY_PREDICTIONS.jsonl.gz')}
    table=list(csv.DictReader((scored/'ROW_METRICS.csv').open()))
    bykey={(r['id'],r['arm']):r for r in table}
    assert len(fresh)==len(baseline)==238
    errors=[];counts=defaultdict(Counter);values=defaultdict(list)
    for fid,b in baseline.items():
        predictions=dict(b['predictions']);predictions.update(fresh[fid]['predictions'])
        ref=references[b['partition']][fid]
        for arm,p in predictions.items():
            distance,angle,success=independent_error(p['actual_pose'],ref)
            expected=bykey[(fid,arm)]
            d0=float(expected['translation_cm']) if expected['translation_cm'] else None
            a0=float(expected['rotation_deg']) if expected['rotation_deg'] else None
            de=abs(distance-d0) if distance is not None and d0 is not None else 0.
            ae=abs(angle-a0) if angle is not None and a0 is not None else 0.
            if (distance is None)!=(d0 is None) or (angle is None)!=(a0 is None):
                raise AssertionError('Availability mismatch '+fid+' '+arm)
            expected_success={'True':True,'False':False,'':None}[expected['success_le5']]
            if de>1e-9 or ae>1e-6 or success!=expected_success:
                errors.append(dict(id=fid,arm=arm,translation_difference_cm=de,
                                   rotation_difference_deg=ae,success_match=success==expected_success))
            cohorts=[b['partition']]+(['PRIMARY'] if b['partition']!='CAL' else [])
            for cohort in cohorts:
                counts[(cohort,arm)]['N']+=1
                counts[(cohort,arm)]['success']+=int(success is True)
                if distance is not None and angle is not None:values[(cohort,arm)].append((distance,angle))
    summaries=[]
    for key,c in sorted(counts.items()):
        array=np.asarray(values[key])
        summaries.append(dict(population=key[0],arm=key[1],**c,available_n=len(array),
            translation_mean_cm=float(array[:,0].mean()) if len(array) else None,
            rotation_mean_deg=float(array[:,1].mean()) if len(array) else None,
            translation_sample_variance_cm2=float(array[:,0].var(ddof=1)) if len(array)>1 else None,
            rotation_sample_variance_deg2=float(array[:,1].var(ddof=1)) if len(array)>1 else None))
    write(root/'INDEPENDENT_REAGGREGATION.json',dict(rows=238,arm_rows=len(table),
        success_all_match=not errors,discrepancies=errors,summary=summaries,
        primary_metric_helper_imports=0,norm_formula='sqrt(sum squared centre difference)',
        rotation_formula='minimum norm of SciPy rotation vector over fixed physical I/Y180',
        tolerance=dict(translation_cm=1e-9,rotation_deg=1e-6),
        no_GT_inference=True,no_new_fits=True,source_sha256=sha(__file__),
        reference_artifact_sha256=sha(scored/'FIXED_REFERENCES.json')))
    record_stage(root,'independent_Rt_reaggregation',start,worker_cpu_seconds=time.process_time()-cpu,new_pose_calls=0)
    if errors:raise AssertionError('Independent metric discrepancies retained in receipt')


def verify_summary(folder,score_folder):
    """Compare independently aggregated counts/means/variances, no new scoring."""
    root,scored=Path(folder),Path(score_folder)
    seal=read(scored/'SCORE_SEAL.json')
    if sha(scored/'RESULTS.json')!=seal['artifacts']['RESULTS.json']:
        raise ValueError('Original summary bytes changed')
    independent=read(root/'INDEPENDENT_REAGGREGATION.json')
    if not independent['success_all_match']:raise ValueError('Independent row errors unresolved')
    target=read(scored/'RESULTS.json')['populations']
    comparisons=[]
    for value in independent['summary']:
        actual=target[value['population']]['arms'][value['arm']]
        checks=dict(N=value['N']==actual['denominator'],success=value['success']==actual['successes_le5'])
        differences={}
        for name,unit,epsilon in [('translation','cm',1e-9),('rotation','deg',1e-6)]:
            original=actual['errors'][name+'_'+unit]
            checks[name+'_available_N']=value['available_n']==original['n']
            for suffix,key in [('mean','mean'),('sample_variance','sample_variance')]:
                observed=value[name+'_'+suffix+'_'+(unit+'2' if suffix=='sample_variance' else unit)]
                expected=original[key]
                difference=abs(observed-expected) if observed is not None and expected is not None else 0.
                # Fixed row-error tolerances propagated through variance. This
                # is a numerical bound, never a statistical equivalence margin.
                n=original['n']
                bound=epsilon if suffix=='mean' else (4*original['max']*epsilon+4*epsilon**2)*n/max(n-1,1)
                checks[name+'_'+suffix]=(observed is None)==(expected is None) and difference<=bound
                differences[name+'_'+suffix]=difference
        comparisons.append(dict(population=value['population'],arm=value['arm'],checks=checks,differences=differences))
    write(root/'INDEPENDENT_SUMMARY_CHECK.json',dict(arm_cohort_rows=len(comparisons),
        all_pass=all(all(r['checks'].values()) for r in comparisons),rows=comparisons,
        independent_metric_receipt_sha256=sha(root/'INDEPENDENT_REAGGREGATION.json'),
        no_new_pose_scoring_or_inference=True,source_sha256=sha(__file__)))
    if not all(all(r['checks'].values()) for r in comparisons):raise AssertionError('Independent aggregate mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--score-folder',required=True)
    p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if a.summary_only:verify_summary(a.folder,a.score_folder)
    else:run(a.folder,a.score_folder)
